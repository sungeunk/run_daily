#!/usr/bin/env python3
"""Streamlit viewer for the daily benchmark DuckDB.

Run with::

    streamlit run daily/viewer/app.py -- --db daily/viewer/bench.duckdb

The DB is built by ``python -m viewer.ingest.cli``. This app is a pure
read-only consumer.

Tabs
----
1. Dashboard    — latest daily_CB run review from summary/report/raw log.
2. Excel Paste  — wide matrix for a fixed display profile, selected runs
                  become columns.
3. Regression   — trend comparison table plus one selected series chart.
4. Geomean      — geometric-mean trend across a bucket (machine-wide health).
5. Noise        — per-series coefficient of variation. Useful for iGPU
                  diagnostics where fluctuation is inherent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `viewer.queries` importable when launched via `streamlit run`.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from viewer import queries as q  # noqa: E402

# ---------------------------------------------------------------------------
# Config / connection
# ---------------------------------------------------------------------------

DEFAULT_DB = _HERE / "bench.duckdb"


def _resolve_db_path() -> Path:
    """Streamlit consumes its own CLI flags, so we read our DB path from env
    or from ``-- --db <path>`` (pytest-style)."""
    # streamlit forwards trailing args after ``--`` as sys.argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if args.db:
        return args.db
    env = os.environ.get("DAILY_DB")
    if env:
        return Path(env)
    return DEFAULT_DB


DB = _resolve_db_path()


# ---------------------------------------------------------------------------
# Cached queries (streamlit caches by argument tuple, keyed on DB mtime)
# ---------------------------------------------------------------------------

def _db_version() -> float:
    try:
        return DB.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def cached_machines(_v: float) -> list[str]:
    return q.list_machines(DB)


@st.cache_data(show_spinner=False)
def cached_runs(machine: str, _v: float) -> pd.DataFrame:
    return q.list_runs(DB, machine)


@st.cache_data(show_spinner=False)
def cached_excel(run_ids: tuple[str, ...], profile: str, _v: float) -> pd.DataFrame:
    return q.build_excel_matrix(DB, list(run_ids), profile)


@st.cache_data(show_spinner=False)
def cached_extra_rows(run_ids: tuple[str, ...], profile: str, _v: float) -> pd.DataFrame:
    return q.extra_rows(DB, list(run_ids), profile)


@st.cache_data(show_spinner=False)
def cached_profiles(_v: float) -> list[str]:
    return q.list_profiles(DB)


@st.cache_data(show_spinner=False)
def cached_series(machine: str, model: str, precision: str,
                  in_token: int, out_token: int, exec_mode: str,
                  days: int, purpose_filter: str, _v: float) -> pd.DataFrame:
    return q.series_history(DB, machine, model, precision,
                            in_token, out_token, exec_mode, days=days,
                            purpose_filter=purpose_filter)


@st.cache_data(show_spinner=False)
def cached_noise(machine: str, days: int, _v: float) -> pd.DataFrame:
    return q.noise_summary(DB, machine=machine, days=days)


@st.cache_data(show_spinner=False)
def cached_geomean(machine: str, exec_mode: str, in_bucket: str | None,
                   out_bucket: str | None, days: int,
                   purpose_filter: str, _v: float) -> pd.DataFrame:
    return q.geomean_trend(DB, machine, exec_mode=exec_mode,
                           in_bucket=in_bucket, out_bucket=out_bucket,
                           days=days, purpose_filter=purpose_filter)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PERF_COL_NON_VALUE = {"model", "precision", "in_spec", "out_spec",
                      "exec_mode", "label"}


def _worse_direction(unit: str | None) -> int:
    """+1 if up = worse (latency), -1 if up = better (throughput)."""
    if unit is None:
        return +1
    return +1 if unit in ("ms", "s", "%") else -1


def _stable_y_range(values: pd.Series, min_relative_span: float = 0.10) -> list[float] | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None

    low = float(clean.min())
    high = float(clean.max())
    midpoint = (low + high) / 2.0
    actual_span = high - low
    min_span = max(abs(midpoint) * min_relative_span, 1e-9)
    span = max(actual_span * 1.10, min_span)
    lower = midpoint - span / 2.0
    upper = midpoint + span / 2.0
    if low >= 0 and lower < 0:
        lower = 0.0
    return [lower, upper]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Canonical daily rigs. Kept here (not in DB) because the list is small,
# human-curated, and tied to the ops team's owned machines rather than
# whatever happens to show up under /var/www/html/daily/ (old, one-off
# folders often linger there).
DAILY_MACHINES = (
    "dg2alderlake",
    "MTL-01",
    "ARLH-01",
    "BMG-02",
    "LNL-03",
    "LNL-04",
    "DUT4580PTLH",
    "DUT6047BMGFRD",
)

DEFAULT_RUN_FILTER = "daily_CB timer"


def _sidebar() -> dict:
    st.sidebar.header("Settings")
    v = _db_version()
    if v == 0.0:
        st.sidebar.error(f"DB not found at {DB}")
        st.stop()
    st.sidebar.caption(f"DB: `{DB}`")
    all_machines = cached_machines(v)
    if not all_machines:
        st.sidebar.warning("No runs in DB yet — run `viewer.ingest.cli` first.")
        st.stop()

    daily_only = st.sidebar.checkbox("Daily machines only", value=True,
                                     help="Filter to the canonical daily rig set.")
    if daily_only:
        machines = [m for m in all_machines if m in DAILY_MACHINES]
        # Fall back to the full list if the DB has none of the canonical
        # rigs yet — avoids an empty dropdown on fresh installs.
        if not machines:
            st.sidebar.caption("_No canonical daily rigs in DB — showing all._")
            machines = all_machines
    else:
        machines = all_machines
    machine = st.sidebar.selectbox("Machine", machines)

    profile_options = cached_profiles(v) or ["default"]
    profile = st.sidebar.selectbox("Display profile", profile_options)

    st.sidebar.divider()
    st.sidebar.subheader("Regression thresholds")
    z = st.sidebar.slider("z-score |threshold|", 1.0, 6.0, 3.0, 0.5,
                          help="Robust z based on MAD.")
    pct = st.sidebar.slider("% diff threshold", 0.01, 0.50, 0.05, 0.01)
    cv = st.sidebar.slider("Noisy CV threshold", 0.02, 0.50, 0.10, 0.01,
                           help="CV = MAD/median. Above this, series is "
                                "treated as inherently noisy.")

    days = st.sidebar.slider("Trend history (days)", 7, 60, 30)
    return dict(v=v, machine=machine, profile=profile,
                z=z, pct=pct, cv=cv, days=days)


# ---------------------------------------------------------------------------
# Run artifact review helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return _HERE.parent.parent


def _existing_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        root = _repo_root()
        candidates.extend([root / path, root / "output" / path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_json_file(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_text_file(path: Path | None, max_chars: int = 120_000) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


# Artifact resolvers: `source_path` is the authoritative anchor (always
# absolute, written by both loaders). Siblings are derived from its stem.
# `report_file` stores the source filename, NOT a `.report` path — do not
# feed it through `_existing_path` expecting a text report.
_SOURCE_SUFFIXES = (".summary.json", ".pickle")


def _source_path_for_run(run: pd.Series) -> Path | None:
    return _existing_path(run.get("source_path"))


def _source_stem(source_path: Path) -> str | None:
    for suffix in _SOURCE_SUFFIXES:
        if source_path.name.endswith(suffix):
            return source_path.name.removesuffix(suffix)
    return None


def _sibling(run: pd.Series, suffix: str) -> Path | None:
    source_path = _source_path_for_run(run)
    if source_path is None:
        return None
    stem = _source_stem(source_path)
    if stem is None:
        return None
    candidate = source_path.with_name(stem + suffix)
    return candidate if candidate.exists() else None


def _summary_path_for_run(run: pd.Series) -> Path | None:
    """Return the `.summary.json` path, which exists for new-format runs and
    occasionally alongside old-format pickles."""
    if run.get("source_format") == "new":
        return _source_path_for_run(run)
    return _sibling(run, ".summary.json")


def _report_path_for_run(run: pd.Series) -> Path | None:
    """Return the `.report` text path. Same stem for both formats."""
    return _sibling(run, ".report")


def _pytest_json_path_for_run(run: pd.Series) -> Path | None:
    """Return the sibling `.pytest.json` (raw pytest-json-report output)."""
    return _sibling(run, ".pytest.json")


def _rawlog_path_for_run(run: pd.Series) -> Path | None:
    """Return the `.raw` pytest stdout/stderr text captured at run time."""
    return _existing_path(run.get("rawlog_path"))


def _metric_from_user_properties(test: dict, key: str) -> object | None:
    """Look up ``key`` in the latest ``metrics`` user_property.

    pytest-json-report serialises ``user_properties`` as either a list of
    ``{name: value}`` dicts (>=1.5) or ``[name, value]`` pairs. Mirrors
    ``daily/report/builder.py:_extract_metrics`` — last metrics entry wins.
    """
    for prop in reversed(test.get("user_properties", []) or []):
        metrics: object | None = None
        if isinstance(prop, dict):
            metrics = prop.get("metrics")
        elif isinstance(prop, (list, tuple)) and len(prop) == 2 and prop[0] == "metrics":
            metrics = prop[1]
        if isinstance(metrics, dict):
            if key in metrics:
                return metrics[key]
            return None
    fallback = test.get("metrics")
    if isinstance(fallback, dict):
        return fallback.get(key)
    return None


def _shorten(text: object, limit: int = 500) -> str:
    if text is None:
        return ""
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[:limit - 3] + "..."


def _classify_failure(text: str) -> str:
    lowered = text.lower()
    if "no such file or directory" in lowered or "could not open the file" in lowered:
        if "model" in lowered or ".xml" in lowered:
            return "missing model/artifact path"
        if "openvino.genai" in lowered or "benchmark.py" in lowered:
            return "missing tool/script path"
        return "missing file path"
    if "modulenotfounderror" in lowered:
        return "missing python package"
    if "spawn failed" in lowered:
        return "missing executable"
    if "attributeerror" in lowered and "none" in lowered:
        return "invalid cached/model state"
    if "returncode" in lowered:
        return "command returned non-zero"
    return "test failure"


def _extract_failures(summary: dict, pytest_log: dict) -> pd.DataFrame:
    tests = pytest_log.get("tests") or summary.get("tests") or []
    rows = []
    for test in tests:
        outcome = test.get("outcome")
        if outcome not in {"failed", "error"}:
            continue
        call = test.get("call") or {}
        crash = call.get("crash") or {}
        message = (crash.get("message") or test.get("failure") or
                   test.get("longrepr") or _metric_from_user_properties(test, "output") or "")
        command = _metric_from_user_properties(test, "cmd")
        returncode = _metric_from_user_properties(test, "returncode")
        combined = f"{message} {command or ''}"
        rows.append({
            "test": test.get("nodeid", ""),
            "outcome": outcome,
            "cause": _classify_failure(combined),
            "returncode": returncode,
            "message": _shorten(message),
            "command": _shorten(command, 220),
        })
    return pd.DataFrame(rows)


def _latest_daily_run(runs: pd.DataFrame) -> pd.Series | None:
    if runs.empty:
        return None
    text = (runs["purpose"].fillna("") + " " + runs["description"].fillna(""))
    mask = text.str.contains(DEFAULT_RUN_FILTER, case=False, regex=False)
    if not mask.any():
        mask = text.str.contains("daily_CB", case=False, regex=False)
    view = runs[mask]
    if view.empty:
        return None
    return view.iloc[0]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _tab_dashboard(cfg: dict) -> None:
    st.subheader("Dashboard — latest daily_CB review")
    runs = cached_runs(cfg["machine"], cfg["v"])
    run = _latest_daily_run(runs)
    if run is None:
        st.info(f"No daily_CB run found for {cfg['machine']}.")
        return

    summary_path = _summary_path_for_run(run)
    report_path = _report_path_for_run(run)
    pytest_json_path = _pytest_json_path_for_run(run)
    rawlog_path = _rawlog_path_for_run(run)
    summary = _read_json_file(summary_path)
    pytest_log = _read_json_file(pytest_json_path)
    report_text = _read_text_file(report_path)
    rawlog_text = _read_text_file(rawlog_path)
    totals = summary.get("totals") or pytest_log.get("summary") or {}
    failures = _extract_failures(summary, pytest_log)

    total = int(totals.get("total") or 0)
    passed = int(totals.get("passed") or 0)
    failed = int(totals.get("failed") or 0)
    errors = int(totals.get("error") or totals.get("errors") or 0)
    skipped = int(totals.get("skipped") or 0)
    duration = summary.get("duration_sec") or pytest_log.get("duration")

    st.markdown(
        f"**{run['stamp']}** · {run['ww'] or ''} · "
        f"{run['ov_version'] or ''} · {run['purpose'] or run['description'] or ''}"
    )
    cols = st.columns(6)
    cols[0].metric("Total", total)
    cols[1].metric("Passed", passed)
    cols[2].metric("Failed", failed)
    cols[3].metric("Error", errors)
    cols[4].metric("Skipped", skipped)
    cols[5].metric("Duration", f"{float(duration):.0f}s" if duration else "-")

    artifacts = {
        "summary":     str(summary_path)     if summary_path     else "missing",
        "report":      str(report_path)      if report_path      else "missing",
        "pytest_json": str(pytest_json_path) if pytest_json_path else "missing",
        "raw_log":     str(rawlog_path)      if rawlog_path      else "missing",
    }
    missing = [name for name, path in artifacts.items() if path == "missing"]
    if failed or errors:
        st.error(f"Run failed: {failed} failed, {errors} error.")
    elif total and passed == total:
        st.success("Run completed successfully.")
    else:
        st.warning("Run status is incomplete or summary data is missing.")
    if missing:
        st.warning("Missing artifacts: " + ", ".join(missing))

    if failures.empty:
        st.markdown("### Failure analysis")
        st.caption("No failed/error tests found in summary or raw log.")
    else:
        cause_counts = Counter(failures["cause"])
        st.markdown("### Failure analysis")
        st.dataframe(
            pd.DataFrame(cause_counts.items(), columns=["cause", "count"])
              .sort_values("count", ascending=False),
            width="stretch",
            hide_index=True,
        )
        st.markdown("### Failed tests")
        st.dataframe(failures, width="stretch", hide_index=True)

    with st.expander("Run artifacts"):
        st.json(artifacts)
    if report_text:
        with st.expander("Report"):
            st.code(report_text, language="text")
    if pytest_log:
        with st.expander("pytest-json-report"):
            st.json({
                "exitcode": pytest_log.get("exitcode"),
                "summary": pytest_log.get("summary"),
                "failed_tests": failures.to_dict(orient="records"),
            })
    if rawlog_text:
        with st.expander("Raw pytest log"):
            st.code(rawlog_text, language="text")


def _tab_excel(cfg: dict) -> None:
    st.subheader("Excel Paste")
    runs = cached_runs(cfg["machine"], cfg["v"])
    if runs.empty:
        st.info("No runs for this machine.")
        return

    filt = st.text_input("Filter by purpose/description",
                         value=DEFAULT_RUN_FILTER, key="excel_filter")
    view = runs
    if filt:
        mask = (view["purpose"].fillna("").str.contains(filt, case=False) |
                view["description"].fillna("").str.contains(filt, case=False))
        view = view[mask]

    st.caption(f"{len(view)} runs")
    event = st.dataframe(
        view[["stamp", "ww", "ov_version", "purpose", "source_format"]],
        width="stretch",
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="excel_run_table",
    )
    sel = event.selection.rows if event and event.selection else []
    if not sel:
        st.info("Select one or more runs above to build the paste block.")
        return
    run_ids = tuple(view.iloc[sel]["run_id"].tolist())

    matrix = cached_excel(run_ids, cfg["profile"], cfg["v"])
    if matrix.empty:
        st.warning("Display profile produced no rows.")
        return

    st.markdown("**Paste block**  (tab-separated; headers: OV version / workweek / stamp)")
    meta = runs.set_index("run_id").loc[list(run_ids)]
    stamps = [c for c in matrix.columns if c not in PERF_COL_NON_VALUE]
    header_rows = [
        "\t".join(str(meta.loc[rid, "ov_version"] or "") for rid in run_ids),
        "\t".join(str(meta.loc[rid, "ww"] or "") for rid in run_ids),
        "",
        "\t".join(stamps),
    ]
    data_text = matrix[stamps].to_csv(sep="\t", index=False, header=False,
                                      float_format="%.2f")
    paste = "\n".join(header_rows) + "\n" + data_text
    st.text_area("Copy & paste into Excel", value=paste, height=260)

    st.markdown("**Matrix preview**")
    st.dataframe(matrix, width="stretch", hide_index=True)

    extras = cached_extra_rows(run_ids, cfg["profile"], cfg["v"])
    if not extras.empty:
        with st.expander(f"⚠ {len(extras)} perf rows not covered by display profile"):
            st.dataframe(extras, width="stretch", hide_index=True)


@st.cache_data(show_spinner=False)
def cached_trend_regressions(machine: str, recent_days: int,
                             baseline_days: int, purpose_filter: str,
                             _v: float) -> pd.DataFrame:
    return q.trend_regressions(DB, machine,
                               recent_days=recent_days,
                               baseline_days=baseline_days,
                               purpose_filter=purpose_filter)


def _series_label(row: pd.Series) -> str:
    return (f"{row['model']} | {row['precision']} | "
            f"in={row['in_token']} out={row['out_token']} | "
            f"{row['exec_mode']} [{row['unit']}]")


def _tab_regression(cfg: dict) -> None:
    """Trend + regression in a single view.

    The question this tab answers: 'is this series trending slower lately,
    compared to its own recent past?' — not 'did today's single point miss
    the band'. We compare the median of a recent window to the median of
    an older baseline window, so ingest noise and one-off outliers get
    averaged out.
    """
    st.subheader("Regression — trend comparison")

    c1, c2 = st.columns(2)
    recent_days = c1.slider("Recent window (days)", 3, 21, 7, 1,
                            help="Last N days worth of points; their "
                                 "median represents 'now'.")
    baseline_days = c2.slider("Baseline window (days)", 7, 60, 21, 1,
                              help="The N days before the recent window; "
                                   "their median is the comparison point.")

    df = cached_trend_regressions(cfg["machine"], recent_days,
                                  baseline_days, DEFAULT_RUN_FILTER, cfg["v"])
    if df.empty:
        st.info("No data for this machine / window.")
        return

    valid = df[df["status"] == "ok"].copy()
    valid["severity"] = pd.concat([
        valid["worsening_pct"].fillna(0) / cfg["pct"],
        valid["worsening_z"].fillna(0) / cfg["z"],
    ], axis=1).max(axis=1)
    valid = valid.sort_values("severity", ascending=False).reset_index(drop=True)
    noisy_count = int((valid["recent_cv"].fillna(0) >= cfg["cv"]).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Series tracked", len(valid))
    # Worsening above either threshold, regardless of noise — surfaces the
    # strongest signals. The user should cross-check with the graph for
    # series that are flagged noisy.
    bad = valid[(valid["worsening_pct"].fillna(0) >= cfg["pct"]) |
                (valid["worsening_z"].fillna(0) >= cfg["z"])]
    m2.metric("Worsening ≥ threshold", len(bad))
    better = valid[(valid["worsening_pct"].fillna(0) <= -cfg["pct"]) |
                   (valid["worsening_z"].fillna(0) <= -cfg["z"])]
    m3.metric("Improving ≥ threshold", len(better))
    m4.metric("Noisy (recent CV high)", noisy_count)

    # Build a compact, sortable table.
    def _fmt(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "severity" not in out.columns:
            out["severity"] = pd.concat([
                out["worsening_pct"].fillna(0) / cfg["pct"],
                out["worsening_z"].fillna(0) / cfg["z"],
            ], axis=1).max(axis=1)
        out["series"] = out.apply(_series_label, axis=1)
        out["worsening_%"] = (out["worsening_pct"] * 100).round(2)
        out["worsening_z"] = out["worsening_z"].round(2)
        out["severity"] = out["severity"].round(2)
        out["recent_cv_%"] = (out["recent_cv"] * 100).round(2)
        # Display-only columns that carry the unit suffix so SD pipelines
        # (seconds) aren't mistaken for LLM latencies (ms) at a glance. The
        # raw numeric recent_median / baseline_median columns remain intact
        # below because the trend plot and caption need them as floats.
        def _with_unit(val: float | None, unit: str | None) -> str:
            if val is None or pd.isna(val):
                return ""
            return f"{val:.3f} {unit or ''}".rstrip()
        out["recent"] = [_with_unit(v, u) for v, u in
                         zip(out["recent_median"], out["unit"].fillna(""))]
        out["baseline"] = [_with_unit(v, u) for v, u in
                           zip(out["baseline_median"], out["unit"].fillna(""))]
        return out[["series", "severity", "worsening_%", "worsening_z", "recent", "baseline",
                    "recent_n", "baseline_n", "recent_cv_%",
                    "direction", "status",
                    # keep these for downstream selection, hidden via
                    # column_config below
                    "model", "precision", "in_token", "out_token",
                    "exec_mode", "unit",
                    # raw numerics for the plot/caption code paths
                    "recent_median", "baseline_median"]]

    table = _fmt(valid)
    insufficient = df[df["status"] == "insufficient_data"]

    st.markdown("### Series ranked by worsening %")
    st.caption("Positive values mean the recent window is slower / worse "
               "than the baseline. Click a row to plot it below.")
    event = st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="regression_table",
        column_config={
            # Hide columns we carry for row-selection / plotting only.
            "model":           st.column_config.Column(width="small", disabled=True),
            "precision":       None,
            "in_token":        None,
            "out_token":       None,
            "exec_mode":       None,
            "unit":            None,
            "recent_median":   None,
            "baseline_median": None,
        },
    )

    # Default to the worst series (row 0) so the user sees something useful
    # without clicking.
    sel_rows = event.selection.rows if event and event.selection else []
    sel_idx = sel_rows[0] if sel_rows else 0
    if sel_idx >= len(table):
        return
    row = table.iloc[sel_idx]

    st.markdown(
        f"### Trend — {row['model']} / {row['precision']} / "
        f"in={row['in_token']} out={row['out_token']} / {row['exec_mode']} "
        f"[{row['unit']}]"
    )

    hist = cached_series(
        cfg["machine"], row["model"], row["precision"],
        int(row["in_token"]), int(row["out_token"]), row["exec_mode"],
        cfg["days"], DEFAULT_RUN_FILTER, cfg["v"])
    if hist.empty:
        st.info("No history for this series in the selected window.")
        return

    unit = row["unit"] or ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["ts"], y=hist["value"], mode="lines+markers",
        name="value",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>value=%{y:.2f} "
                      + unit + "<br>%{text}",
        text=[f"ov={v}" for v in hist["ov_version"].fillna("")],
    ))

    # Horizontal reference lines drawn as Scatter traces (not add_hline)
    # so they show up as proper legend entries. Each spans the full x range
    # with a constant y.
    x_span = [hist["ts"].min(), hist["ts"].max()]
    if pd.notna(row["baseline_median"]):
        fig.add_trace(go.Scatter(
            x=x_span,
            y=[float(row["baseline_median"])] * 2,
            mode="lines",
            line=dict(dash="dot", color="#1f77b4", width=2),
            name=f"baseline median ({baseline_days}d prior)",
            hovertemplate=f"baseline median = {row['baseline_median']:.2f} "
                          + unit + "<extra></extra>",
        ))
    if pd.notna(row["recent_median"]):
        fig.add_trace(go.Scatter(
            x=x_span,
            y=[float(row["recent_median"])] * 2,
            mode="lines",
            line=dict(dash="dash", color="firebrick", width=2),
            name=f"recent median (last {recent_days}d)",
            hovertemplate=f"recent median = {row['recent_median']:.2f} "
                          + unit + "<extra></extra>",
        ))

    # Rolling ±2σ band around the series' own rolling median. Bumped the
    # opacity and used a stronger fill colour so the envelope is actually
    # visible on a bright monitor (the 0.12 default washed out).
    if hist["win_median"].notna().any():
        band_hi = hist["win_median"] + 2 * hist["win_sigma"]
        band_lo = hist["win_median"] - 2 * hist["win_sigma"]
        fig.add_trace(go.Scatter(
            x=pd.concat([hist["ts"], hist["ts"][::-1]]),
            y=pd.concat([band_hi, band_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.28)",
            mode="none",
            name="±2σ band (rolling)",
            showlegend=True,
            hoverinfo="skip",
        ))

    y_values = [hist["value"]]
    if pd.notna(row["baseline_median"]):
        y_values.append(pd.Series([float(row["baseline_median"])]))
    if pd.notna(row["recent_median"]):
        y_values.append(pd.Series([float(row["recent_median"])]))
    if hist["win_median"].notna().any():
        y_values.extend([band_hi, band_lo])
    yaxis = {}
    y_range = _stable_y_range(pd.concat(y_values, ignore_index=True))
    if y_range is not None:
        yaxis["range"] = y_range

    fig.update_layout(
        height=450, hovermode="x unified",
        xaxis_title="timestamp",
        yaxis_title=f"value [{unit}]" if unit else "value",
        yaxis=yaxis,
        xaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig)

    st.caption(
        f"Recent median = {row['recent_median']:.3f} {unit} "
        f"(n={int(row['recent_n'])}) vs baseline median = "
        f"{row['baseline_median']:.3f} {unit} (n={int(row['baseline_n'])}). "
        f"Worsening = {row['worsening_%']:+.2f}%, "
        f"z={row['worsening_z']:+.2f}. "
        f"Direction: {row['direction']}. "
        f"Recent CV = {row['recent_cv_%']:.2f}% "
        + ("(noisy — interpret with care)."
           if row["recent_cv_%"] >= cfg["cv"] * 100
           else "(stable).")
    )

    if not insufficient.empty:
        with st.expander(f"{len(insufficient)} series with insufficient data"):
            st.caption("Fewer points than min_recent/min_baseline in the "
                       "chosen windows.")
            st.dataframe(_fmt(insufficient), width="stretch",
                         hide_index=True)


def _tab_geomean(cfg: dict) -> None:
    st.subheader("Geomean trend — machine-wide health")
    exec_mode = st.selectbox("Exec mode", ["1st", "2nd", "pipeline", "tps"],
                             index=1)
    cols = st.columns(3)
    in_bucket = cols[0].selectbox("in bucket",
                                  [None, "short", "long", "0"],
                                  format_func=lambda v: v or "(any)")
    out_bucket = cols[1].selectbox("out bucket",
                                   [None, "short", "long", "0"],
                                   format_func=lambda v: v or "(any)")
    days = cols[2].number_input("days", min_value=7, max_value=60,
                                value=cfg["days"])

    df = cached_geomean(cfg["machine"], exec_mode, in_bucket, out_bucket,
                        int(days), DEFAULT_RUN_FILTER, cfg["v"])
    if df.empty:
        st.info("No data for this filter.")
        return

    # Baseline = median of the *geomean* series, robust band using MAD.
    median = df["geomean"].median()
    mad = (df["geomean"] - median).abs().median()
    sigma = 1.4826 * mad

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["geomean"], mode="lines+markers",
        name="geomean",
        text=[f"ov={v}<br>n={n}" for v, n in
              zip(df["ov_version"].fillna(""), df["n_samples"])],
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>geomean=%{y:.2f}<br>%{text}",
    ))
    if not math.isnan(sigma) and sigma > 0:
        fig.add_hline(y=median, line=dict(dash="dash"), annotation_text="median")
        fig.add_hrect(y0=median - 2 * sigma, y1=median + 2 * sigma,
                      line_width=0, fillcolor="LightBlue", opacity=0.2,
                      annotation_text="±2σ")
    fig.update_layout(height=450, hovermode="x unified",
                      xaxis_title="timestamp", yaxis_title="geomean")
    st.plotly_chart(fig)

    # Alert banner when the latest point is outside the band.
    if len(df) >= 5 and sigma > 0:
        latest = df.iloc[-1]["geomean"]
        z = (latest - median) / sigma
        pct = (latest - median) / median * 100 if median else 0
        sign = -1 if exec_mode == "tps" else 1
        worsening_z = sign * z
        worsening_pct = sign * pct
        if abs(worsening_z) >= cfg["z"] and abs(worsening_pct) >= cfg["pct"] * 100:
            direction = "worse" if worsening_z > 0 else "better"
            st.error(f"⚠ Latest geomean is {direction} by "
                     f"z={worsening_z:+.2f}, {worsening_pct:+.1f}%.")
        else:
            st.success("Latest geomean within band "
                       f"(z={worsening_z:+.2f}, {worsening_pct:+.1f}%).")

    st.dataframe(df[["ts", "ww", "ov_version", "geomean", "n_samples"]]
                 .tail(30), width="stretch", hide_index=True)


def _tab_noise(cfg: dict) -> None:
    st.subheader("Noise diagnostics")
    st.caption("Series with high coefficient of variation (CV = σ / median) "
               "over the selected window. iGPU results often live here.")
    days = st.number_input("Window (days)", min_value=7, max_value=60,
                           value=cfg["days"], key="noise_days")
    df = cached_noise(cfg["machine"], int(days), cfg["v"])
    if df.empty:
        st.info("No data.")
        return
    df["cv_pct"] = (df["cv"] * 100).round(2)
    df["median_value"] = df["median_value"].round(3)
    df["std_value"] = df["std_value"].round(3)
    st.dataframe(
        df[["model", "precision", "in_token", "out_token", "exec_mode",
            "unit", "n", "median_value", "std_value", "cv_pct"]],
        width="stretch", hide_index=True,
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(layout="wide", page_title="Daily LLM Viewer")
    pd.set_option("display.float_format", "{:.2f}".format)
    st.title("Daily LLM Benchmark Viewer")

    cfg = _sidebar()

    tabs = st.tabs(["Dashboard", "Excel", "Regression", "Geomean", "Noise"])
    with tabs[0]:
        _tab_dashboard(cfg)
    with tabs[1]:
        _tab_excel(cfg)
    with tabs[2]:
        _tab_regression(cfg)
    with tabs[3]:
        _tab_geomean(cfg)
    with tabs[4]:
        _tab_noise(cfg)


if __name__ == "__main__":
    main()
