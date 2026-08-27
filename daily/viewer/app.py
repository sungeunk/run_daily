#!/usr/bin/env python3
"""Streamlit viewer for the daily benchmark DuckDB.

Run with::

    streamlit run daily/viewer/app.py -- --db daily/viewer/bench.duckdb

The DB is built by ``python -m viewer.ingest.cli``. The sidebar can refresh
the configured daily DB by running the local ingestion script.

Tabs
----
1. Dashboard    — latest daily_CB run review from summary/report/raw log.
2. Excel Paste  — wide matrix for a fixed display profile, selected runs
                  become columns.
3. Regression   — trend comparison table plus one selected series chart.
4. Geomean      — geometric-mean trend across a bucket (machine-wide health).
5. Noise        — per-series coefficient of variation. Useful for iGPU
                  diagnostics where fluctuation is inherent.
6. Functional   — functional issue history and per-run health summary.
7. Compare      — run-to-run direct comparison at the series level.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
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
INGEST_SCRIPT = Path("/var/www/html/daily2/ingest_db.sh")


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


def _refresh_database() -> None:
    """Rebuild the daily benchmark DB and invalidate cached query results."""
    if not INGEST_SCRIPT.is_file() or not os.access(INGEST_SCRIPT, os.X_OK):
        st.error(f"Ingestion script is unavailable: {INGEST_SCRIPT}")
        return

    progress = st.progress(5, text="Starting database refresh...")
    with st.status("Refreshing database...", expanded=True) as status:
        st.write("Running ingestion script")
        progress.progress(25, text="Ingesting daily benchmark artifacts...")
        result = subprocess.run(
            [str(INGEST_SCRIPT)],
            cwd=INGEST_SCRIPT.parent,
            capture_output=True,
            check=False,
            text=True,
        )

        if result.returncode != 0:
            status.update(label="Database refresh failed", state="error")
            progress.empty()
            st.error(f"Ingestion failed with exit code {result.returncode}.")
            output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            if output:
                st.code(output, language="text")
            return

        progress.progress(90, text="Reloading cached data...")
        st.cache_data.clear()
        status.update(label="Database refresh completed", state="complete")

    progress.progress(100, text="Database is up to date")
    st.session_state["db_refresh_message"] = "Database refresh completed."
    st.rerun()


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
def cached_success_counts(run_ids: tuple[str, ...], _v: float) -> dict[str, int]:
    return q.success_counts(DB, list(run_ids))


@st.cache_data(show_spinner=False)
def cached_legacy_geomean_summary(run_ids: tuple[str, ...], _v: float) -> pd.DataFrame:
    return q.legacy_geomean_summary(DB, list(run_ids))


@st.cache_data(show_spinner=False)
def cached_profiles(_v: float) -> list[str]:
    return q.list_profiles(DB)



# --- count-based cohort helpers -------------------------------------------
# Every analysis tab resolves its scope through these so the run set, run kind
# and model selection stay consistent across the app.

@st.cache_data(show_spinner=False)
def cached_run_kinds(machine: str, _v: float) -> list[str]:
    return q.list_run_kinds(DB, machine)


@st.cache_data(show_spinner=False)
def cached_models(machine: str, run_kinds: tuple[str, ...],
                  _v: float) -> list[str]:
    return q.list_models(DB, machine, run_kinds=run_kinds)


@st.cache_data(show_spinner=False)
def cached_cohort(machine: str, limit: int, run_kinds: tuple[str, ...],
                  include_short_run: bool, _v: float) -> pd.DataFrame:
    return q.recent_runs(DB, machine, limit=limit, run_kinds=run_kinds,
                         include_short_run=include_short_run)


@st.cache_data(show_spinner=False)
def cached_series_trend(machine: str, recent_n: int, history_n: int,
                        run_kinds: tuple[str, ...], models: tuple[str, ...],
                        include_short_run: bool, _v: float) -> pd.DataFrame:
    return q.series_trend(DB, machine, recent_runs_n=recent_n,
                          history_runs_n=history_n, run_kinds=run_kinds,
                          models=models or None,
                          include_short_run=include_short_run)


@st.cache_data(show_spinner=False)
def cached_series_runs(machine: str, model: str, precision: str,
                       in_token: int, out_token: int, exec_mode: str,
                       runs_n: int, run_kinds: tuple[str, ...],
                       include_short_run: bool, _v: float) -> pd.DataFrame:
    return q.series_history_for_runs(DB, machine, model=model,
                                     precision=precision, in_token=in_token,
                                     out_token=out_token, exec_mode=exec_mode,
                                     runs_n=runs_n, run_kinds=run_kinds,
                                     include_short_run=include_short_run)


@st.cache_data(show_spinner=False)
def cached_geomean_runs(run_ids: tuple[str, ...], models: tuple[str, ...],
                        exec_modes: tuple[str, ...], _v: float) -> pd.DataFrame:
    return q.geomean_for_runs(DB, run_ids, models=models or None,
                              exec_modes=exec_modes or None)


@st.cache_data(show_spinner=False)
def cached_machine_health(run_ids: tuple[str, ...], _v: float) -> pd.DataFrame:
    return q.machine_health_for_runs(DB, run_ids)


@st.cache_data(show_spinner=False)
def cached_functional_issues(run_ids: tuple[str, ...], models: tuple[str, ...],
                             _v: float) -> pd.DataFrame:
    return q.functional_issues_for_runs(DB, run_ids, models=models or None)


@st.cache_data(show_spinner=False)
def cached_functional_summary(run_ids: tuple[str, ...], _v: float) -> pd.DataFrame:
    return q.functional_summary_for_runs(DB, run_ids)


@st.cache_data(show_spinner=False)
def cached_trend_compare(machine: str, run_a: str, run_b: str,
                         history_n: int, run_kinds: tuple[str, ...],
                         models: tuple[str, ...], _v: float) -> pd.DataFrame:
    return q.compare_runs_with_trend(DB, machine, run_a, run_b,
                                     history_runs_n=history_n,
                                     run_kinds=run_kinds,
                                     models=models or None)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PERF_COL_NON_VALUE = {"model", "precision", "in_spec", "out_spec",
                      "exec_mode", "unit", "label"}


# Plotly fits the y-axis to the data range, which turns a stable 1000 ms series
# that happens to wander between 998 and 1003 into a full-height sawtooth. The
# axis is therefore given a minimum span expressed as a percentage of the
# series level, so normal scatter renders as a flat line and only real moves
# fill the plot.
Y_SCALE_OPTIONS: dict[str, float | str] = {
    "±1%": 0.01,
    "±2%": 0.02,
    "±5%": 0.05,
    "±10%": 0.10,
    "±25%": 0.25,
    "Zero-based": "zero",
    "Auto (fit data)": "auto",
}
DEFAULT_Y_SCALE = "±10%"


def _stable_y_range(values: pd.Series,
                    scale: float | str = 0.10) -> list[float] | None:
    """Y-axis range honouring a minimum span so flat series look flat.

    ``scale`` is either a fraction interpreted as ± around the midpoint, or
    ``"auto"`` / ``"zero"``. The range always grows to fit the data.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None

    low = float(clean.min())
    high = float(clean.max())

    if scale == "auto":
        return None
    if scale == "zero":
        return [0.0, high * 1.1 if high > 0 else 1.0]

    midpoint = (low + high) / 2.0
    min_span = max(abs(midpoint) * float(scale) * 2.0, 1e-9)
    span = max((high - low) * 1.10, min_span)
    lower = midpoint - span / 2.0
    upper = midpoint + span / 2.0
    if low >= 0 and lower < 0:
        lower = 0.0
    return [lower, upper]


def _apply_y_scale(fig: go.Figure, cfg: dict, *series: pd.Series | list) -> None:
    """Apply the sidebar y-axis scale to ``fig`` using every plotted series."""
    parts = [pd.Series(s, dtype="float64") if not isinstance(s, pd.Series)
             else s for s in series if s is not None]
    if not parts:
        return
    y_range = _stable_y_range(pd.concat(parts, ignore_index=True),
                              cfg.get("y_scale", 0.10))
    if y_range is not None:
        fig.update_yaxes(range=y_range)


# Time flows right-to-left in the charts and top-to-bottom in the tables, so
# the newest run is always the first thing read. Queries still return oldest
# first because the trend maths needs chronological order.

def _newest_left(fig: go.Figure) -> None:
    """Put the newest run on the left of a time-ordered chart."""
    fig.update_xaxes(autorange="reversed")


def _newest_first(frame: pd.DataFrame) -> pd.DataFrame:
    """Reverse a chronologically ordered frame for display."""
    if frame.empty:
        return frame
    if "ts" in frame.columns:
        return frame.sort_values("ts", ascending=False)
    if "stamp" in frame.columns:
        return frame.sort_values("stamp", ascending=False)
    return frame.iloc[::-1]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Canonical daily rigs. Kept here (not in DB) because the list is small,
# human-curated, and tied to the ops team's owned machines rather than
# whatever happens to show up under /var/www/html/daily/ (old, one-off
# folders often linger there).
DAILY_MACHINES = (
    "PTLH-01",
    "PTLH-02",
    "ARLH-01",
    "LNL-03",
    "LNL-04",
    "MTL-01",
    "RAPTOR-ELLY",
    "BMG-02",
    "dg2alderlake",
)

DEFAULT_RUN_FILTER = "daily"


def _sidebar() -> dict:
    st.sidebar.markdown("##### Daily LLM Benchmark Viewer")
    st.sidebar.header("Settings")
    v = _db_version()
    if v == 0.0:
        st.sidebar.error(f"DB not found at {DB}")
        st.stop()
    st.sidebar.caption(f"DB: `{DB}`")
    if st.sidebar.button("Refresh database", width="stretch"):
        _refresh_database()
    if message := st.session_state.pop("db_refresh_message", None):
        st.sidebar.success(message)

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
    if len(profile_options) == 1:
        profile = profile_options[0]
        st.sidebar.caption(f"Display profile: `{profile}`")
    else:
        profile = st.sidebar.selectbox("Display profile", profile_options)

    st.sidebar.divider()
    st.sidebar.subheader("Analysis scope")
    # History depth is counted in runs, not days: benchmark cadence is
    # irregular, so a day window silently under-samples a rig that skipped a
    # night and over-samples one that ran twice.
    history_runs = st.sidebar.slider(
        "History (runs)", 3, 60, 10, 1,
        help="How many past runs each trend/regression view compares against.")
    recent_runs_n = st.sidebar.slider(
        "Recent window (runs)", 1, 10, 3, 1,
        help="Newest N runs treated as 'now'.")

    available_kinds = cached_run_kinds(machine, v) or list(q.RUN_KINDS)
    default_kinds = [k for k in available_kinds if k == "daily"] or available_kinds
    run_kinds = st.sidebar.multiselect(
        "Run kinds", available_kinds, default=default_kinds,
        help="Jenkins PR/test runs are excluded by default — they are not "
             "comparable to scheduled daily runs.")
    if not run_kinds:
        st.sidebar.caption("_No run kind selected — showing all._")
        run_kinds = available_kinds

    include_short_run = st.sidebar.checkbox(
        "Include short runs", value=False,
        help="Short runs measure fewer prompts and skew comparisons.")

    all_models = cached_models(machine, tuple(run_kinds), v)
    models = st.sidebar.multiselect("Models", all_models, default=[],
                                    help="Empty means all models.")

    st.sidebar.divider()
    st.sidebar.subheader("Regression thresholds")
    z = st.sidebar.slider("z-score |threshold|", 1.0, 6.0, 3.0, 0.5,
                          help="Robust z based on MAD.")
    pct = st.sidebar.slider("% diff threshold", 0.01, 0.50, 0.05, 0.01)
    cv = st.sidebar.slider("Noisy CV threshold", 0.02, 0.50, 0.10, 0.01,
                           help="CV = MAD/median. Above this, series is "
                                "treated as inherently noisy.")

    st.sidebar.divider()
    y_scale_label = st.sidebar.selectbox(
        "Chart y-axis range", list(Y_SCALE_OPTIONS),
        index=list(Y_SCALE_OPTIONS).index(DEFAULT_Y_SCALE),
        help="Minimum span the y-axis covers. Auto-fit makes a 998-1003 ms "
             "series fill the chart; a ±10% floor keeps it visibly flat.")

    return dict(v=v, machine=machine, profile=profile,
                daily_only=daily_only,
                z=z, pct=pct, cv=cv,
                y_scale=Y_SCALE_OPTIONS[y_scale_label],
                history_runs=history_runs, recent_runs=recent_runs_n,
                run_kinds=tuple(run_kinds), models=tuple(models),
                include_short_run=include_short_run)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _cohort(cfg: dict) -> pd.DataFrame:
    """Runs in scope for the current sidebar filters, oldest first."""
    return cached_cohort(cfg["machine"], cfg["history_runs"] + cfg["recent_runs"],
                         cfg["run_kinds"], cfg["include_short_run"], cfg["v"])


def _machine_state_by_run(cfg: dict, run_ids: tuple[str, ...]) -> tuple[dict, pd.DataFrame]:
    """Map run_id -> machine state label, plus the raw health frame."""
    health = cached_machine_health(run_ids, cfg["v"])
    if health.empty:
        return {}, health
    clock_baseline = health["avg_gpu_clock_mhz"].median()
    states = {
        str(row["run_id"]): q.classify_machine_state(row, clock_baseline)
        for _, row in health.iterrows()
    }
    return states, health


_STATE_ICON = {"stable": "🟢", "fluctuating": "🟡",
               "throttled": "🔴", "unknown": "⚪"}

# Artefacts are published at
# http://<relay>/daily2/<MACHINE>/<YYYY.MM>/daily.<stamp>.*
REPORT_BASE_URL = os.environ.get(
    "DAILY_REPORT_BASE_URL",
    "http://dg2raptorlake.ikor.intel.com/daily2").rstrip("/")


@st.cache_data(show_spinner=False)
def cached_machines_overview(machines: tuple[str, ...],
                             run_kinds: tuple[str, ...],
                             include_short_run: bool, history_runs: int,
                             _v: float) -> pd.DataFrame:
    return q.machines_overview(DB, machines or None, run_kinds=run_kinds,
                               include_short_run=include_short_run,
                               history_runs=history_runs)


def _report_url(machine: str, stamp: str, suffix: str) -> str:
    month = f"{stamp[:4]}.{stamp[4:6]}" if len(stamp) >= 6 else ""
    bucket = f"{machine}/{month}" if month else machine
    return f"{REPORT_BASE_URL}/{bucket}/daily.{stamp}.{suffix}"


def _fleet_overview(cfg: dict) -> pd.DataFrame:
    """Every daily rig's latest run, rendered as one scannable table."""
    machines = tuple(m for m in DAILY_MACHINES) if cfg["daily_only"] else ()
    overview = cached_machines_overview(machines, cfg["run_kinds"],
                                        cfg["include_short_run"],
                                        cfg["history_runs"], cfg["v"])
    if overview.empty:
        st.info("No runs match the current filters.")
        return overview

    # Counts are benchmark cases (perf series), not pytest functions: one LLM
    # test covers 2 prompts x 1st/2nd = 4 cases, resnet50 covers 2 batches.
    skipped = overview["skipped_cases"].fillna(0)
    success = overview["success_cases"].fillna(0)
    expected = overview["expected_cases"].fillna(0)
    failed = (expected - success).clip(lower=0)
    total = expected + skipped
    stale = overview["age_hours"] > 24

    def _status(idx: int) -> str:
        if failed.iloc[idx] > 0:
            return "🔴 failed"
        if stale.iloc[idx]:
            return "🟡 stale"
        if success.iloc[idx] > 0:
            return "🟢 success"
        return "⚪ unknown"

    def _run_cell(machine: str, stamp: object) -> str:
        """Timestamp plus its artefact links; st.dataframe allows only one
        link per cell, hence the markdown table below."""
        if stamp is None or pd.isna(stamp):
            return "—"
        stamp = str(stamp)
        short = stamp[4:] if len(stamp) >= 13 else stamp
        return (f"{short} ([html]({_report_url(machine, stamp, 'html')}), "
                f"[raw]({_report_url(machine, stamp, 'raw')}))")

    header = ("| Status | Machine | device | last success | last fail | "
              "OV version | total | skip | success | failed | Duration (min) |")
    divider = "|" + "---|" * 11
    rows = [header, divider]
    for i, row in enumerate(overview.itertuples(index=False)):
        duration = "" if pd.isna(row.duration_sec) else f"{row.duration_sec / 60:.1f}"
        rows.append(
            f"| {_status(i)} | {row.machine} | "
            f"{q.short_device_name(row.gpu_name)} | "
            f"{_run_cell(row.machine, row.last_success_stamp)} | "
            f"{_run_cell(row.machine, row.last_fail_stamp)} | "
            f"{row.ov_version or ''} | {int(total.iloc[i])} | "
            f"{int(skipped.iloc[i])} | {int(success.iloc[i])} | "
            f"{int(failed.iloc[i])} | {duration} |"
        )
    st.markdown("\n".join(rows))

    st.caption("Counts are benchmark cases, not pytest tests: one LLM test "
               "contributes 2 prompts x 1st/2nd = 4 cases. They describe the "
               "machine's newest run. `success` is the cases that produced a "
               "number; `total` is the most this machine has produced in its "
               "recent runs plus skipped cases.")
    return overview


def _tab_dashboard(cfg: dict) -> None:
    overview = _fleet_overview(cfg)

    st.divider()
    st.markdown(f"### {cfg['machine']} — recent trend")

    cohort = _cohort(cfg)
    if cohort.empty:
        st.info(f"No runs match the current filters for {cfg['machine']}.")
        return

    run_ids = tuple(cohort["run_id"].tolist())
    st.caption(f"{len(run_ids)} runs in scope · kinds: "
               f"{', '.join(cfg['run_kinds'])} · "
               f"models: {'all' if not cfg['models'] else len(cfg['models'])}")

    # --- functional first: a broken run makes its perf numbers meaningless ---
    func_summary = cached_functional_summary(run_ids, cfg["v"])
    issues = cached_functional_issues(run_ids, cfg["models"], cfg["v"])
    latest_run_id = str(cohort.iloc[-1]["run_id"])
    latest_stamp = str(cohort.iloc[-1]["stamp"])

    latest_issues = (issues[issues["run_id"] == latest_run_id]
                     if not issues.empty else pd.DataFrame())
    prior_nodeids = (set(issues[issues["run_id"] != latest_run_id]["nodeid"])
                     if not issues.empty else set())
    latest_nodeids = set(latest_issues["nodeid"]) if not latest_issues.empty else set()
    new_issues = latest_nodeids - prior_nodeids
    resolved = prior_nodeids - latest_nodeids

    if latest_nodeids:
        st.error(f"🔴 Functional issues in the latest run ({latest_stamp}): "
                 f"{len(latest_nodeids)} failing test(s), "
                 f"{len(new_issues)} new.")
    else:
        st.success(f"🟢 No functional issues in the latest run ({latest_stamp}).")

    fcols = st.columns(4)
    fcols[0].metric("Failing now", len(latest_nodeids))
    fcols[1].metric("New", len(new_issues))
    fcols[2].metric("Persisting", len(latest_nodeids & prior_nodeids))
    fcols[3].metric("Resolved", len(resolved))

    if not latest_issues.empty:
        with st.expander("Failing tests in the latest run", expanded=bool(new_issues)):
            view = latest_issues.copy()
            view["is_new"] = view["nodeid"].isin(new_issues)
            st.dataframe(view[["model", "nodeid", "outcome", "is_new", "message"]],
                         width="stretch", hide_index=True)

    states, _health = _machine_state_by_run(cfg, run_ids)

    # --- overall performance trend across the cohort ---
    st.markdown("### Overall performance trend")
    geo = cached_geomean_runs(run_ids, cfg["models"], (), cfg["v"])
    if geo.empty:
        st.info("Not enough perf data for a geomean trend.")
    else:
        geo = geo.merge(cohort[["run_id", "stamp"]], on="run_id", how="left")
        geo["machine_state"] = geo["run_id"].map(lambda r: states.get(r, "unknown"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=geo["stamp"], y=geo["geomean"], mode="lines+markers",
            name="geomean",
            marker=dict(
                color=[{"throttled": "#EF5350", "fluctuating": "#FFA726"}
                       .get(s, "#42A5F5") for s in geo["machine_state"]],
                size=9,
            ),
            text=[f"ov={v}<br>n={n}<br>machine={s}" for v, n, s in
                  zip(geo["ov_version"].fillna(""), geo["n_samples"],
                      geo["machine_state"])],
            hovertemplate="%{x}<br>geomean=%{y:.2f}<br>%{text}<extra></extra>",
        ))
        median = geo["geomean"].median()
        fig.add_hline(y=median, line=dict(dash="dash"),
                      annotation_text="cohort median")
        fig.update_layout(height=380, xaxis_title="run",
                          yaxis_title="geomean (common series)")
        _apply_y_scale(fig, cfg, geo["geomean"], [median])
        _newest_left(fig)
        st.plotly_chart(fig, width="stretch")
        st.caption("Restricted to series present in every run in scope, so a "
                   "run that measured fewer models does not shift the curve. "
                   "Red/orange markers flag throttled or fluctuating machines.")

    if not func_summary.empty:
        with st.expander("Run health history"):
            st.dataframe(_newest_first(func_summary)[
                             ["stamp", "ov_version", "overall_status",
                              "functional_issue_count",
                              "regressed_count", "compared_count"]],
                         width="stretch", hide_index=True)


def _tab_excel(cfg: dict) -> None:
    runs = cached_runs(cfg["machine"], cfg["v"])
    if runs.empty:
        st.info("No runs for this machine.")
        return

    filt = st.text_input("Filter by purpose/description",
                         value=DEFAULT_RUN_FILTER, key="excel_filter",
                         help="Use a shared term such as 'daily' to include both the legacy daily2 runs and the newer daily pipeline runs.")
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

    # Synthetic summary rows (parity with the legacy report viewer): success
    # count plus overall/bucketed geomeans, inserted right after the
    # Resnet50 rows, matching the legacy row order.
    stamp_by_run = {rid: str(runs.set_index("run_id").loc[rid, "stamp"]) for rid in run_ids}
    counts = cached_success_counts(run_ids, cfg["v"])
    success_row = {col: "" for col in matrix.columns}
    success_row.update(model="Success count", unit="count")
    for rid in run_ids:
        success_row[stamp_by_run[rid]] = counts.get(rid, 0)

    geomeans = cached_legacy_geomean_summary(run_ids, cfg["v"])
    geomean_labels = [
        ("geomean", "geomean"),
        ("geomean_2nd_short", "geomean (LLM/2nd/Short)"),
        ("geomean_1st_short", "geomean (LLM/1st/Short)"),
        ("geomean_2nd_long", "geomean (LLM/2nd/Long)"),
        ("geomean_1st_long", "geomean (LLM/1st/Long)"),
    ]
    geomean_rows = []
    for col_name, label in geomean_labels:
        row = {col: "" for col in matrix.columns}
        row.update(model=label, unit="")
        for rid in run_ids:
            value = geomeans.loc[rid, col_name] if rid in geomeans.index else None
            row[stamp_by_run[rid]] = value if pd.notna(value) else ""
        geomean_rows.append(row)

    insert_at = matrix["model"].ne("Resnet50").idxmax() if (matrix["model"] == "Resnet50").any() else 0
    matrix = pd.concat([
        matrix.iloc[:insert_at],
        pd.DataFrame([success_row, *geomean_rows]),
        matrix.iloc[insert_at:],
    ], ignore_index=True)

    st.markdown("**Paste block**  (tab-separated; headers: OV version / workweek / stamp)")
    meta = runs.set_index("run_id").loc[list(run_ids)]
    stamps = [c for c in matrix.columns if c not in PERF_COL_NON_VALUE]
    header_rows = [
        "\t".join(str(meta.loc[rid, "ov_version"] or "") for rid in run_ids),
        "\t".join(str(meta.loc[rid, "ww"] or "") for rid in run_ids),
        "",
        "\t".join(stamps),
    ]
    no_decimal_models = {
        "flux.1-schnell",
        "stable-diffusion-v1-5",
        "stable-diffusion-3.5-large-turbo",
        "Resnet50",
        "resnet50",
    }
    # FPS/count rows paste as whole numbers; ms/s latency rows keep 2
    # decimals — mirrors the legacy report viewer's per-row formatting
    # instead of a single float_format for the whole sheet.
    def _format_cell(value: object, unit: object, model: object) -> str:
        if pd.isna(value) or value == "":
            return ""
        unit_upper = str(unit).upper()
        number = float(value)
        if model in no_decimal_models or unit_upper in ("FPS", "COUNT"):
            return f"{number:.0f}"
        return f"{number:.2f}"

    units = matrix["unit"] if "unit" in matrix.columns else pd.Series("", index=matrix.index)
    formatted = pd.DataFrame({
        col: [_format_cell(v, u, model)
              for v, u, model in zip(matrix[col], units, matrix["model"])]
        for col in stamps
    })
    data_text = formatted.to_csv(sep="\t", index=False, header=False)
    paste = "\n\n" + "\n".join(header_rows) + "\n" + data_text
    st.text_area("Copy & paste into Excel", value=paste, height=260)

    st.markdown("**Matrix preview**")
    def _format_preview_value(model: object, value: object) -> str:
        if pd.isna(value):
            return ""
        number = float(value)
        if model in no_decimal_models:
            return f"{number:.0f}"
        return f"{number:.2f}" if not number.is_integer() else f"{number:.0f}"

    preview = matrix.copy()
    for col in stamps:
        preview[col] = [
            _format_preview_value(model, value)
            for model, value in zip(preview["model"], preview[col])
        ]
    preview = preview.style.set_properties(
        subset=stamps, **{"text-align": "right"}
    )
    st.dataframe(preview, width="stretch", hide_index=True)

    extras = cached_extra_rows(run_ids, cfg["profile"], cfg["v"])
    if not extras.empty:
        with st.expander(f"⚠ {len(extras)} perf rows not covered by display profile"):
            st.dataframe(extras, width="stretch", hide_index=True)


def _series_label(row: pd.Series) -> str:
    return (f"{row['model']} | {row['precision']} | "
            f"in={row['in_token']} out={row['out_token']} | "
            f"{row['exec_mode']} [{row['unit']}]")


def _tab_regression(cfg: dict) -> None:
    """Trend + regression over a run-counted window.

    The question this tab answers is 'has this series drifted vs its own
    recent past', not 'did today's single point miss the band'. Windows are
    counted in runs so an irregular benchmark schedule cannot silently empty
    them, and each candidate regression is annotated with the machine state
    during the run so throttling is not mistaken for a code change.
    """
    st.caption("Recent runs compared with the history before them. Windows "
               "are counted in runs, and each candidate regression is "
               "annotated with the machine state during the run.")

    cohort = _cohort(cfg)
    if cohort.empty:
        st.info("No runs match the current filters.")
        return

    run_ids = tuple(cohort["run_id"].tolist())
    recent_ids = run_ids[-cfg["recent_runs"]:]
    st.caption(f"Recent {len(recent_ids)} run(s) vs the "
               f"{len(run_ids) - len(recent_ids)} run(s) before them.")

    df = cached_series_trend(cfg["machine"], cfg["recent_runs"],
                             cfg["history_runs"], cfg["run_kinds"],
                             cfg["models"], cfg["include_short_run"], cfg["v"])
    if df.empty:
        st.info("No data for this machine / window.")
        return

    valid = df[df["status"] == "ok"].copy()
    if valid.empty:
        st.info("Not enough history for any series in this window.")
        return

    valid["severity"] = pd.concat([
        valid["worsening_pct"].fillna(0) / cfg["pct"],
        valid["worsening_z"].fillna(0) / cfg["z"],
    ], axis=1).max(axis=1)
    valid = valid.sort_values("severity", ascending=False).reset_index(drop=True)

    states, _health = _machine_state_by_run(cfg, run_ids)
    recent_states = [states.get(r, "unknown") for r in recent_ids]
    disturbed_recent = [s for s in recent_states
                        if s in {"throttled", "fluctuating"}]
    machine_state = ("throttled" if "throttled" in recent_states else
                     "fluctuating" if "fluctuating" in recent_states else
                     "unknown" if all(s == "unknown" for s in recent_states)
                     else "stable")

    if disturbed_recent:
        st.warning(f"⚠ {len(disturbed_recent)} of {len(recent_states)} recent "
                   f"run(s) ran on a disturbed machine ({machine_state}). "
                   "Regressions below are attributed accordingly.")

    noisy_count = int((valid["history_cv"].fillna(0) >= cfg["cv"]).sum())
    bad = valid[(valid["worsening_pct"].fillna(0) >= cfg["pct"]) |
                (valid["worsening_z"].fillna(0) >= cfg["z"])]
    better = valid[(valid["worsening_pct"].fillna(0) <= -cfg["pct"]) |
                   (valid["worsening_z"].fillna(0) <= -cfg["z"])]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Series tracked", len(valid))
    m2.metric("Worsening ≥ threshold", len(bad))
    m3.metric("Improving ≥ threshold", len(better))
    m4.metric("Noisy history", noisy_count)

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
        out["history_cv_%"] = (out["history_cv"] * 100).round(2)

        def _with_unit(val: float | None, unit: str | None) -> str:
            if val is None or pd.isna(val):
                return ""
            return f"{val:.3f} {unit or ''}".rstrip()

        out["recent"] = [_with_unit(v, u) for v, u in
                         zip(out["recent_median"], out["unit"].fillna(""))]
        out["history"] = [_with_unit(v, u) for v, u in
                          zip(out["history_median"], out["unit"].fillna(""))]

        verdicts = []
        attributions = []
        for _, r in out.iterrows():
            pct = r["worsening_pct"] if pd.notna(r["worsening_pct"]) else 0.0
            cv = r["history_cv"] if pd.notna(r["history_cv"]) else 0.0
            if pct >= cfg["pct"] and cv >= cfg["cv"]:
                verdict = "noisy"
            elif pct >= cfg["pct"]:
                verdict = "regressed"
            elif pct <= -cfg["pct"]:
                verdict = "improved"
            else:
                verdict = "same"
            verdicts.append(verdict)
            attributions.append(q.attribute_regression(verdict, machine_state))
        out["verdict"] = verdicts
        out["attribution"] = attributions

        return out[["series", "verdict", "attribution", "severity",
                    "worsening_%", "worsening_z", "recent", "history",
                    "recent_n", "history_n", "history_cv_%", "direction",
                    "model", "precision", "in_token", "out_token",
                    "exec_mode", "unit",
                    "recent_median", "history_median", "history_sigma"]]

    table = _fmt(valid)
    insufficient = df[df["status"] == "insufficient_data"]

    show_only_regressed = st.checkbox(
        "Show only regressed / noisy series", value=False,
        key="regression_only_bad")
    view = (table[table["verdict"].isin(["regressed", "noisy"])]
            if show_only_regressed else table)
    if view.empty:
        st.success("No regressed series in this window.")
        view = table

    st.markdown("### Series ranked by worsening %")
    st.caption("Positive values mean the recent runs are slower / worse than "
               "the preceding history. Click a row to plot it below.")
    event = st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="regression_table",
        column_config={
            "model":          st.column_config.Column(width="small", disabled=True),
            "precision":      None,
            "in_token":       None,
            "out_token":      None,
            "exec_mode":      None,
            "unit":           None,
            "recent_median":  None,
            "history_median": None,
            "history_sigma":  None,
        },
    )

    sel_rows = event.selection.rows if event and event.selection else []
    sel_idx = sel_rows[0] if sel_rows else 0
    if sel_idx >= len(view):
        return
    row = view.iloc[sel_idx]

    st.markdown(
        f"### Trend — {row['model']} / {row['precision']} / "
        f"in={row['in_token']} out={row['out_token']} / {row['exec_mode']} "
        f"[{row['unit']}]"
    )

    hist = cached_series_runs(
        cfg["machine"], row["model"], row["precision"],
        int(row["in_token"]), int(row["out_token"]), row["exec_mode"],
        cfg["history_runs"] + cfg["recent_runs"], cfg["run_kinds"],
        cfg["include_short_run"], cfg["v"])
    if hist.empty:
        st.info("No history for this series in the selected window.")
        return

    hist = hist.merge(cohort[["run_id", "stamp"]], on="run_id", how="left")
    hist["machine_state"] = hist["run_id"].map(lambda r: states.get(r, "unknown"))

    unit = row["unit"] or ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["stamp"], y=hist["value"], mode="lines+markers",
        name="value",
        marker=dict(
            color=[{"throttled": "#EF5350", "fluctuating": "#FFA726"}
                   .get(s, "#42A5F5") for s in hist["machine_state"]],
            size=9,
        ),
        text=[f"ov={v}<br>machine={s}" for v, s in
              zip(hist["ov_version"].fillna(""), hist["machine_state"])],
        hovertemplate="%{x}<br>value=%{y:.2f} " + unit + "<br>%{text}<extra></extra>",
    ))

    x_span = [hist["stamp"].iloc[0], hist["stamp"].iloc[-1]]
    extra_y: list[float] = []
    if pd.notna(row["history_median"]):
        median = float(row["history_median"])
        extra_y.append(median)
        fig.add_trace(go.Scatter(
            x=x_span, y=[median] * 2, mode="lines",
            line=dict(dash="dot", color="#1f77b4", width=2),
            name=f"history median ({int(row['history_n'])} pts)",
        ))
        sigma = row["history_sigma"]
        if pd.notna(sigma) and float(sigma) > 0:
            hi, lo = median + 2 * float(sigma), median - 2 * float(sigma)
            extra_y.extend([hi, lo])
            fig.add_trace(go.Scatter(
                x=[*x_span, *x_span[::-1]], y=[hi, hi, lo, lo],
                fill="toself", fillcolor="rgba(99, 110, 250, 0.20)",
                mode="none", name="history ±2σ", hoverinfo="skip",
            ))
    if pd.notna(row["recent_median"]):
        extra_y.append(float(row["recent_median"]))
        fig.add_trace(go.Scatter(
            x=x_span, y=[float(row["recent_median"])] * 2, mode="lines",
            line=dict(dash="dash", color="firebrick", width=2),
            name=f"recent median (last {cfg['recent_runs']} runs)",
        ))

    fig.update_layout(
        height=450, hovermode="x unified",
        xaxis_title="run", yaxis_title=f"value [{unit}]" if unit else "value",
        legend=dict(orientation="h", y=-0.25),
    )
    _apply_y_scale(fig, cfg, hist["value"], extra_y)
    _newest_left(fig)
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Verdict: **{row['verdict']}** · attribution: **{row['attribution']}** · "
        f"recent median = {row['recent']} (n={int(row['recent_n'])}) vs "
        f"history median = {row['history']} (n={int(row['history_n'])}). "
        f"Worsening = {row['worsening_%']:+.2f}%, z={row['worsening_z']:+.2f}. "
        f"History CV = {row['history_cv_%']:.2f}%."
    )

    monitor = q.machine_stats_for_run(DB, str(cohort.iloc[-1]["run_id"]),
                                      models=[row["model"]])
    if not monitor.empty:
        with st.expander("Machine telemetry for this model in the latest run"):
            st.dataframe(
                monitor[["nodeid", "samples", "gpu_clock_mhz_mean",
                         "gpu_clock_max_mhz", "gpu_utilization_mean",
                         "gpu_power_watts_mean", "gpu_temp_c_max",
                         "throttled_sample_ratio", "throttle_reasons"]],
                width="stretch", hide_index=True,
            )

    if not insufficient.empty:
        with st.expander(f"{len(insufficient)} series with insufficient history"):
            st.dataframe(
                insufficient[["model", "precision", "in_token", "out_token",
                              "exec_mode", "unit", "recent_n", "history_n"]],
                width="stretch", hide_index=True)



def _tab_geomean(cfg: dict) -> None:
    st.caption("Geomean over the runs in scope. Restricted to series measured "
               "in every run so a run with fewer models does not look like a "
               "performance change.")

    cohort = _cohort(cfg)
    if cohort.empty:
        st.info("No runs match the current filters.")
        return
    run_ids = tuple(cohort["run_id"].tolist())

    exec_modes = st.multiselect(
        "Exec modes", ["1st", "2nd", "pipeline", "tps"], default=["2nd"],
        key="geomean_exec_modes")

    df = cached_geomean_runs(run_ids, cfg["models"], tuple(exec_modes), cfg["v"])
    if df.empty:
        st.info("No data for this filter.")
        return

    df = df.merge(cohort[["run_id", "stamp"]], on="run_id", how="left")
    states, _health = _machine_state_by_run(cfg, run_ids)
    df["machine_state"] = df["run_id"].map(lambda r: states.get(r, "unknown"))

    median = df["geomean"].median()
    mad = (df["geomean"] - median).abs().median()
    sigma = 1.4826 * mad

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["stamp"], y=df["geomean"], mode="lines+markers", name="geomean",
        marker=dict(
            color=[{"throttled": "#EF5350", "fluctuating": "#FFA726"}
                   .get(s, "#42A5F5") for s in df["machine_state"]],
            size=9,
        ),
        text=[f"ov={v}<br>n={n}<br>machine={s}" for v, n, s in
              zip(df["ov_version"].fillna(""), df["n_samples"],
                  df["machine_state"])],
        hovertemplate="%{x}<br>geomean=%{y:.2f}<br>%{text}<extra></extra>",
    ))
    band_y: list[float] = [median] if not math.isnan(median) else []
    if not math.isnan(sigma) and sigma > 0:
        band_y.extend([median - 2 * sigma, median + 2 * sigma])
        fig.add_hline(y=median, line=dict(dash="dash"), annotation_text="median")
        fig.add_hrect(y0=median - 2 * sigma, y1=median + 2 * sigma,
                      line_width=0, fillcolor="LightBlue", opacity=0.2,
                      annotation_text="±2σ")
    fig.update_layout(height=450, hovermode="x unified",
                      xaxis_title="run", yaxis_title="geomean")
    _apply_y_scale(fig, cfg, df["geomean"], band_y)
    _newest_left(fig)
    st.plotly_chart(fig, width="stretch")

    if len(df) >= 5 and sigma > 0:
        latest = df.iloc[-1]
        z = (latest["geomean"] - median) / sigma
        pct = (latest["geomean"] - median) / median * 100 if median else 0
        # Only 'tps' style throughput improves as the number grows; the LLM
        # latency modes are lower-is-better.
        sign = -1 if exec_modes == ["tps"] else 1
        worsening_z, worsening_pct = sign * z, sign * pct
        if abs(worsening_z) >= cfg["z"] and abs(worsening_pct) >= cfg["pct"] * 100:
            direction = "worse" if worsening_z > 0 else "better"
            note = (f" Machine state for this run: {latest['machine_state']}."
                    if latest["machine_state"] != "stable" else "")
            st.error(f"⚠ Latest geomean is {direction} by "
                     f"z={worsening_z:+.2f}, {worsening_pct:+.1f}%.{note}")
        else:
            st.success("Latest geomean within band "
                       f"(z={worsening_z:+.2f}, {worsening_pct:+.1f}%).")

    st.dataframe(_newest_first(df)[["stamp", "ov_version", "geomean",
                                    "n_samples", "machine_state"]],
                 width="stretch", hide_index=True)


def _tab_noise(cfg: dict) -> None:
    st.caption("Series scatter over the runs in scope, next to the machine "
               "telemetry recorded during those runs. High series CV on a "
               "throttling rig points at the machine rather than the code.")

    cohort = _cohort(cfg)
    if cohort.empty:
        st.info("No runs match the current filters.")
        return
    run_ids = tuple(cohort["run_id"].tolist())

    perf = q.perf_for_runs(DB, run_ids, models=cfg["models"] or None)
    if perf.empty:
        st.info("No data.")
        return

    keys = ["model", "precision", "in_token", "out_token", "exec_mode", "unit"]
    grouped = perf.groupby(keys, dropna=False)["value"]
    noise = grouped.agg(n="size", median_value="median").reset_index()
    mad = grouped.agg(lambda v: float((v - v.median()).abs().median()))
    noise["mad"] = mad.to_numpy()
    noise["cv_pct"] = (noise["mad"] / noise["median_value"] * 100).round(2)
    noise["median_value"] = noise["median_value"].round(3)
    noise["mad"] = noise["mad"].round(3)
    noise = noise[noise["n"] >= 3].sort_values("cv_pct", ascending=False)

    if noise.empty:
        st.info("Not enough points per series in this window.")
        return

    st.metric("Series above noisy CV threshold",
              int((noise["cv_pct"] >= cfg["cv"] * 100).sum()))
    st.dataframe(noise, width="stretch", hide_index=True)

    health = cached_machine_health(run_ids, cfg["v"])
    if not health.empty:
        st.markdown("#### Machine fluctuation over the same runs")
        st.dataframe(
            _newest_first(health)[
                ["stamp", "ov_version", "avg_gpu_clock_mhz",
                 "gpu_clock_ratio", "max_throttle_ratio", "max_gpu_temp_c",
                 "avg_gpu_utilization", "avg_cpu_usage",
                 "max_sample_duration_ms"]].round(3),
            width="stretch", hide_index=True,
        )
        st.caption("`gpu_clock_ratio` is the mean GPU clock divided by the "
                   "card's own max clock; sustained low values mean the run "
                   "never reached the speed its peers did.")


def _tab_functional(cfg: dict) -> None:
    st.caption("Failed / errored test cases across the runs in scope.")

    cohort = _cohort(cfg)
    if cohort.empty:
        st.info("No runs match the current filters.")
        return
    run_ids = tuple(cohort["run_id"].tolist())

    summary_df = cached_functional_summary(run_ids, cfg["v"])
    issues_df = cached_functional_issues(run_ids, cfg["models"], cfg["v"])

    if summary_df.empty and issues_df.empty:
        st.info("No analysis_results data found for these runs. "
                "Run the analysis pipeline to populate this view.")
        return

    if not summary_df.empty:
        _STATUS_COLOUR = {"green": "🟢", "yellow": "🟡", "red": "🔴", "gray": "⚫"}
        summary_df = summary_df.copy()
        summary_df["status"] = summary_df["overall_status"].map(
            lambda s: f"{_STATUS_COLOUR.get(s, '❓')} {s}"
        )
        st.markdown("#### Run health summary")
        st.dataframe(
            _newest_first(summary_df)[
                ["stamp", "ov_version", "status",
                 "functional_issue_count", "regressed_count",
                 "compared_count"]].rename(columns={
                "stamp": "Stamp", "ov_version": "OV Version",
                "status": "Status",
                "functional_issue_count": "Functional issues",
                "regressed_count": "Regressions",
                "compared_count": "Compared",
            }),
            width="stretch", hide_index=True,
        )

    if issues_df.empty:
        st.success("No functional issues recorded in the selected runs.")
        return

    latest_run_id = str(cohort.iloc[-1]["run_id"])
    latest = set(issues_df[issues_df["run_id"] == latest_run_id]["nodeid"])
    prior = set(issues_df[issues_df["run_id"] != latest_run_id]["nodeid"])

    c1, c2, c3 = st.columns(3)
    c1.metric("New in latest run", len(latest - prior))
    c2.metric("Persisting", len(latest & prior))
    c3.metric("Resolved", len(prior - latest))

    st.markdown("#### Individual failures")
    view = issues_df.copy()
    view["state"] = view["nodeid"].map(
        lambda n: "new" if n in latest - prior
        else "persisting" if n in latest & prior else "past"
    )
    st.dataframe(
        view[["stamp", "ov_version", "model", "precision", "outcome", "state",
              "nodeid", "message"]]
        .rename(columns={
            "stamp": "Stamp", "ov_version": "OV Version",
            "model": "Model", "precision": "Prec",
            "outcome": "Outcome", "state": "State",
            "nodeid": "Test", "message": "Message",
        }),
        width="stretch", hide_index=True,
    )

    if not summary_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary_df["stamp"],
                             y=summary_df["functional_issue_count"],
                             name="Functional issues", marker_color="#EF5350"))
        fig.add_trace(go.Bar(x=summary_df["stamp"],
                             y=summary_df["regressed_count"],
                             name="Regressions", marker_color="#FFA726"))
        fig.update_layout(barmode="stack",
                          title="Failures and regressions per run",
                          xaxis_title="Run stamp", yaxis_title="Count",
                          height=300)
        _newest_left(fig)
        st.plotly_chart(fig, width="stretch")


def _tab_compare(cfg: dict) -> None:
    st.caption("A raw A-vs-B delta cannot separate a real change from ordinary "
               "run-to-run scatter, so each series is also compared against the "
               "history preceding each run.")

    cohort = _cohort(cfg)
    if cohort.empty:
        st.info("No runs match the current filters.")
        return

    picker = cohort.iloc[::-1].reset_index(drop=True)
    options = list(range(len(picker)))
    labels = [f"{picker.iloc[i]['stamp']} · {picker.iloc[i]['run_kind']} · "
              f"{picker.iloc[i]['ov_version'] or ''}" for i in options]

    col_a, col_b = st.columns(2)
    idx_a = col_a.selectbox("Run A (current)", options,
                            format_func=lambda i: labels[i], index=0,
                            key="compare_run_a")
    idx_b = col_b.selectbox("Run B (reference)", options,
                            format_func=lambda i: labels[i],
                            index=min(1, len(options) - 1),
                            key="compare_run_b")

    run_a = str(picker.iloc[idx_a]["run_id"])
    run_b = str(picker.iloc[idx_b]["run_id"])
    if run_a == run_b:
        st.warning("Select two different runs to compare.")
        return

    df = cached_trend_compare(cfg["machine"], run_a, run_b, cfg["history_runs"],
                              cfg["run_kinds"], cfg["models"], cfg["v"])
    if df.empty:
        st.info("No overlapping series found between the two runs.")
        return

    states, _health = _machine_state_by_run(cfg, tuple(cohort["run_id"].tolist()))
    state_a = states.get(run_a, "unknown")
    state_b = states.get(run_b, "unknown")
    st.caption(f"Machine state — A: {_STATE_ICON.get(state_a, '⚪')} {state_a} · "
               f"B: {_STATE_ICON.get(state_b, '⚪')} {state_b}")
    if state_a in {"throttled", "fluctuating"}:
        st.warning(f"⚠ Run A ran on a {state_a} machine; differences below may "
                   "reflect machine state rather than code.")

    df = df.copy()
    df["attribution"] = [
        q.attribute_regression(
            "regressed" if v == "regressed" else "same", state_a)
        for v in df["verdict"]
    ]
    df["change"] = df["improvement_pct"].apply(
        lambda x: f"{x * 100:+.1f}%" if pd.notna(x) else "—")
    df["vs_history"] = df["a_vs_history_pct"].apply(
        lambda x: f"{x * 100:+.1f}%" if pd.notna(x) else "—")
    df["history_z"] = df["a_vs_history_z"].round(2)

    vc = df["verdict"].value_counts().to_dict()
    cols = st.columns(6)
    for col, label, colour in zip(
        cols,
        ["improved", "same", "regressed", "noisy", "insufficient", "unavailable"],
        ["🟢", "🔵", "🔴", "🟡", "⚪", "⚫"],
    ):
        col.metric(f"{colour} {label.capitalize()}", vc.get(label, 0))

    tc = df["trend_context"].value_counts().to_dict()
    st.caption(
        f"Trend context — outside history: {tc.get('outside_history', 0)} · "
        f"within history: {tc.get('within_history', 0)} · "
        f"unknown: {tc.get('unknown', 0)}. "
        "'within_history' means run A's value sits inside its own recent "
        "spread, so the A/B delta is most likely scatter."
    )

    verdict_filter = st.multiselect(
        "Filter by verdict",
        options=["improved", "same", "regressed", "noisy", "insufficient",
                 "unavailable"],
        default=["improved", "same", "regressed", "unavailable"],
        key="compare_verdict_filter",
    )
    context_filter = st.multiselect(
        "Filter by trend context",
        options=["outside_history", "within_history", "unknown"],
        default=["outside_history", "within_history", "unknown"],
        key="compare_context_filter",
    )
    view = df
    if verdict_filter:
        view = view[view["verdict"].isin(verdict_filter)]
    if context_filter:
        view = view[view["trend_context"].isin(context_filter)]

    st.dataframe(
        view[["model", "precision", "in_token", "out_token", "exec_mode",
              "unit", "value_a", "value_b", "change", "median_a",
              "vs_history", "history_z", "trend_context", "verdict",
              "attribution"]].rename(columns={
            "model": "Model", "precision": "Prec",
            "in_token": "In", "out_token": "Out", "exec_mode": "Mode",
            "unit": "Unit", "value_a": "A", "value_b": "B",
            "change": "A vs B", "median_a": "A history median",
            "vs_history": "A vs history", "history_z": "History z",
            "trend_context": "Context", "verdict": "Verdict",
            "attribution": "Attribution",
        }),
        width="stretch", hide_index=True,
    )

    st.markdown("### Series trend")
    if view.empty:
        return
    labels_map = {
        i: (f"{r['model']} | {r['precision']} | in={r['in_token']} "
            f"out={r['out_token']} | {r['exec_mode']}")
        for i, (_, r) in enumerate(view.iterrows())
    }
    sel = st.selectbox("Series", list(labels_map),
                       format_func=lambda i: labels_map[i], key="compare_series")
    row = view.iloc[sel]

    hist = cached_series_runs(
        cfg["machine"], row["model"], row["precision"],
        int(row["in_token"]), int(row["out_token"]), row["exec_mode"],
        cfg["history_runs"] + cfg["recent_runs"], cfg["run_kinds"],
        cfg["include_short_run"], cfg["v"])
    if hist.empty:
        st.info("No history for this series.")
        return

    hist = hist.merge(cohort[["run_id", "stamp"]], on="run_id", how="left")
    unit = row["unit"] or ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["stamp"], y=hist["value"],
                             mode="lines+markers", name="history"))
    for run_id, name, colour in ((run_a, "Run A", "firebrick"),
                                 (run_b, "Run B", "#2E7D32")):
        point = hist[hist["run_id"] == run_id]
        if not point.empty:
            fig.add_trace(go.Scatter(
                x=point["stamp"], y=point["value"], mode="markers",
                marker=dict(size=15, color=colour, symbol="diamond"),
                name=name))
    extra_y: list[float] = []
    if pd.notna(row.get("median_a")):
        extra_y.append(float(row["median_a"]))
        fig.add_hline(y=float(row["median_a"]), line=dict(dash="dot"),
                      annotation_text="A history median")
    fig.update_layout(height=420, xaxis_title="run",
                      yaxis_title=f"value [{unit}]" if unit else "value",
                      legend=dict(orientation="h", y=-0.25))
    _apply_y_scale(fig, cfg, hist["value"], extra_y)
    _newest_left(fig)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(layout="wide", page_title="Daily LLM Viewer")
    pd.set_option("display.float_format", "{:.2f}".format)
    # Trim Streamlit's ~6rem top gap, but stay clear of the fixed header
    # (~2.875rem) that would otherwise overlap the tab bar. The side padding
    # defaults to 5rem in wide mode, which costs a lot of table width.
    st.markdown(
        "<style>.block-container{padding-top:3.5rem;"
        "padding-left:2rem;padding-right:2rem;}</style>",
        unsafe_allow_html=True,
    )

    cfg = _sidebar()

    tabs = st.tabs(["Dashboard", "Excel", "Regression", "Geomean", "Noise",
                    "Functional", "Compare"])
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
    with tabs[5]:
        _tab_functional(cfg)
    with tabs[6]:
        _tab_compare(cfg)


if __name__ == "__main__":
    main()
