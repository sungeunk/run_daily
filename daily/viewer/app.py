#!/usr/bin/env python3
"""Streamlit viewer for the daily benchmark DuckDB.

Run with::

    streamlit run daily/viewer/app.py -- --db daily/viewer/bench.duckdb

The DB is built by ``python -m viewer.ingest.cli``. This app is a pure
read-only consumer.

Tabs
----
1. Excel Paste  — wide matrix for a fixed display profile, selected runs
                  become columns.
2. Trend        — time-series of one perf point with rolling baseline band.
3. Regressions  — MAD-based regression list for the latest run of a
                  machine. Noisy series are flagged.
4. Geomean      — geometric-mean trend across a bucket (machine-wide health).
5. Noise        — per-series coefficient of variation. Useful for iGPU
                  diagnostics where fluctuation is inherent.
"""

from __future__ import annotations

import argparse
import math
import os
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
                  days: int, _v: float) -> pd.DataFrame:
    return q.series_history(DB, machine, model, precision,
                            in_token, out_token, exec_mode, days=days)


@st.cache_data(show_spinner=False)
def cached_noise(machine: str, days: int, _v: float) -> pd.DataFrame:
    return q.noise_summary(DB, machine=machine, days=days)


@st.cache_data(show_spinner=False)
def cached_geomean(machine: str, exec_mode: str, in_bucket: str | None,
                   out_bucket: str | None, days: int, _v: float) -> pd.DataFrame:
    return q.geomean_trend(DB, machine, exec_mode=exec_mode,
                           in_bucket=in_bucket, out_bucket=out_bucket,
                           days=days)


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
# Tabs
# ---------------------------------------------------------------------------

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
        use_container_width=True,
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
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    extras = cached_extra_rows(run_ids, cfg["profile"], cfg["v"])
    if not extras.empty:
        with st.expander(f"⚠ {len(extras)} perf rows not covered by display profile"):
            st.dataframe(extras, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def cached_trend_regressions(machine: str, recent_days: int,
                             baseline_days: int, _v: float) -> pd.DataFrame:
    return q.trend_regressions(DB, machine,
                               recent_days=recent_days,
                               baseline_days=baseline_days)


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
                                  baseline_days, cfg["v"])
    if df.empty:
        st.info("No data for this machine / window.")
        return

    valid = df[df["status"] == "ok"].copy()
    noisy_count = int((valid["recent_cv"].fillna(0) >= cfg["cv"]).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Series tracked", len(valid))
    # Worsening above pct threshold, regardless of noise — surfaces the
    # strongest signals. The user should cross-check with the graph for
    # series that are flagged noisy.
    bad = valid[valid["worsening_pct"].fillna(0) >= cfg["pct"]]
    m2.metric("Worsening ≥ threshold", len(bad))
    better = valid[valid["worsening_pct"].fillna(0) <= -cfg["pct"]]
    m3.metric("Improving ≥ threshold", len(better))
    m4.metric("Noisy (recent CV high)", noisy_count)

    # Build a compact, sortable table.
    def _fmt(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["series"] = out.apply(_series_label, axis=1)
        out["worsening_%"] = (out["worsening_pct"] * 100).round(2)
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
        return out[["series", "worsening_%", "recent", "baseline",
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
        use_container_width=True,
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
        cfg["days"], cfg["v"])
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

    fig.update_layout(
        height=450, hovermode="x unified",
        xaxis_title="timestamp",
        yaxis_title=f"value [{unit}]" if unit else "value",
        xaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Recent median = {row['recent_median']:.3f} {unit} "
        f"(n={int(row['recent_n'])}) vs baseline median = "
        f"{row['baseline_median']:.3f} {unit} (n={int(row['baseline_n'])}). "
        f"Worsening = {row['worsening_%']:+.2f}%. "
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
            st.dataframe(_fmt(insufficient), use_container_width=True,
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
                        int(days), cfg["v"])
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
    st.plotly_chart(fig, use_container_width=True)

    # Alert banner when the latest point is outside the band.
    if len(df) >= 5 and sigma > 0:
        latest = df.iloc[-1]["geomean"]
        z = (latest - median) / sigma
        pct = (latest - median) / median * 100 if median else 0
        if abs(z) >= cfg["z"] and abs(pct) >= cfg["pct"] * 100:
            direction = "worse" if z > 0 else "better"
            st.error(f"⚠ Latest geomean is {direction} by "
                     f"z={z:+.2f}, {pct:+.1f}%.")
        else:
            st.success(f"Latest geomean within band (z={z:+.2f}, {pct:+.1f}%).")

    st.dataframe(df[["ts", "ww", "ov_version", "geomean", "n_samples"]]
                 .tail(30), use_container_width=True, hide_index=True)


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
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(layout="wide", page_title="Daily LLM Viewer")
    pd.set_option("display.float_format", "{:.2f}".format)
    st.title("Daily LLM Benchmark Viewer")

    cfg = _sidebar()

    tabs = st.tabs(["Excel", "Regression", "Geomean", "Noise"])
    with tabs[0]:
        _tab_excel(cfg)
    with tabs[1]:
        _tab_regression(cfg)
    with tabs[2]:
        _tab_geomean(cfg)
    with tabs[3]:
        _tab_noise(cfg)


if __name__ == "__main__":
    main()
