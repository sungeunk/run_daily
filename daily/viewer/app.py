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
def cached_regressions(run_id: str, z: float, pct: float, cv: float,
                       _v: float) -> pd.DataFrame:
    return q.regressions_for_run(DB, run_id,
                                 z_threshold=z, pct_threshold=pct,
                                 noisy_cv=cv)


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

def _sidebar() -> dict:
    st.sidebar.header("Settings")
    v = _db_version()
    if v == 0.0:
        st.sidebar.error(f"DB not found at {DB}")
        st.stop()
    st.sidebar.caption(f"DB: `{DB}`")
    machines = cached_machines(v)
    if not machines:
        st.sidebar.warning("No runs in DB yet — run `viewer.ingest.cli` first.")
        st.stop()
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

    days = st.sidebar.slider("Trend history (days)", 7, 365, 60)
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
                         value="", key="excel_filter")
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


def _tab_trend(cfg: dict) -> None:
    st.subheader("Trend — single series")
    runs = cached_runs(cfg["machine"], cfg["v"])
    if runs.empty:
        return

    # Grab the set of distinct series from the latest run so the user
    # doesn't have to type.
    latest_run_id = runs.iloc[0]["run_id"]
    import duckdb
    with duckdb.connect(str(DB), read_only=True) as con:
        series = con.execute("""
            SELECT DISTINCT model, precision, in_token, out_token, exec_mode, unit
            FROM perf
            WHERE run_id = ?
            ORDER BY model, precision, in_token, out_token, exec_mode
        """, [latest_run_id]).fetchdf()
    if series.empty:
        st.info("No perf points in the latest run.")
        return

    series["label"] = series.apply(
        lambda r: f"{r['model']} | {r['precision']} | in={r['in_token']} "
                  f"out={r['out_token']} | {r['exec_mode']} [{r['unit']}]",
        axis=1)

    picked = st.multiselect("Series", series["label"].tolist(),
                            default=[series.iloc[0]["label"]])
    if not picked:
        return

    fig = go.Figure()
    for label in picked:
        s = series.loc[series["label"] == label].iloc[0]
        df = cached_series(
            cfg["machine"], s["model"], s["precision"],
            int(s["in_token"]), int(s["out_token"]), s["exec_mode"],
            cfg["days"], cfg["v"])
        if df.empty:
            continue
        unit = s["unit"] or ""
        worse = _worse_direction(unit)
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["value"], mode="lines+markers",
            name=f"{label}",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>value=%{y:.2f} "
                          + unit + "<br>%{text}",
            text=[f"ov={v}" for v in df["ov_version"].fillna("")],
        ))
        if df["win_median"].notna().any():
            band_hi = df["win_median"] + 2 * df["win_sigma"]
            band_lo = df["win_median"] - 2 * df["win_sigma"]
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["win_median"], mode="lines",
                name=f"median ({label})", line=dict(dash="dash", width=1),
                opacity=0.6,
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([df["ts"], df["ts"][::-1]]),
                y=pd.concat([band_hi, band_lo[::-1]]),
                fill="toself", mode="none", opacity=0.15,
                name=f"±2σ ({label})", showlegend=False,
            ))
        _ = worse  # color handling can be added later (e.g. red dots when worse)

    fig.update_layout(
        height=450, hovermode="x unified",
        xaxis_title="timestamp", yaxis_title="value",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_regressions(cfg: dict) -> None:
    st.subheader("Regressions — latest run")
    runs = cached_runs(cfg["machine"], cfg["v"])
    if runs.empty:
        return

    # Default to latest; allow override.
    stamps = runs["stamp"].tolist()
    idx = st.selectbox("Run", range(len(stamps)),
                       format_func=lambda i: f"{stamps[i]}  ({runs.iloc[i]['ov_version']})")
    run_id = runs.iloc[idx]["run_id"]

    df = cached_regressions(run_id, cfg["z"], cfg["pct"], cfg["cv"], cfg["v"])
    if df.empty:
        st.info("No data.")
        return

    reg = df[df["status"] == "regression"]
    imp = df[df["status"] == "improvement"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regressions", len(reg))
    c2.metric("Improvements", len(imp))
    c3.metric("Noisy series",
              int((df["is_noisy"].fillna(False)).sum()))
    c4.metric("Tracked points", len(df))

    def _fmt(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in ("value", "win_median", "win_sigma"):
            if col in out:
                out[col] = out[col].round(3)
        if "pct_diff" in out:
            out["pct_diff"] = (out["pct_diff"] * 100).round(2)
        if "z_score" in out:
            out["z_score"] = out["z_score"].round(2)
        if "cv" in out:
            out["cv"] = (out["cv"] * 100).round(2)
        return out

    st.markdown("### ❌ Regressions")
    if reg.empty:
        st.write("_none_")
    else:
        st.dataframe(_fmt(reg), use_container_width=True, hide_index=True)

    st.markdown("### ✅ Improvements")
    if imp.empty:
        st.write("_none_")
    else:
        st.dataframe(_fmt(imp), use_container_width=True, hide_index=True)

    with st.expander("All tracked series for this run"):
        st.dataframe(_fmt(df), use_container_width=True, hide_index=True)


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
    days = cols[2].number_input("days", min_value=7, max_value=365,
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
    days = st.number_input("Window (days)", min_value=7, max_value=365,
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

    tabs = st.tabs(["Excel", "Trend", "Regressions", "Geomean", "Noise"])
    with tabs[0]:
        _tab_excel(cfg)
    with tabs[1]:
        _tab_trend(cfg)
    with tabs[2]:
        _tab_regressions(cfg)
    with tabs[3]:
        _tab_geomean(cfg)
    with tabs[4]:
        _tab_noise(cfg)


if __name__ == "__main__":
    main()
