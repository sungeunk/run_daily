#!/usr/bin/env python3
"""Streamlit viewer for the daily benchmark DuckDB.

Run with::

    streamlit run daily/viewer/app.py -- --db daily_output/<machine>/bench.duckdb

The DB is built by ``python -m viewer.ingest.cli``. The sidebar can refresh
the configured daily DB by running the local ingestion script.

Tabs
----
1. Dashboard    — fleet status, failing models and per-machine geomean trend.
2. Excel Paste  — wide matrix for a fixed display profile, selected runs
                  become columns.
3. Compare      — run-to-run direct comparison at the series level.
4. Exclusions   — manually hide a specific machine+run from every
                  cohort-based view, e.g. a run that measured too few models
                  and would otherwise shrink the common-series comparison for
                  the rest of the cohort.

The machine is chosen per tab, not in the sidebar.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make `viewer.queries` importable when launched via `streamlit run`.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from viewer import queries as q  # noqa: E402

# ---------------------------------------------------------------------------
# Config / connection
# ---------------------------------------------------------------------------

DEFAULT_DB = _HERE.parents[1] / "daily_output" / platform.node() / "bench.duckdb"
INGEST_SCRIPT = Path("/var/www/html/daily2/ingest_db.sh")
# Same lock the script takes; held here so a concurrent refresh queues up
# instead of racing the DuckDB write lock.
INGEST_LOCK = Path("/var/www/html/daily2/.ingest.lock")
INGEST_WAIT_SEC = 60.0


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


def _cache_version() -> float:
    """Cache key: cached frames must also drop when the query code changes,
    not only when the DB is rebuilt."""
    return max(_db_version(), Path(q.__file__).stat().st_mtime)


def _wait_for_ingest_lock(handle, progress, timeout: float = INGEST_WAIT_SEC) -> bool:
    """Take the ingestion lock, queueing behind a concurrent refresh."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            progress.progress(15, text="Preparing database refresh...")
            time.sleep(2)


def _refresh_database() -> None:
    """Rebuild the daily benchmark DB and invalidate cached query results."""
    if not INGEST_SCRIPT.is_file() or not os.access(INGEST_SCRIPT, os.X_OK):
        st.error(f"Ingestion script is unavailable: {INGEST_SCRIPT}")
        return

    progress = st.progress(5, text="Starting database refresh...")
    with st.status("Refreshing database...", expanded=True) as status:
        try:
            lock = INGEST_LOCK.open("a+")
        except OSError as exc:
            status.update(label="Database refresh failed", state="error")
            progress.empty()
            st.error(f"Cannot open ingestion lock {INGEST_LOCK}: {exc}")
            return

        with lock:
            version_before = _db_version()
            if not _wait_for_ingest_lock(lock, progress):
                status.update(label="Database refresh timed out", state="error")
                progress.empty()
                st.error("The database was not updated: refresh timed out after "
                         f"{int(INGEST_WAIT_SEC)} seconds. Please try again.")
                return

            if _db_version() > version_before:
                message = "Database is already up to date — reloaded."
            else:
                st.write("Running ingestion script")
                progress.progress(25, text="Ingesting daily benchmark artifacts...")
                result = subprocess.run(
                    [str(INGEST_SCRIPT)],
                    cwd=INGEST_SCRIPT.parent,
                    capture_output=True,
                    check=False,
                    text=True,
                    env={**os.environ, "INGEST_LOCK_HELD": "1"},
                )

                if result.returncode != 0:
                    status.update(label="Database refresh failed", state="error")
                    progress.empty()
                    st.error(f"Ingestion failed with exit code {result.returncode}.")
                    output = "\n".join(part for part in (result.stdout, result.stderr)
                                       if part)
                    if output:
                        st.code(output, language="text")
                    return
                message = "Database refresh completed."

            progress.progress(90, text="Reloading cached data...")
            st.cache_data.clear()
            status.update(label="Database refresh completed", state="complete")

    progress.progress(100, text="Database is up to date")
    st.session_state["db_refresh_message"] = message
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


@st.cache_data(show_spinner=False)
def cached_exclusions(_v: float) -> pd.DataFrame:
    return q.list_exclusions(DB)



# --- count-based cohort helpers -------------------------------------------
# Every analysis tab resolves its scope through these so the run set, run kind
# and model selection stay consistent across the app.

@st.cache_data(show_spinner=False)
def cached_cohort(machine: str, limit: int, run_kinds: tuple[str, ...],
                  min_series: int, _v: float) -> pd.DataFrame:
    return q.recent_runs(DB, machine, limit=limit, run_kinds=run_kinds,
                         min_success_series=min_series)


@st.cache_data(show_spinner=False)
def cached_series_runs(machine: str, model: str, precision: str,
                       in_token: int, out_token: int, exec_mode: str,
                       runs_n: int, run_kinds: tuple[str, ...],
                       min_series: int, _v: float) -> pd.DataFrame:
    return q.series_history_for_runs(DB, machine, model=model,
                                     precision=precision, in_token=in_token,
                                     out_token=out_token, exec_mode=exec_mode,
                                     runs_n=runs_n, run_kinds=run_kinds,
                                     min_success_series=min_series)


@st.cache_data(show_spinner=False)
def cached_geomean_matrix(machines: tuple[str, ...], limit: int,
                          run_kinds: tuple[str, ...], min_series: int,
                          models: tuple[str, ...], _v: float) -> pd.DataFrame:
    return q.geomean_matrix(DB, machines, limit=limit, run_kinds=run_kinds,
                            min_success_series=min_series,
                            models=models or None)


@st.cache_data(show_spinner=False)
def cached_machine_health(run_ids: tuple[str, ...], _v: float) -> pd.DataFrame:
    return q.machine_health_for_runs(DB, run_ids)


@st.cache_data(show_spinner=False)
def cached_trend_compare(machine: str, run_a: str, run_b: str,
                         history_n: int, run_kinds: tuple[str, ...],
                         models: tuple[str, ...], min_series: int,
                         _v: float) -> pd.DataFrame:
    return q.compare_runs_with_trend(DB, machine, run_a, run_b,
                                     history_runs_n=history_n,
                                     run_kinds=run_kinds,
                                     models=models or None,
                                     min_success_series=min_series)


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


# Time flows left-to-right in the charts and top-to-bottom in the tables.
# Queries return oldest first, which is also the chronological order the trend
# maths needs, so charts can plot the rows as they come.

def _newest_right(fig: go.Figure) -> None:
    """Put the newest run on the right of a time-ordered chart."""
    fig.update_xaxes(autorange=True)


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
    if _db_version() == 0.0:
        st.sidebar.error(f"DB not found at {DB}")
        st.stop()
    v = _cache_version()
    st.sidebar.caption(f"DB: `{DB}`")
    if st.sidebar.button("Refresh database", width="stretch"):
        _refresh_database()
    if message := st.session_state.pop("db_refresh_message", None):
        st.sidebar.success(message)

    all_machines = cached_machines(v)
    if not all_machines:
        st.sidebar.warning("No runs in DB yet — run `viewer.ingest.cli` first.")
        st.stop()

    profile_options = cached_profiles(v) or ["default"]
    if len(profile_options) == 1:
        profile = profile_options[0]
        st.sidebar.caption(f"Display profile: `{profile}`")
    else:
        profile = st.sidebar.selectbox("Display profile", profile_options)

    st.sidebar.divider()
    y_scale_label = st.sidebar.selectbox(
        "Chart y-axis range", list(Y_SCALE_OPTIONS),
        index=list(Y_SCALE_OPTIONS).index(DEFAULT_Y_SCALE),
        help="Minimum span the y-axis covers. Auto-fit makes a 998-1003 ms "
             "series fill the chart; a ±10% floor keeps it visibly flat.")

    return dict(v=v, profile=profile,
                y_scale=Y_SCALE_OPTIONS[y_scale_label])

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _scope_controls(cfg: dict, key_prefix: str,
                    containers: list | None = None) -> dict:
    """Analysis scope, rendered per tab so each view owns its own window."""
    cols = containers if containers is not None else st.columns(3)
    # History depth is counted in runs, not days: benchmark cadence is
    # irregular, so a day window silently under-samples a rig that skipped a
    # night and over-samples one that ran twice.
    history_runs = cols[0].slider(
        "History (runs)", 3, 20, 10, 1, key=f"{key_prefix}_history",
        help="How many past runs each view covers.")
    purpose = cols[1].text_input(
        "Purpose", value="daily", key=f"{key_prefix}_purpose",
        help="Matched as a case-insensitive substring of a run's "
             "purpose/description, so a PR number or username works too. "
             "The canonical kinds (daily, pr, test, manual) match the run "
             "kind instead. Empty means every run.")
    min_series = cols[2].slider(
        "Min successful series per run", 0, 120, 70, 5,
        key=f"{key_prefix}_min_series",
        help="Runs at or below this many successful series are dropped. "
             "Cohort metrics only cover the series all runs measured, so one "
             "partial run collapses the comparison.")

    purpose = purpose.strip()
    return {**cfg, "history_runs": history_runs,
            "run_kinds": (purpose,) if purpose else (),
            "min_series": min_series}

def _cohort(cfg: dict, machine: str) -> pd.DataFrame:
    """Runs in scope for the current sidebar filters, oldest first."""
    return cached_cohort(machine, cfg["history_runs"],
                         cfg["run_kinds"], cfg["min_series"], cfg["v"])


def _machine_picker(cfg: dict, key: str) -> str:
    """Per-tab machine selector; the sidebar no longer owns this choice."""
    machines = list(_machines_in_scope(cfg))
    if not machines:
        st.warning("No machines in DB.")
        st.stop()
    return st.selectbox("Machine", machines, key=key)


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
                             run_kinds: tuple[str, ...], history_runs: int,
                             _v: float) -> pd.DataFrame:
    return q.machines_overview(DB, machines or None, run_kinds=run_kinds,
                               history_runs=history_runs)


@st.cache_data(show_spinner=False)
def cached_failing_models(machines: tuple[str, ...],
                          run_kinds: tuple[str, ...], history_runs: int,
                          _v: float) -> pd.DataFrame:
    return q.failing_models_overview(DB, machines or None, run_kinds=run_kinds,
                                     history_runs=history_runs)


@st.cache_data(show_spinner=False)
def cached_env_changes(machines: tuple[str, ...], run_kinds: tuple[str, ...],
                       history_runs: int, _v: float) -> pd.DataFrame:
    return q.environment_changes(DB, machines or None, run_kinds=run_kinds,
                                 history_runs=history_runs)


def _report_url(machine: str, stamp: str, suffix: str) -> str:
    month = f"{stamp[:4]}.{stamp[4:6]}" if len(stamp) >= 6 else ""
    bucket = f"{machine}/{month}" if month else machine
    return f"{REPORT_BASE_URL}/{bucket}/daily.{stamp}.{suffix}"


def _fleet_records(overview: pd.DataFrame) -> pd.DataFrame:
    """Two rows per machine: its newest clean run and its newest failed run."""
    # Counts are benchmark cases (perf series), not pytest functions: one LLM
    # test covers 2 prompts x 1st/2nd = 4 cases, resnet50 covers 2 batches.
    total = overview["expected_cases"].fillna(0)
    latest_success = overview["success_cases"].fillna(0)
    # Same verdict the last success / last fail rows are built from.
    is_fail = overview["latest_failed"].fillna(False).astype(bool)
    stale = overview["age_hours"] > 24

    def _status(idx: int) -> str:
        if is_fail.iloc[idx]:
            return "🔴 failed"
        if stale.iloc[idx]:
            return "🟡 stale"
        if latest_success.iloc[idx] > 0:
            return "🟢 success"
        return "⚪ unknown"

    def _text(value: object) -> str:
        return "" if value is None or pd.isna(value) else str(value)

    def _count(value: object) -> int:
        return 0 if value is None or pd.isna(value) else int(value)

    def _console_url(build_url: object) -> str | None:
        # Only runs launched by Jenkins carry a build URL.
        if build_url is None or pd.isna(build_url) or not str(build_url):
            return None
        return str(build_url).rstrip("/") + "/consoleText"

    def _edge_record(row, kind: str, expected: int, status: str,
                     device: str) -> dict:
        """One table row describing a machine's newest clean or failed run."""
        label = "🟢 last success" if kind == "last_success" else "🔴 last fail"
        record = {"Machine": row.machine, "Status": status, "Run": label,
                  "newest": False, "device": device, "Stamp": "never",
                  "report": None, "jenkins log": None, "Purpose": "",
                  "OV version": "",
                  "total": None, "skip": None, "success": None,
                  "failed": None, "Duration (min)": None}
        stamp = getattr(row, f"{kind}_stamp")
        if stamp is None or pd.isna(stamp):
            return record

        stamp = str(stamp)
        skip = _count(getattr(row, f"{kind}_skipped_cases"))
        success = _count(getattr(row, f"{kind}_success_cases"))
        duration = getattr(row, f"{kind}_duration_sec")
        record.update({
            "newest": stamp == str(row.stamp),
            "Stamp": stamp[4:] if len(stamp) >= 13 else stamp,
            "report": _report_url(row.machine, stamp, "html"),
            "jenkins log": _console_url(getattr(row, f"{kind}_build_url")),
            "Purpose": _text(getattr(row, f"{kind}_purpose")),
            "OV version": _text(getattr(row, f"{kind}_ov_version")),
            "total": expected,
            "skip": skip,
            "success": success,
            "failed": max(expected - success - skip, 0),
            "Duration (min)": None if pd.isna(duration) else duration / 60,
        })
        return record

    records = []
    for i, row in enumerate(overview.itertuples(index=False)):
        expected = int(total.iloc[i])
        device = q.short_device_name(row.gpu_name)
        for kind in ("last_success", "last_fail"):
            records.append(_edge_record(row, kind, expected, _status(i), device))
    return pd.DataFrame(records)


_CASE_COLOURS = {"success": "#1f77b4", "skip": "#9e9e9e", "failed": "#d62728"}


def _case_bar(success: int, skip: int, failed: int, total: int) -> str:
    """Stacked bar of a run's case outcomes; st.progress is single-colour."""
    segments = "".join(
        f'<div style="width:{value / total * 100:.2f}%;'
        f'background:{_CASE_COLOURS[name]};"></div>'
        for name, value in (("success", success), ("skip", skip),
                            ("failed", failed)) if value > 0
    )
    return ('<div style="display:flex;height:10px;border-radius:5px;'
            f'overflow:hidden;background:#eeeeee;">{segments}</div>')


def _run_card(column, record: pd.Series) -> None:
    """One edge run as a compact card: identity, links, then its counters."""
    with column.container(border=True):
        title = record["Run"] + (" ◀ newest" if record["newest"] else "")
        st.markdown(f"**{title}**")
        if record["Stamp"] == "never":
            st.caption("No such run in this window.")
            return

        # Angle-bracket destinations: Jenkins URLs contain parentheses.
        links = [f"[report](<{record['report']}>)"]
        if record["jenkins log"]:
            links.append(f"[jenkins log](<{record['jenkins log']}>)")
        st.markdown(f"`{record['Stamp']}` &nbsp; "
                    + " &nbsp;·&nbsp; ".join(links))
        st.caption(f"purpose: {record['Purpose'] or '—'}  \n"
                   f"OpenVINO: {record['OV version'] or '—'}")

        total = int(record["total"] or 0)
        success = int(record["success"] or 0)
        skip = int(record["skip"] or 0)
        failed = int(record["failed"] or 0)
        st.markdown(f"total {total} / :blue[success {success}] / "
                    f":gray[skip {skip}] / :red[failed {failed}]")
        if total:
            st.markdown(_case_bar(success, skip, failed, total),
                        unsafe_allow_html=True)
        duration = record["Duration (min)"]
        st.caption("duration: —" if pd.isna(duration)
                   else f"duration: {duration:.1f} min")



def _failing_models_view(df: pd.DataFrame) -> pd.DataFrame:
    """Failing-model rows of one machine, formatted for display."""
    def _cache_cell(name: object, changed: object, previous: object) -> str:
        text = "" if name is None or pd.isna(name) else str(name)
        if not changed:
            return text
        return f"⚠ {text} (was {previous})"

    return pd.DataFrame({
        "Model": df["model"],
        "Prec": df["precision"].fillna(""),
        "Model cache": [_cache_cell(n, c, p) for n, c, p
                        in zip(df["model_cache"], df["model_cache_changed"],
                               df["last_pass_model_cache"])],
        "State": ["🔴 also failed before" if before else "🆕 new in latest run"
                  for before in df["failed_before"]],
        "Failed runs": [f"{int(a)} / {int(b)}" for a, b
                        in zip(df["failed_runs"], df["window_runs"])],
        "First seen": df["first_seen"],
        "Last passed": df["last_pass_stamp"].fillna("—"),
    })


_METRIC_LABEL = {"1st": "1st token", "2nd": "2nd token",
                 "pipeline": "pipeline", "throughput": "throughput"}


def _machines_in_scope(cfg: dict) -> tuple[str, ...]:
    """Machines the dashboard reports on: the curated daily rigs present in
    the DB, since one-off folders linger under the report root."""
    machines = cached_machines(cfg["v"])
    return tuple(m for m in machines if m in DAILY_MACHINES) or tuple(machines)


def _ring_newest(fig: go.Figure, panel: pd.DataFrame, y_column: str, col: int,
                 y_range: list[float] | None) -> None:
    """Dotted ring around the newest run so the eye lands on it."""
    if panel.empty:
        return
    values = pd.to_numeric(panel[y_column], errors="coerce").dropna()
    if values.empty:
        return
    y = float(values.iloc[-1])
    low, high = (y_range if y_range is not None
                 else [float(values.min()), float(values.max())])
    span = (high - low) or max(abs(y) * 0.1, 1.0)
    # Category axes place the i-th run at x=i, so the newest sits at n-1.
    x = len(panel) - 1
    fig.add_shape(
        type="circle", x0=x - 0.28, x1=x + 0.28,
        y0=y - span * 0.07, y1=y + span * 0.07,
        line=dict(color="#212121", width=1.5, dash="dot"),
        row=1, col=col,
    )


def _machine_geomean_figure(frame: pd.DataFrame, cfg: dict) -> go.Figure:
    """One row of panels for a machine: a geomean per metric, then successes."""
    metrics = [m for m in q.GEOMEAN_METRICS if (frame["metric"] == m).any()]

    titles = []
    for metric in metrics:
        panel = frame[frame["metric"] == metric]
        unit = str(panel["unit"].iloc[0] or "")
        n_series = int(panel["n_series"].iloc[0])
        titles.append(f"{_METRIC_LABEL[metric]} [{unit}] · {n_series} series")
    titles.append("success count")

    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles)
    for col, metric in enumerate(metrics, start=1):
        panel = frame[frame["metric"] == metric].sort_values("ts")
        fig.add_trace(go.Scatter(
            x=panel["stamp"].str[4:], y=panel["geomean"],
            mode="lines+markers", showlegend=False, marker=dict(size=7),
            text=[f"ov={v}<br>purpose={p}" for v, p in
                  zip(panel["ov_version"].fillna(""),
                      panel["purpose"].fillna(""))],
            hovertemplate="%{x}<br>%{y:.2f}<br>%{text}<extra></extra>",
        ), row=1, col=col)
        y_range = _stable_y_range(panel["geomean"], cfg["y_scale"])
        if y_range is not None:
            fig.update_yaxes(range=y_range, row=1, col=col)
        _ring_newest(fig, panel, "geomean", col, y_range)

    runs = frame.drop_duplicates("run_id").sort_values("ts")
    counts = runs["success_count"]
    fig.add_trace(go.Scatter(
        x=runs["stamp"].str[4:], y=counts, mode="lines+markers",
        showlegend=False, marker=dict(size=7),
        text=[f"ov={v}<br>purpose={p}" for v, p in
              zip(runs["ov_version"].fillna(""), runs["purpose"].fillna(""))],
        hovertemplate="%{x}<br>%{y} cases<br>%{text}<extra></extra>",
    ), row=1, col=len(titles))
    y_range = _stable_y_range(counts, cfg["y_scale"])
    if y_range is not None:
        fig.update_yaxes(range=y_range, row=1, col=len(titles))
    _ring_newest(fig, runs, "success_count", len(titles), y_range)

    fig.update_layout(height=300, margin=dict(t=52, b=44, l=8, r=8))
    fig.update_annotations(font_size=12)
    fig.update_xaxes(autorange=True, tickangle=-45, tickfont_size=10)
    return fig


def _machine_card(machine: str, records: pd.DataFrame,
                  matrix: pd.DataFrame, failing: pd.DataFrame,
                  env: pd.DataFrame, cfg: dict) -> None:
    """One machine's edge runs, perf trend and failing models in one block."""
    group = records[records["Machine"] == machine]
    with st.container(border=True):
        head = group.iloc[0]
        st.markdown(f"##### {head['Status']} &nbsp; {machine} &nbsp; "
                    f"`{head['device']}`")

        fails = (failing[failing["machine"] == machine]
                 if not failing.empty else failing)
        if fails.empty:
            st.caption("Failing models in the latest run: none.")
        else:
            st.caption("Failing models in the latest run")
            st.dataframe(_failing_models_view(fails), width="stretch",
                         hide_index=True)

        notes = env[env["machine"] == machine] if not env.empty else env
        if not notes.empty:
            st.info("Rig changed in this window:\n" + "\n".join(
                f"- **{r.field}**: `{r.previous}` → `{r.current}` "
                f"since `{r.stamp}` ({r.ov_version or '?'})"
                for r in notes.itertuples()))

        left, right = st.columns(2)
        _run_card(left, group.iloc[0])
        _run_card(right, group.iloc[1])

        frame = (matrix[matrix["machine"] == machine]
                 if not matrix.empty else matrix)
        if frame.empty:
            st.caption("Performance trend: no data in the selected window.")
        else:
            plotted = frame["run_id"].nunique()
            missing = int(frame["cohort_runs"].iloc[0]) - plotted
            note = f"Performance trend · {plotted} runs"
            if missing > 0:
                note += f" · {missing} run(s) produced no data"
            st.caption(note)
            st.plotly_chart(_machine_geomean_figure(frame, cfg),
                            width="stretch", key=f"geomean_{machine}")


def _tab_dashboard(cfg: dict) -> None:
    cols = st.columns([2, 1, 1.5, 1.5], vertical_alignment="bottom")
    query = cols[0].text_input(
        "Machine filter", key="dashboard_machine_filter",
        placeholder="Type part of a machine name, e.g. LNL")
    cfg = _scope_controls(cfg, "dashboard", cols[1:])

    scope = _machines_in_scope(cfg)
    overview = cached_machines_overview(scope, cfg["run_kinds"],
                                        cfg["history_runs"], cfg["v"])
    if overview.empty:
        st.info("No runs match the current filters.")
        return

    names = [m for m in overview["machine"]
             if not query.strip() or query.strip().lower() in m.lower()]
    if not names:
        st.info(f"No machine matches '{query}'.")
        return
    st.caption(
        f"{len(names)} of {len(overview)} machines. Counts are benchmark "
        "cases, not pytest tests: one LLM test contributes 2 prompts x "
        "1st/2nd = 4 cases. A run counts as failed when pytest reported a "
        "failure or error, when it executed nothing, or when it produced "
        "fewer cases than a full run. Each geomean covers only the series "
        "every run of that machine measured, so a run that lost models moves "
        "the success count instead of the geomean.")

    records = _fleet_records(overview)
    matrix = cached_geomean_matrix(
        _machines_in_scope(cfg), cfg["history_runs"],
        cfg["run_kinds"], cfg["min_series"], (), cfg["v"])
    failing = cached_failing_models(scope, cfg["run_kinds"],
                                    cfg["history_runs"], cfg["v"])
    env = cached_env_changes(scope, cfg["run_kinds"], cfg["history_runs"],
                             cfg["v"])

    for machine in names:
        _machine_card(machine, records, matrix, failing, env, cfg)


def _tab_excel(cfg: dict) -> None:
    machine = _machine_picker(cfg, "excel_machine")
    runs = cached_runs(machine, cfg["v"])
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


def _tab_exclusions(cfg: dict) -> None:
    """Manually hide a specific machine+run from every cohort-based view.

    The Dashboard and Compare views resolve their scope through
    ``recent_runs`` and restrict each metric to the series *every* run in
    the cohort measured (see ``geomean_matrix``'s docstring) — a single run
    that only measured a handful of models (partial/broken run) collapses
    that intersection for the whole cohort, hiding good older builds too.
    Excluding it here removes it from the cohort instead. The Excel tab is
    unaffected: it lets you pick any run on purpose, excluded or not.
    """
    st.caption("Hide a run from the Dashboard/Compare cohort so a sparse or "
               "broken build stops shrinking the common-series comparison "
               "for the rest of the cohort.")

    machine = _machine_picker(cfg, "exclusions_machine")
    runs = cached_runs(machine, cfg["v"])
    if runs.empty:
        st.info(f"No runs for {machine}.")
    else:
        st.markdown(f"**Runs on {machine}**")
        event = st.dataframe(
            runs[["stamp", "ww", "ov_version", "purpose", "source_format"]],
            width="stretch",
            hide_index=True,
            selection_mode="multi-row",
            on_select="rerun",
            key="exclusion_run_table",
        )
        sel = event.selection.rows if event and event.selection else []
        reason = st.text_input("Reason (optional)", key="exclusion_reason")
        if st.button("Exclude selected", disabled=not sel):
            for i in sel:
                row = runs.iloc[i]
                q.add_exclusion(DB, row["run_id"], row["machine"],
                                row["stamp"], reason)
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.markdown("**Currently excluded** (all machines)")
    excluded = cached_exclusions(cfg["v"])
    if excluded.empty:
        st.caption("_None._")
        return

    event2 = st.dataframe(
        excluded[["machine", "stamp", "reason", "excluded_at"]],
        width="stretch",
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="restore_run_table",
    )
    sel2 = event2.selection.rows if event2 and event2.selection else []
    if st.button("Restore selected", disabled=not sel2):
        for i in sel2:
            q.remove_exclusion(DB, excluded.iloc[i]["run_id"])
        st.cache_data.clear()
        st.rerun()


def _tab_compare(cfg: dict) -> None:
    st.caption("A raw A-vs-B delta cannot separate a real change from ordinary "
               "run-to-run scatter, so each series is also compared against the "
               "history preceding each run.")

    machine = _machine_picker(cfg, "compare_machine")
    cfg = _scope_controls(cfg, "compare")
    cohort = _cohort(cfg, machine)
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

    df = cached_trend_compare(machine, run_a, run_b, cfg["history_runs"],
                              cfg["run_kinds"], (),
                              cfg["min_series"], cfg["v"])
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
        machine, row["model"], row["precision"],
        int(row["in_token"]), int(row["out_token"]), row["exec_mode"],
        cfg["history_runs"], cfg["run_kinds"],
        cfg["min_series"], cfg["v"])
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
    _newest_right(fig)
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

    tabs = st.tabs(["Dashboard", "Excel", "Compare", "Exclusions"])
    with tabs[0]:
        _tab_dashboard(cfg)
    with tabs[1]:
        _tab_excel(cfg)
    with tabs[2]:
        _tab_compare(cfg)
    with tabs[3]:
        _tab_exclusions(cfg)


if __name__ == "__main__":
    main()
