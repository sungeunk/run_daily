"""DuckDB query helpers used by the Streamlit viewer.

All functions accept a path to the DuckDB file and open a read-only
connection per call. Streamlit caches at the dataframe level, so the
per-call connection cost is paid once per cache bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import logging
import re
import time

import duckdb
import pandas as pd

from analysis.types import AnalysisConfig
from analysis.verdict import improvement_pct, verdict_from_pct

log = logging.getLogger(__name__)


_COMPARE_CONFIG = AnalysisConfig()

# Cache table existence per database file + mtime so the cache stays bounded
# and automatically refreshes when the DuckDB file is rewritten.


def _read_only(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


@lru_cache(maxsize=8)
def _cached_tables(db_path_str: str, mtime_ns: int) -> frozenset[str]:
    with _read_only(Path(db_path_str)) as con:
        return frozenset(
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        )


def _tables_for_db(db_path: Path) -> set[str]:
    stat = db_path.stat()
    return set(_cached_tables(str(db_path), stat.st_mtime_ns))


@lru_cache(maxsize=32)
def _cached_columns(db_path_str: str, relation: str,
                    mtime_ns: int) -> frozenset[str]:
    with _read_only(Path(db_path_str)) as con:
        try:
            return frozenset(r[0] for r in
                             con.execute(f"DESCRIBE {relation}").fetchall())
        except duckdb.Error:
            return frozenset()


def _has_column(db_path: Path, relation: str, column: str) -> bool:
    """Whether a table/view exposes a column.

    The viewer opens the DB read-only and so cannot migrate it. A deployment
    where ingest has not yet run with the current schema must degrade to the
    columns it does have instead of failing to render.
    """
    stat = db_path.stat()
    return column in _cached_columns(str(db_path), relation, stat.st_mtime_ns)


def _fill_missing_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing verdicts with the canonical analysis threshold logic using vectorization."""
    if df.empty:
        return df
    if "verdict" not in df.columns:
        df["verdict"] = pd.NA

    # Use verdict_from_pct() vectorized via apply on missing rows only (idiomatic pandas)
    mask_missing = df["verdict"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "verdict"] = df.loc[mask_missing, "improvement_pct"].apply(
            lambda pct: verdict_from_pct(None if pd.isna(pct) else float(pct), _COMPARE_CONFIG)
        )
    return df


def _apply_fallback_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive improvement_pct/verdict for raw fallback rows via canonical helpers."""
    if df.empty:
        return df

    def _to_float(value) -> float | None:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _row_metrics(row: pd.Series) -> tuple[float | None, str]:
        current_unit = row.get("current_unit")
        baseline_unit = row.get("baseline_unit")
        unit = row.get("unit")
        cur = _to_float(row.get("value_a"))
        base = _to_float(row.get("value_b"))

        if current_unit is not None and baseline_unit is not None and current_unit != baseline_unit:
            return None, "unavailable"

        pct = improvement_pct(cur, base, unit)
        return pct, verdict_from_pct(pct, _COMPARE_CONFIG)

    metrics = df.apply(_row_metrics, axis=1)
    df["improvement_pct"] = [m[0] for m in metrics]
    if "verdict" not in df.columns:
        df["verdict"] = pd.NA
    df["verdict"] = df["verdict"].where(df["verdict"].notna(), [m[1] for m in metrics])
    return df


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def list_machines(db_path: Path) -> list[str]:
    with _read_only(db_path) as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT machine FROM runs ORDER BY machine").fetchall()]


# ---------------------------------------------------------------------------
# Run cohort: the shared "which runs are we looking at" primitive.
#
# Every analysis view resolves its scope through here so the tabs agree on
# machine, run kind and history depth. History depth is a *run count* because
# benchmark cadence is irregular — a day-based window silently under- or
# over-samples depending on how often the rig happened to run.
# ---------------------------------------------------------------------------

RUN_KINDS = ("daily", "pr", "test", "manual")
DEFAULT_RUN_KINDS = ("daily",)


def _run_kind_clause(run_kinds: Sequence[str] | None,
                     alias: str = "r") -> tuple[str, list]:
    if not run_kinds:
        return "", []
    placeholders = ",".join(["?"] * len(run_kinds))
    return (f" AND COALESCE({alias}.run_kind, 'daily') IN ({placeholders})",
            list(run_kinds))


def list_run_kinds(db_path: Path, machine: str | None = None) -> list[str]:
    if not _has_column(db_path, "runs", "run_kind"):
        return ["daily"]
    where = "" if machine is None else "WHERE machine = ?"
    params = [] if machine is None else [machine]
    with _read_only(db_path) as con:
        rows = con.execute(f"""
            SELECT DISTINCT COALESCE(run_kind, 'daily') AS run_kind
            FROM runs {where}
        """, params).fetchall()
    found = {r[0] for r in rows if r[0]}
    return [k for k in RUN_KINDS if k in found] + sorted(found - set(RUN_KINDS))


def recent_runs(db_path: Path, machine: str, *, limit: int = 10,
                run_kinds: Sequence[str] | None = DEFAULT_RUN_KINDS,
                include_short_run: bool = True,
                before_ts=None) -> pd.DataFrame:
    """Return the newest ``limit`` runs matching the filters, oldest first."""
    has_kind = _has_column(db_path, "runs", "run_kind")
    clause, params = _run_kind_clause(run_kinds if has_kind else None)
    kind_select = ("COALESCE(run_kind, 'daily')" if has_kind else "'daily'")
    short_clause = "" if include_short_run else " AND COALESCE(r.short_run, FALSE) = FALSE"
    ts_clause = ""
    ts_params: list = []
    if before_ts is not None:
        ts_clause = " AND r.ts <= ?"
        ts_params = [before_ts]

    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT run_id, machine, ts,
                   strftime(ts, '%Y%m%d_%H%M') AS stamp,
                   ww, ov_version, ov_build, ov_sha,
                   purpose, {kind_select} AS run_kind,
                   description, short_run, source_format,
                   source_path, rawlog_path
            FROM (
                SELECT r.*
                FROM runs r
                WHERE r.machine = ?{clause}{short_clause}{ts_clause}
                ORDER BY r.ts DESC
                LIMIT ?
            )
            ORDER BY ts
        """, [machine, *params, *ts_params, int(limit)]).fetchdf()


def list_models(db_path: Path, machine: str | None = None,
                run_kinds: Sequence[str] | None = None) -> list[str]:
    """Model names available for filtering, optionally scoped to a machine."""
    filters = []
    params: list = []
    if machine:
        filters.append("r.machine = ?")
        params.append(machine)
    if run_kinds and _has_column(db_path, "runs", "run_kind"):
        placeholders = ",".join(["?"] * len(run_kinds))
        filters.append(f"COALESCE(r.run_kind, 'daily') IN ({placeholders})")
        params.extend(run_kinds)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    with _read_only(db_path) as con:
        return [r[0] for r in con.execute(f"""
            SELECT DISTINCT p.model
            FROM perf p JOIN runs r USING (run_id)
            {where}
            ORDER BY p.model
        """, params).fetchall()]


def perf_for_runs(db_path: Path, run_ids: Sequence[str], *,
                  models: Sequence[str] | None = None,
                  exec_modes: Sequence[str] | None = None) -> pd.DataFrame:
    """Perf rows for an explicit run cohort, with run metadata attached."""
    if not run_ids:
        return pd.DataFrame()
    filters = ["f.run_id IN ({})".format(",".join(["?"] * len(run_ids)))]
    params: list = list(run_ids)
    if models:
        filters.append("f.model IN ({})".format(",".join(["?"] * len(models))))
        params.extend(models)
    if exec_modes:
        filters.append("f.exec_mode IN ({})".format(",".join(["?"] * len(exec_modes))))
        params.extend(exec_modes)
    kind_select = ("f.run_kind" if _has_column(db_path, "perf_flat", "run_kind")
                   else "'daily'")
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT f.run_id, f.ts, f.ww, f.ov_version, {kind_select} AS run_kind,
                   f.model, f.precision, f.in_token, f.out_token,
                   f.exec_mode, f.value, f.unit
            FROM perf_flat f
            WHERE {' AND '.join(filters)}
              AND f.value > 0
            ORDER BY f.ts, f.model, f.precision, f.in_token, f.out_token, f.exec_mode
        """, params).fetchdf()


def list_runs(db_path: Path, machine: str | None = None) -> pd.DataFrame:
    """Return runs metadata with one row per run, newest first."""
    where = "" if machine is None else "WHERE machine = ?"
    params = [] if machine is None else [machine]
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT run_id,
                   machine,
                   ts,
                   strftime(ts, '%Y%m%d_%H%M') AS stamp,
                   ww,
                   ov_version,
                   ov_build,
                   ov_sha,
                   purpose,
                   description,
                   short_run,
                   source_format,
                   report_file,
                     source_path,
                   rawlog_path
            FROM runs
            {where}
            ORDER BY ts DESC
        """, params).fetchdf()


# ---------------------------------------------------------------------------
# Excel-paste: ordered rows for selected runs using display_rows
# ---------------------------------------------------------------------------

def list_profiles(db_path: Path) -> list[str]:
    with _read_only(db_path) as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT profile FROM display_rows ORDER BY profile").fetchall()]


def success_counts(db_path: Path, run_ids: list[str]) -> dict[str, int]:
    """Count of perf value rows ingested per run — mirrors the legacy
    viewer's 'Success count' (number of benchmark data points that produced
    a parsable value), independent of pytest pass/fail status."""
    if not run_ids:
        return {}
    placeholders = ",".join(["?"] * len(run_ids))
    with _read_only(db_path) as con:
        rows = con.execute(
            f"SELECT run_id, count(*) FROM perf WHERE run_id IN ({placeholders}) "
            "GROUP BY run_id",
            run_ids,
        ).fetchall()
    return {run_id: count for run_id, count in rows}


def legacy_geomean_summary(db_path: Path, run_ids: list[str]) -> pd.DataFrame:
    """Reproduce the legacy report viewer's summary geomean rows per run:
    'geomean' (all perf values, SD seconds converted to ms to match the old
    unit convention) plus 'geomean (LLM/<1st|2nd>/<Short|Long>)' — LLM-only
    (exec_mode in ('1st','2nd')), bucketed by raw in_token (<400 short,
    401-1200 long, matching the legacy classify_token_size thresholds)."""
    if not run_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(run_ids))
    sql = f"""
    WITH base AS (
        SELECT run_id, exec_mode, in_token,
               CASE WHEN unit = 's' THEN value * 1000 ELSE value END AS value
        FROM perf
        WHERE run_id IN ({placeholders}) AND value > 0
    )
    SELECT
        run_id,
        exp(avg(ln(value))) AS geomean,
        exp(avg(ln(value)) FILTER (
            WHERE exec_mode = '2nd' AND in_token > 0 AND in_token < 400)) AS geomean_2nd_short,
        exp(avg(ln(value)) FILTER (
            WHERE exec_mode = '1st' AND in_token > 0 AND in_token < 400)) AS geomean_1st_short,
        exp(avg(ln(value)) FILTER (
            WHERE exec_mode = '2nd' AND in_token BETWEEN 401 AND 1200)) AS geomean_2nd_long,
        exp(avg(ln(value)) FILTER (
            WHERE exec_mode = '1st' AND in_token BETWEEN 401 AND 1200)) AS geomean_1st_long
    FROM base
    GROUP BY run_id
    """
    with _read_only(db_path) as con:
        return con.execute(sql, run_ids).fetchdf().set_index("run_id")


def build_excel_matrix(db_path: Path, run_ids: list[str],
                       profile: str = "default") -> pd.DataFrame:
    """Return a wide dataframe: rows = display_rows order, columns = run stamps.

    Matching rules:
      in_spec: '*' → any prompt; otherwise exact match against perf.prompt_idx
               (prompt index is a stable join key — token counts for the
               same prompt slot can shift slightly between models/runs,
               which made bucket-based ('short'/'long') matching brittle).
      out_spec: '*' → any; otherwise exact numeric equality against out_token
    """
    if not run_ids:
        return pd.DataFrame()

    # Preserve the caller's run order. The Excel paste headers are generated
    # from the selected run_ids in app.py, so matrix columns must follow the
    # same order rather than sorting stamps alphabetically.
    run_order = {run_id: idx for idx, run_id in enumerate(run_ids)}
    placeholders = ",".join(["?"] * len(run_ids))

    sql = f"""
    WITH rs AS (
        SELECT run_id, stamp
        FROM (
            SELECT run_id, strftime(ts, '%Y%m%d_%H%M') AS stamp, ts
            FROM runs
            WHERE run_id IN ({placeholders})
        )
    ),
    joined AS (
        SELECT
            d.seq,
            d.model    AS d_model,
            d.precision AS d_precision,
            d.in_spec,
            d.out_spec,
            d.exec_mode AS d_exec,
            d.label,
            rs.run_id,
            rs.stamp,
            p.viewer_value AS value,
            p.in_token,
            p.viewer_unit AS unit
        FROM display_rows d
        CROSS JOIN rs
        LEFT JOIN perf_with_buckets p
          ON p.run_id = rs.run_id
         AND p.model = d.model
         AND p.precision = d.precision
         AND p.exec_mode = d.exec_mode
         AND (
               d.in_spec = '*'
            OR TRY_CAST(d.in_spec AS INTEGER) = p.prompt_idx
         )
         AND (
               d.out_spec = '*'
            OR TRY_CAST(d.out_spec AS INTEGER) = p.out_token
         )
        WHERE d.profile = ?
    )
        SELECT seq, d_model AS model, d_precision AS precision,
           in_spec, out_spec, d_exec AS exec_mode, label,
            run_id, stamp, median(value) AS value,
            max(in_token) AS matched_in_token, max(unit) AS unit
    FROM joined
        GROUP BY seq, d_model, d_precision, in_spec, out_spec, d_exec, label,
              run_id, stamp
        ORDER BY seq
    """
    with _read_only(db_path) as con:
        df = con.execute(sql, [*run_ids, profile]).fetchdf()

    if df.empty:
        return df

    # display_rows.in_spec holds a prompt index (e.g. '0', '1', '2', ...),
    # not a token count — show the actual matched input token count in the
    # preview instead of the raw index number.
    actual_in = df.groupby("seq")["matched_in_token"].max()

    def _resolve_in_spec(spec: str, actual) -> str:
        if spec != "*" and pd.notna(actual):
            return str(int(actual))
        return spec

    df["in_spec"] = [
        _resolve_in_spec(spec, actual_in.get(seq))
        for seq, spec in zip(df["seq"], df["in_spec"])
    ]

    # Build one row per (seq, spec) with a column per run stamp. pivot_table
    # drops NaN index values and also explodes on large cross-products when
    # dropna=False, so we do it manually.
    spec_cols = ["seq", "model", "precision", "in_spec", "out_spec",
                 "exec_mode", "unit", "label"]
    specs = (df[spec_cols]
             .drop_duplicates(subset="seq")
             .sort_values("seq")
             .reset_index(drop=True))
    run_stamps = (df[["run_id", "stamp"]]
                  .drop_duplicates()
                  .assign(_order=lambda frame: frame["run_id"].map(run_order))
                  .sort_values("_order"))
    result = specs.copy()
    for _, run in run_stamps.iterrows():
        vals = (df[df["run_id"] == run["run_id"]]
                .set_index("seq")["value"])
        result[run["stamp"]] = result["seq"].map(vals)
    return result.drop(columns=["seq"])


def extra_rows(db_path: Path, run_ids: list[str],
               profile: str = "default") -> pd.DataFrame:
    """Perf rows in the selected runs that match no display_row (QA/sanity)."""
    if not run_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(run_ids))
    sql = f"""
    WITH m AS (
        SELECT DISTINCT model, precision, in_token, out_token, exec_mode, prompt_idx
        FROM perf
        WHERE run_id IN ({placeholders})
    )
    SELECT m.model, m.precision, m.in_token, m.out_token, m.exec_mode
    FROM m
    LEFT JOIN display_rows d
      ON  d.profile = ?
     AND d.model = m.model
     AND d.precision = m.precision
     AND d.exec_mode = m.exec_mode
     AND (
           d.in_spec = '*'
        OR TRY_CAST(d.in_spec AS INTEGER) = m.prompt_idx
     )
     AND (
           d.out_spec = '*'
        OR TRY_CAST(d.out_spec AS INTEGER) = m.out_token
     )
    WHERE d.profile IS NULL
    ORDER BY m.model, m.precision, m.in_token, m.out_token, m.exec_mode
    """
    with _read_only(db_path) as con:
        return con.execute(sql, [*run_ids, profile]).fetchdf()


# ---------------------------------------------------------------------------
# Trend + Regression
# ---------------------------------------------------------------------------

def series_history(db_path: Path, machine: str, model: str, precision: str,
                   in_token: int, out_token: int, exec_mode: str,
                   days: int = 60,
                   purpose_filter: str | None = None) -> pd.DataFrame:
    """Time-series of one perf point with rolling baseline stats."""
    start = time.time()
    purpose_like = f"%{purpose_filter}%" if purpose_filter else None
    with _read_only(db_path) as con:
        result = con.execute("""
            WITH base AS (
                SELECT ts, date, ov_version, ov_build, ww,
                       value, unit
                FROM perf_flat
                WHERE machine = ?
                  AND model = ? AND precision = ?
                  AND in_token = ? AND out_token = ?
                  AND exec_mode = ?
                  AND (? IS NULL OR COALESCE(purpose, '') ILIKE ?)
                  AND ts >= current_date - (? || ' DAY')::INTERVAL
            ),
            with_baseline AS (
                SELECT
                    b.*,
                    (
                        SELECT median(b2.value)
                        FROM base b2
                        WHERE b2.ts < b.ts
                          AND b2.ts >= b.ts - INTERVAL '30 DAY'
                    ) AS win_median,
                    (
                        SELECT count(*)
                        FROM base b2
                        WHERE b2.ts < b.ts
                          AND b2.ts >= b.ts - INTERVAL '30 DAY'
                    ) AS win_n
                FROM base b
            ),
            with_mad AS (
                SELECT
                    w.*,
                    (
                        SELECT median(abs(b2.value - w.win_median))
                        FROM base b2
                        WHERE b2.ts < w.ts
                          AND b2.ts >= w.ts - INTERVAL '30 DAY'
                    ) AS win_mad
                FROM with_baseline w
            )
            SELECT
                ts, date, ov_version, ov_build, ww,
                value, unit,
                win_median, win_mad, 1.4826 * win_mad AS win_sigma, win_n,
                CASE
                    WHEN win_mad IS NULL OR win_median IS NULL OR win_mad = 0 THEN NULL
                    ELSE (value - win_median) / (1.4826 * win_mad)
                END AS z_score,
                CASE
                    WHEN win_median IS NULL OR win_median = 0 THEN NULL
                    ELSE (value - win_median) / win_median
                END AS pct_diff,
                CASE
                    WHEN win_median IS NULL OR win_median = 0 OR win_mad IS NULL THEN NULL
                    ELSE win_mad / win_median
                END AS cv
            FROM with_mad
            ORDER BY ts
        """, [machine, model, precision, in_token, out_token, exec_mode,
               purpose_filter, purpose_like, days]
                           ).fetchdf()
        elapsed = time.time() - start
        log.debug(f"series_history({model}, {precision}, {in_token}, {out_token}) took {elapsed:.2f}s")
        return result


# ---------------------------------------------------------------------------
# Noise diagnostics: per-series CV across recent window
# ---------------------------------------------------------------------------

def noise_summary(db_path: Path, machine: str | None = None,
                  days: int = 30) -> pd.DataFrame:
    where = "WHERE ts >= current_date - (? || ' DAY')::INTERVAL"
    params: list = [str(days)]
    if machine:
        where += " AND machine = ?"
        params.append(machine)
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT
                machine, model, precision, in_token, out_token, exec_mode, unit,
                count(*)       AS n,
                median(value)  AS median_value,
                stddev_samp(value) AS std_value,
                CASE WHEN median(value) = 0 THEN NULL
                     ELSE stddev_samp(value) / median(value) END AS cv
            FROM perf_flat
            {where}
            GROUP BY ALL
            HAVING count(*) >= 3
            ORDER BY cv DESC NULLS LAST
        """, params).fetchdf()


# ---------------------------------------------------------------------------
# Trend-based regression detection: recent window vs older baseline.
# ---------------------------------------------------------------------------

def trend_regressions(db_path: Path, machine: str,
                      *, recent_days: int = 7,
                      baseline_days: int = 21,
                      min_recent_points: int = 5,
                      min_baseline_points: int = 7,
                      purpose_filter: str | None = None) -> pd.DataFrame:
    """Per-series regression signal based on median comparison between two
    time windows.

    We want to answer 'has this series drifted slower in the last N days
    vs the N-days-before-that window?', not 'is today's point an outlier'.
    Single-point outliers and ingest noise get washed out by the medians.

    Rules:
      - recent window:  last ``recent_days``, ending now.
      - baseline window: the ``baseline_days`` preceding the recent window.
      - both windows need enough points (>= min_*_points) to be meaningful;
        otherwise ``status`` = 'insufficient_data'.

    Direction handling: for 'ms', 's', '%', higher is worse; for 'FPS'/'tps',
    lower is worse. ``pct_change`` is signed so that positive means "worse"
    regardless of unit, making sort-by-worst trivial.
    """
    start = time.time()
    purpose_like = f"%{purpose_filter}%" if purpose_filter else None
    sql = """
    WITH base AS (
        SELECT machine, model, precision, in_token, out_token, exec_mode, unit,
               ts, value
        FROM perf_flat
        WHERE machine = ?
          AND ts >= current_date - ((? + ?) || ' DAY')::INTERVAL
          AND (? IS NULL OR COALESCE(purpose, '') ILIKE ?)
          AND value > 0
    ),
    tagged AS (
        SELECT *,
            CASE
                WHEN ts >= current_date - (? || ' DAY')::INTERVAL THEN 'recent'
                ELSE 'baseline'
            END AS window_tag
        FROM base
    ),
    window_medians AS (
        SELECT machine, model, precision, in_token, out_token, exec_mode,
               median(value) FILTER (WHERE window_tag = 'recent')   AS recent_median,
               median(value) FILTER (WHERE window_tag = 'baseline') AS baseline_median
        FROM tagged
        GROUP BY machine, model, precision, in_token, out_token, exec_mode
    ),
    window_mads AS (
        -- DuckDB can't mix GROUP BY with a named window that references
        -- an aggregate. Compute MAD in a separate pass, per series.
        SELECT t.machine, t.model, t.precision, t.in_token, t.out_token,
               t.exec_mode,
               median(abs(t.value - wm.recent_median))
                   FILTER (WHERE t.window_tag = 'recent') AS recent_mad,
               median(abs(t.value - wm.baseline_median))
                   FILTER (WHERE t.window_tag = 'baseline') AS baseline_mad
        FROM tagged t
        JOIN window_medians wm
          USING (machine, model, precision, in_token, out_token, exec_mode)
        GROUP BY t.machine, t.model, t.precision, t.in_token, t.out_token, t.exec_mode
    ),
    agg AS (
        SELECT t.machine, t.model, t.precision, t.in_token, t.out_token,
               t.exec_mode, t.unit,
               any_value(wm.recent_median) AS recent_median,
               any_value(wm.baseline_median) AS baseline_median,
               count(*)        FILTER (WHERE window_tag = 'recent')   AS recent_n,
               count(*)        FILTER (WHERE window_tag = 'baseline') AS baseline_n,
               any_value(wmad.recent_mad) AS recent_mad,
               any_value(wmad.baseline_mad) AS baseline_mad
        FROM tagged t
        LEFT JOIN window_medians wm
          USING (machine, model, precision, in_token, out_token, exec_mode)
        LEFT JOIN window_mads wmad
          USING (machine, model, precision, in_token, out_token, exec_mode)
        GROUP BY t.machine, t.model, t.precision, t.in_token, t.out_token,
                 t.exec_mode, t.unit
    )
    SELECT
        machine, model, precision, in_token, out_token, exec_mode, unit,
        recent_median, baseline_median, recent_n, baseline_n,
        recent_mad, baseline_mad,
        CASE WHEN unit IN ('ms', 's', '%') THEN 'lower_is_better'
             ELSE 'higher_is_better' END AS direction,
        CASE
            WHEN baseline_median IS NULL OR recent_median IS NULL
              OR baseline_median = 0 THEN NULL
            -- pct_change is signed positive = worse for both directions,
            -- so the UI can just sort DESC to surface regressions.
            WHEN unit IN ('ms', 's', '%')
              THEN (recent_median - baseline_median) / baseline_median
            ELSE -((recent_median - baseline_median) / baseline_median)
        END AS worsening_pct,
        CASE
            WHEN recent_median IS NULL OR recent_median = 0 THEN NULL
            ELSE recent_mad / recent_median
        END AS recent_cv
    FROM agg
    """
    params = [machine, recent_days, baseline_days,
              purpose_filter, purpose_like, recent_days]
    with _read_only(db_path) as con:
        df = con.execute(sql, params).fetchdf()

    if df.empty:
        return df

    # Status derived in Python so we can thread sidebar thresholds through
    # without re-running SQL (viewer caches on threshold tuple).
    # Vectorized status check
    insufficient = (df["recent_median"].isna() | df["baseline_median"].isna() |
                    (df["recent_n"] < min_recent_points) |
                    (df["baseline_n"] < min_baseline_points))
    df["status"] = "ok"
    df.loc[insufficient, "status"] = "insufficient_data"

    # Vectorized direction sign (1 for lower_is_better units, -1 otherwise)
    direction_sign = df["unit"].apply(lambda unit: 1 if unit in ("ms", "s", "%") else -1)
    sigma = 1.4826 * df["baseline_mad"]
    df["worsening_z"] = (direction_sign
                          * (df["recent_median"] - df["baseline_median"])
                          / sigma.where(sigma > 0))
    result = df.sort_values("worsening_pct", ascending=False,
                            na_position="last").reset_index(drop=True)
    elapsed = time.time() - start
    log.debug(f"trend_regressions(machine={machine}, recent_days={recent_days}) returned {len(result)} rows in {elapsed:.2f}s")
    return result


# ---------------------------------------------------------------------------
# Geomean trend: one number per run across a stable model set
# ---------------------------------------------------------------------------

def geomean_trend(db_path: Path, machine: str,
                  *, exec_mode: str = "2nd",
                  in_bucket: str | None = None,
                  out_bucket: str | None = None,
                  exclude_models: tuple[str, ...] = (),
                  days: int = 90,
                  purpose_filter: str | None = None) -> pd.DataFrame:
    """Geomean of ``value`` per run for a bucket of perf rows.

    ``exec_mode`` filters rows ('1st', '2nd', 'pipeline', ...).
    Bucket filters let the UI separate short-prompt from long-prompt trends,
    which is the usual way to read LLM 2nd-token latency.
    """
    filters = ["f.machine = ?", "f.exec_mode = ?"]
    params: list = [machine, exec_mode]
    if purpose_filter:
        filters.append("COALESCE(f.purpose, '') ILIKE ?")
        params.append(f"%{purpose_filter}%")
    if in_bucket:
        filters.append("f.in_bucket = ?")
        params.append(in_bucket)
    if out_bucket:
        filters.append("f.out_bucket = ?")
        params.append(out_bucket)
    if exclude_models:
        filters.append("f.model NOT IN (" + ",".join(["?"] * len(exclude_models)) + ")")
        params.extend(exclude_models)
    filters.append("f.ts >= current_date - (? || ' DAY')::INTERVAL")
    params.append(str(days))

    where = " AND ".join(filters)
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT
                f.run_id,
                f.ts,
                f.date,
                f.ov_version,
                f.ov_build,
                f.ww,
                exp(avg(ln(f.value))) AS geomean,
                count(*) AS n_samples
            FROM perf_flat f
            WHERE {where}
              AND f.value > 0
            GROUP BY f.run_id, f.ts, f.date, f.ov_version, f.ov_build, f.ww
            ORDER BY f.ts
        """, params).fetchdf()


# ---------------------------------------------------------------------------
# Functional issue history
# ---------------------------------------------------------------------------

def fetch_functional_history(
    db_path: Path,
    machine: str | None = None,
    days: int = 30,
) -> pd.DataFrame:
    """Return functional_issues joined with run metadata, newest first.

    Each row represents one failed/errored test in a run.
    Returns an empty DataFrame when the table does not exist (pre-migration DB).
    """
    with _read_only(db_path) as con:
        if "functional_issues" not in _tables_for_db(db_path):
            return pd.DataFrame()

        machine_filter = "AND r.machine = ?" if machine else ""
        params: list[object] = [str(days)]
        if machine:
            params.append(machine)

        return con.execute(f"""
            SELECT
                fi.run_id,
                r.machine,
                strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
                r.ts,
                r.ov_version,
                fi.nodeid,
                fi.outcome,
                CAST(NULL AS TEXT) AS category,
                fi.message
            FROM functional_issues fi
            JOIN runs r ON r.run_id = fi.run_id
            WHERE r.ts >= current_date - (? || ' DAY')::INTERVAL
              {machine_filter}
            ORDER BY r.ts DESC, fi.nodeid
        """, params).fetchdf()


def fetch_functional_summary(
    db_path: Path,
    machine: str | None = None,
    days: int = 30,
) -> pd.DataFrame:
    """Return per-run functional counts joined with analysis_results.

    Useful for building the functional health history chart.
    Returns an empty DataFrame when analysis tables do not exist.
    """
    with _read_only(db_path) as con:
        if "analysis_results" not in _tables_for_db(db_path):
            return pd.DataFrame()

        machine_filter = "AND r.machine = ?" if machine else ""
        params: list[object] = [str(days)]
        if machine:
            params.append(machine)

        return con.execute(f"""
            SELECT
                r.run_id,
                r.machine,
                strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
                r.ts,
                r.ov_version,
                ar.overall_status,
                ar.functional_fail_count,
                ar.functional_fail_count AS functional_issue_count,
                ar.regressed_count,
                ar.compared_count,
                ar.improved_count,
                ar.same_count
            FROM analysis_results ar
            JOIN runs r ON r.run_id = ar.run_id
            WHERE r.ts >= current_date - (? || ' DAY')::INTERVAL
              {machine_filter}
            ORDER BY r.ts
        """, params).fetchdf()


# ---------------------------------------------------------------------------
# Run-to-run comparison
# ---------------------------------------------------------------------------

def fetch_run_comparison(
    db_path: Path,
    run_id_a: str,
    run_id_b: str,
) -> pd.DataFrame:
    """Compare two runs at the series level using analysis_comparisons.

    Returns rows for run_id_a enriched with matching rows from run_id_b.
    Falls back to a direct perf join when analysis_comparisons lacks one run.
    """
    with _read_only(db_path) as con:
        if "analysis_comparisons" in _tables_for_db(db_path):
            # Try to use pre-computed comparisons (run_a is current, run_b is baseline).
            df = con.execute("""
                SELECT
                    ac.model,
                    ac.precision,
                    ac.in_token,
                    ac.out_token,
                    ac.exec_mode,
                    ac.unit,
                    ac.current_value  AS value_a,
                    ac.baseline_value AS value_b,
                    ac.improvement_pct,
                    ac.verdict
                FROM analysis_comparisons ac
                WHERE ac.run_id          = ?
                  AND ac.baseline_run_id = ?
                ORDER BY model, precision, in_token, out_token, exec_mode
            """, [run_id_a, run_id_b]).fetchdf()
            if not df.empty:
                return _fill_missing_verdicts(df)

        # Fallback: direct perf join on series key.
        df = con.execute("""
            SELECT
                COALESCE(a.model,     b.model)     AS model,
                COALESCE(a.precision, b.precision) AS precision,
                COALESCE(a.in_token,  b.in_token)  AS in_token,
                COALESCE(a.out_token, b.out_token) AS out_token,
                COALESCE(a.exec_mode, b.exec_mode) AS exec_mode,
                COALESCE(a.unit,      b.unit)       AS unit,
                a.unit AS current_unit,
                b.unit AS baseline_unit,
                a.value AS value_a,
                b.value AS value_b
            FROM perf a
            FULL OUTER JOIN perf b
              ON a.model     = b.model
             AND a.precision = b.precision
             AND a.in_token  = b.in_token
             AND a.out_token = b.out_token
             AND a.exec_mode = b.exec_mode
             AND a.run_id    = ?
             AND b.run_id    = ?
            WHERE a.run_id = ? OR b.run_id = ?
            ORDER BY model, precision, in_token, out_token, exec_mode
        """, [run_id_a, run_id_b, run_id_a, run_id_b]).fetchdf()
        return _apply_fallback_metrics(df)


def fetch_analysis_overview(db_path: Path, run_id: str) -> pd.DataFrame:
    """Return analysis summary row for one run, enriched with baseline metadata.

    Returns empty DataFrame when analysis tables are unavailable or the run
    has not been analyzed yet.
    """
    with _read_only(db_path) as con:
        if "analysis_results" not in _tables_for_db(db_path):
            return pd.DataFrame()

        return con.execute("""
            SELECT
                ar.run_id,
                ar.overall_status,
                ar.compared_count,
                ar.improved_count,
                ar.same_count,
                ar.regressed_count,
                ar.functional_fail_count,
                ar.functional_fail_count AS functional_issue_count,
                ar.baseline_run_id,
                strftime(rb.ts, '%Y%m%d_%H%M') AS baseline_stamp,
                rb.ov_version AS baseline_ov_version,
                rs.source_path AS run_source_path
            FROM analysis_results ar
            LEFT JOIN runs rb ON rb.run_id = ar.baseline_run_id
            LEFT JOIN runs rs ON rs.run_id = ar.run_id
            WHERE ar.run_id = ?
            LIMIT 1
        """, [run_id]).fetchdf()


# ---------------------------------------------------------------------------
# Count-based trend analysis
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER_UNITS = {"ms", "s", "%"}


def _direction_sign(unit: object) -> int:
    """+1 when a rising value is worse, -1 when a rising value is better."""
    return 1 if unit in _LOWER_IS_BETTER_UNITS else -1


def series_trend(db_path: Path, machine: str, *, recent_runs_n: int = 3,
                 history_runs_n: int = 10,
                 run_kinds: Sequence[str] | None = DEFAULT_RUN_KINDS,
                 models: Sequence[str] | None = None,
                 exec_modes: Sequence[str] | None = None,
                 include_short_run: bool = True,
                 min_history_points: int = 3) -> pd.DataFrame:
    """Per-series comparison of the newest runs against the runs before them.

    Both windows are counted in *runs*, not days. The recent window is the
    last ``recent_runs_n`` runs; the history window is the ``history_runs_n``
    runs immediately preceding it. Medians and MAD make the comparison robust
    to the single-point outliers that iGPU rigs produce routinely.
    """
    cohort = recent_runs(db_path, machine,
                         limit=recent_runs_n + history_runs_n,
                         run_kinds=run_kinds,
                         include_short_run=include_short_run)
    if cohort.empty:
        return pd.DataFrame()

    run_ids = cohort["run_id"].tolist()
    recent_ids = set(run_ids[-recent_runs_n:])
    history_ids = set(run_ids[:-recent_runs_n])
    if not history_ids:
        return pd.DataFrame()

    perf = perf_for_runs(db_path, run_ids, models=models, exec_modes=exec_modes)
    if perf.empty:
        return pd.DataFrame()

    perf["window"] = perf["run_id"].map(
        lambda r: "recent" if r in recent_ids else "history"
    )
    keys = ["model", "precision", "in_token", "out_token", "exec_mode", "unit"]

    rows = []
    for key, group in perf.groupby(keys, dropna=False):
        recent = group.loc[group["window"] == "recent", "value"]
        history = group.loc[group["window"] == "history", "value"]
        unit = key[-1]
        row = dict(zip(keys, key))
        row["recent_n"] = int(recent.size)
        row["history_n"] = int(history.size)
        row["recent_median"] = float(recent.median()) if recent.size else None
        row["history_median"] = float(history.median()) if history.size else None
        row["latest_value"] = (float(group.sort_values("ts")["value"].iloc[-1])
                               if group.size else None)

        if history.size >= min_history_points and recent.size and history.median():
            mad = float((history - history.median()).abs().median())
            sigma = 1.4826 * mad
            median = float(history.median())
            delta = row["recent_median"] - median
            sign = _direction_sign(unit)
            row["history_mad"] = mad
            row["history_sigma"] = sigma
            row["history_cv"] = mad / median if median else None
            row["worsening_pct"] = sign * delta / median
            row["worsening_z"] = (sign * delta / sigma) if sigma > 0 else None
            row["status"] = "ok"
        else:
            row.update(history_mad=None, history_sigma=None, history_cv=None,
                       worsening_pct=None, worsening_z=None,
                       status="insufficient_data")
        row["direction"] = ("lower_is_better" if _direction_sign(unit) == 1
                            else "higher_is_better")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("worsening_pct", ascending=False,
                          na_position="last").reset_index(drop=True)


def series_history_for_runs(db_path: Path, machine: str, *,
                            model: str, precision: str,
                            in_token: int, out_token: int, exec_mode: str,
                            runs_n: int = 20,
                            run_kinds: Sequence[str] | None = DEFAULT_RUN_KINDS,
                            include_short_run: bool = True) -> pd.DataFrame:
    """Value history for one series over the newest ``runs_n`` runs."""
    cohort = recent_runs(db_path, machine, limit=runs_n, run_kinds=run_kinds,
                         include_short_run=include_short_run)
    if cohort.empty:
        return pd.DataFrame()
    run_ids = cohort["run_id"].tolist()
    placeholders = ",".join(["?"] * len(run_ids))
    kind_select = ("f.run_kind" if _has_column(db_path, "perf_flat", "run_kind")
                   else "'daily'")
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT f.run_id, f.ts, f.ww, f.ov_version, {kind_select} AS run_kind,
                   f.value, f.unit
            FROM perf_flat f
            WHERE f.run_id IN ({placeholders})
              AND f.model = ? AND f.precision = ?
              AND f.in_token = ? AND f.out_token = ?
              AND f.exec_mode = ?
            ORDER BY f.ts
        """, [*run_ids, model, precision, in_token, out_token,
              exec_mode]).fetchdf()


def geomean_for_runs(db_path: Path, run_ids: Sequence[str], *,
                     models: Sequence[str] | None = None,
                     exec_modes: Sequence[str] | None = None,
                     common_series_only: bool = True) -> pd.DataFrame:
    """Geomean per run.

    With ``common_series_only`` the geomean is restricted to series present in
    every run of the cohort, so a run that simply measured fewer models does
    not look like a performance change.
    """
    perf = perf_for_runs(db_path, run_ids, models=models, exec_modes=exec_modes)
    if perf.empty:
        return pd.DataFrame()

    keys = ["model", "precision", "in_token", "out_token", "exec_mode"]
    if common_series_only:
        counts = perf.groupby(keys, dropna=False)["run_id"].nunique()
        complete = counts[counts == perf["run_id"].nunique()].index
        if len(complete):
            perf = perf.set_index(keys).loc[complete].reset_index()

    if perf.empty:
        return pd.DataFrame()

    import numpy as np

    grouped = perf.groupby(["run_id", "ts", "ww", "ov_version"], dropna=False)
    out = grouped["value"].agg(
        geomean=lambda v: float(np.exp(np.log(v).mean())),
        n_samples="size",
    ).reset_index()
    return out.sort_values("ts").reset_index(drop=True)


def machine_health_for_runs(db_path: Path,
                            run_ids: Sequence[str]) -> pd.DataFrame:
    """Per-run machine telemetry summary for an explicit cohort."""
    if not run_ids or "machine_monitor_stats" not in _tables_for_db(db_path):
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(run_ids))
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT h.*, r.ts, strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
                   r.ov_version
            FROM run_machine_health h
            JOIN runs r USING (run_id)
            WHERE h.run_id IN ({placeholders})
            ORDER BY r.ts
        """, list(run_ids)).fetchdf()


def machine_stats_for_run(db_path: Path, run_id: str,
                          models: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-test telemetry rows for one run."""
    if "machine_monitor_stats" not in _tables_for_db(db_path):
        return pd.DataFrame()
    filters = ["run_id = ?"]
    params: list = [run_id]
    if models:
        filters.append("model IN ({})".format(",".join(["?"] * len(models))))
        params.extend(models)
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT *
            FROM machine_monitor_stats
            WHERE {' AND '.join(filters)}
            ORDER BY model, nodeid
        """, params).fetchdf()


def monitor_samples_for_run(db_path: Path, run_id: str,
                            label: str | None = None) -> pd.DataFrame:
    """Raw per-sample telemetry for one run, read straight from Parquet.

    The samples live next to the published artefacts rather than in the
    database: they are two orders of magnitude larger than the per-test
    summary in ``machine_monitor_stats`` and are only needed when a specific
    run is being dissected.
    """
    pattern = (db_path.parent / '**' / '*.monitor.parquet').as_posix()
    filters = ['run_id = ?']
    params: list = [run_id]
    if label:
        filters.append('monitor_label = ?')
        params.append(label)
    with _read_only(db_path) as con:
        try:
            return con.execute(f"""
                SELECT *
                FROM read_parquet('{pattern}', union_by_name=true)
                WHERE {' AND '.join(filters)}
                ORDER BY monitor_label, t_monotonic
            """, params).fetchdf()
        except duckdb.Error:
            # No parquet published yet (or only pre-conversion tar.gz runs).
            return pd.DataFrame()


def functional_issues_for_runs(db_path: Path,
                               run_ids: Sequence[str],
                               models: Sequence[str] | None = None) -> pd.DataFrame:
    """Functional issues for an explicit run cohort."""
    if not run_ids or "functional_issues" not in _tables_for_db(db_path):
        return pd.DataFrame()
    filters = ["fi.run_id IN ({})".format(",".join(["?"] * len(run_ids)))]
    params: list = list(run_ids)
    has_model = _has_column(db_path, "functional_issues", "model")
    if models and has_model:
        # Issues recorded before the model column existed have NULL there;
        # keep them visible rather than silently dropping old failures.
        filters.append(
            "(fi.model IN ({}) OR fi.model IS NULL)".format(
                ",".join(["?"] * len(models)))
        )
        params.extend(models)
    model_select = ("fi.model, fi.precision" if has_model
                    else "CAST(NULL AS TEXT) AS model, CAST(NULL AS TEXT) AS precision")
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT fi.run_id, r.machine,
                   strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
                   r.ts, r.ov_version,
                   fi.nodeid, fi.outcome, fi.message,
                   {model_select}
            FROM functional_issues fi
            JOIN runs r USING (run_id)
            WHERE {' AND '.join(filters)}
            ORDER BY r.ts DESC, fi.nodeid
        """, params).fetchdf()


def functional_summary_for_runs(db_path: Path,
                                run_ids: Sequence[str]) -> pd.DataFrame:
    """Per-run functional health for an explicit cohort."""
    if not run_ids or "analysis_results" not in _tables_for_db(db_path):
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(run_ids))
    with _read_only(db_path) as con:
        return con.execute(f"""
            SELECT r.run_id, r.machine,
                   strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
                   r.ts, r.ov_version,
                   ar.overall_status,
                   ar.functional_fail_count AS functional_issue_count,
                   ar.regressed_count, ar.compared_count
            FROM analysis_results ar
            JOIN runs r USING (run_id)
            WHERE r.run_id IN ({placeholders})
            ORDER BY r.ts
        """, list(run_ids)).fetchdf()


# ---------------------------------------------------------------------------
# Trend-aware run comparison
# ---------------------------------------------------------------------------

def compare_runs_with_trend(db_path: Path, machine: str,
                            run_id_a: str, run_id_b: str, *,
                            history_runs_n: int = 10,
                            run_kinds: Sequence[str] | None = DEFAULT_RUN_KINDS,
                            models: Sequence[str] | None = None,
                            include_short_run: bool = True,
                            min_history_points: int = 3) -> pd.DataFrame:
    """Compare two runs with each run's own preceding history as context.

    A raw A-vs-B delta cannot tell a real change from ordinary run-to-run
    scatter. For each series we also compute the median of the runs preceding
    each side, so the UI can say whether A moved outside its own normal range
    or merely landed on a different point of the same distribution.
    """
    base = fetch_run_comparison(db_path, run_id_a, run_id_b)
    if base.empty:
        return base
    if models:
        base = base[base["model"].isin(models)]
        if base.empty:
            return base

    def _history(run_id: str) -> pd.DataFrame:
        with _read_only(db_path) as con:
            row = con.execute("SELECT ts FROM runs WHERE run_id = ?",
                              [run_id]).fetchone()
        if not row:
            return pd.DataFrame()
        # +1 then drop: `recent_runs` is inclusive of the anchor run itself.
        cohort = recent_runs(db_path, machine, limit=history_runs_n + 1,
                             run_kinds=run_kinds,
                             include_short_run=include_short_run,
                             before_ts=row[0])
        if cohort.empty:
            return pd.DataFrame()
        prior_ids = [r for r in cohort["run_id"].tolist() if r != run_id]
        if not prior_ids:
            return pd.DataFrame()
        return perf_for_runs(db_path, prior_ids, models=models)

    keys = ["model", "precision", "in_token", "out_token", "exec_mode"]

    def _stats(history: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if history.empty:
            return pd.DataFrame(columns=[*keys, f"median_{suffix}",
                                         f"sigma_{suffix}", f"n_{suffix}"])
        grouped = history.groupby(keys, dropna=False)["value"]
        out = grouped.agg(**{
            f"median_{suffix}": "median",
            f"n_{suffix}": "size",
        }).reset_index()
        mad = grouped.agg(lambda v: float((v - v.median()).abs().median()))
        out[f"sigma_{suffix}"] = (1.4826 * mad).to_numpy()
        return out

    merged = base.merge(_stats(_history(run_id_a), "a"), on=keys, how="left")
    merged = merged.merge(_stats(_history(run_id_b), "b"), on=keys, how="left")

    sign = merged["unit"].apply(_direction_sign)
    delta_a = merged["value_a"] - merged["median_a"]
    merged["a_vs_history_pct"] = -sign * delta_a / merged["median_a"]
    merged["a_vs_history_z"] = (sign * delta_a / merged["sigma_a"]).where(
        merged["sigma_a"] > 0
    )
    merged["history_cv_a"] = (merged["sigma_a"] / 1.4826) / merged["median_a"]

    enough = merged["n_a"].fillna(0) >= min_history_points
    # A perfectly flat history has sigma 0, which makes the z-score undefined.
    # Any real deviation from such a history is meaningful, so fall back to a
    # relative check instead of silently reporting "within history".
    degenerate = merged["sigma_a"].isna() | (merged["sigma_a"] <= 0)
    outside = (merged["a_vs_history_z"].abs() >= 2.0).fillna(False)
    outside |= degenerate & (merged["a_vs_history_pct"].abs() > 1e-9).fillna(False)
    merged["trend_context"] = "unknown"
    merged.loc[enough & ~outside, "trend_context"] = "within_history"
    merged.loc[enough & outside, "trend_context"] = "outside_history"
    return merged


# Intel GPUs report a throttle reason for short stretches of almost every run,
# so a brief blip says nothing. Only a run that spent most of its time
# throttled is treated as machine-limited.
THROTTLE_SUSPECT_RATIO = 0.50
# A run whose GPU averaged below this fraction of the card's own max clock was
# not running at the speed its earlier runs did.
LOW_CLOCK_RATIO = 0.80


def classify_machine_state(health_row: pd.Series | None,
                           clock_baseline: float | None = None) -> str:
    """Label one run's machine state: stable / fluctuating / throttled / unknown."""
    if health_row is None or health_row.empty:
        return "unknown"

    # Averaged over the run's tests, not the worst single test, so a momentary
    # thermal excursion does not label the whole run.
    throttle = health_row.get("avg_throttle_ratio")
    if pd.notna(throttle) and float(throttle) >= THROTTLE_SUSPECT_RATIO:
        return "throttled"

    ratio = health_row.get("gpu_clock_ratio")
    if pd.notna(ratio) and float(ratio) < LOW_CLOCK_RATIO:
        return "throttled"

    clock = health_row.get("avg_gpu_clock_mhz")
    if (clock_baseline and pd.notna(clock)
            and clock_baseline > 0
            and float(clock) < clock_baseline * 0.95):
        return "fluctuating"

    if pd.isna(health_row.get("avg_gpu_clock_mhz")):
        return "unknown"
    return "stable"


def attribute_regression(verdict: str, machine_state: str) -> str:
    """Combine the perf verdict with machine state into a cause hint.

    Deliberately conservative: a disturbed machine never *erases* a
    regression, it only downgrades confidence that the code caused it.
    """
    if verdict != "regressed":
        return "n/a"
    if machine_state in {"throttled", "fluctuating"}:
        return "likely-machine"
    if machine_state == "stable":
        return "likely-code"
    return "inconclusive"


# Marketing names carry the model in one token: "Intel(R) Arc(TM) Pro B70
# Graphics (dGPU)" -> "B70". Anything that does not match keeps a trimmed
# form rather than an empty cell.
_DEVICE_MODEL_RE = re.compile(
    r"Arc\(TM\)\s+(?:Pro\s+)?([A-Za-z]?\d{2,4}[A-Za-z]?)\b")


def short_device_name(full_name: object) -> str:
    """Reduce a GPU marketing name to its model token."""
    if full_name is None or pd.isna(full_name):
        return ""
    text = str(full_name)
    match = _DEVICE_MODEL_RE.search(text)
    if match:
        return match.group(1)
    # No model token (e.g. "Intel(R) Arc(TM) Graphics (iGPU)"): drop the
    # bracketed suffixes and vendor boilerplate rather than show an empty cell.
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    cleaned = re.sub(r"\b(?:Intel|Graphics|GPU)\b", " ", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


# ---------------------------------------------------------------------------
# Fleet overview: one row per machine, newest run first
# ---------------------------------------------------------------------------

def machines_overview(db_path: Path,
                      machines: Sequence[str] | None = None, *,
                      run_kinds: Sequence[str] | None = DEFAULT_RUN_KINDS,
                      include_short_run: bool = False,
                      history_runs: int = 10,
                      expected_window: int = 30) -> pd.DataFrame:
    """Latest run per machine with its health counters and perf delta.

    The perf delta is a geomean over latency-unit series only: mixing ms with
    FPS in one geomean would make the direction meaningless. It is further
    restricted to series measured in every scoped run of that machine, so a
    run that skipped a model does not masquerade as a performance change.
    """
    def _filters(alias: str) -> tuple[str, list]:
        parts: list[str] = []
        args: list = []
        if run_kinds and _has_column(db_path, "runs", "run_kind"):
            parts.append("COALESCE({}.run_kind, 'daily') IN ({})".format(
                alias, ",".join(["?"] * len(run_kinds))))
            args.extend(run_kinds)
        if not include_short_run:
            parts.append(f"COALESCE({alias}.short_run, FALSE) = FALSE")
        if machines:
            parts.append("{}.machine IN ({})".format(
                alias, ",".join(["?"] * len(machines))))
            args.extend(machines)
        return (" AND ".join(parts), args)

    filter_sql, params = _filters("r")
    where = ("WHERE " + filter_sql) if filter_sql else ""
    sub_sql, sub_params = _filters("r2")
    sub_where = (" AND " + sub_sql) if sub_sql else ""

    tables = _tables_for_db(db_path)
    if "analysis_results" in tables:
        analysis_cols = """,
                ar.overall_status,
                ar.functional_fail_count AS functional_issue_count,
                ar.regressed_count,
                ar.compared_count"""
        analysis_join = "LEFT JOIN analysis_results ar ON ar.run_id = l.run_id"
    else:
        analysis_cols = """,
                CAST(NULL AS TEXT)    AS overall_status,
                CAST(NULL AS INTEGER) AS functional_issue_count,
                CAST(NULL AS INTEGER) AS regressed_count,
                CAST(NULL AS INTEGER) AS compared_count"""
        analysis_join = ""

    if "functional_issues" in tables:
        issue_col = """,
                (SELECT count(*) FROM functional_issues fi
                  WHERE fi.run_id = l.run_id) AS failing_tests"""
    else:
        issue_col = ", CAST(NULL AS INTEGER) AS failing_tests"

    with _read_only(db_path) as con:
        latest = con.execute(f"""
            WITH ranked AS (
                SELECT r.*,
                       ROW_NUMBER() OVER (PARTITION BY r.machine
                                          ORDER BY r.ts DESC) AS rn
                FROM runs r
                {where}
            ),
            l AS (SELECT * FROM ranked WHERE rn = 1)
            SELECT l.machine, l.run_id, l.ts,
                   strftime(l.ts, '%Y%m%d_%H%M') AS stamp,
                   l.ww, l.ov_version, l.purpose, l.report_file,
                   l.total_tests, l.passed_tests, l.failed_tests,
                   l.error_tests, l.skipped_tests, l.skipped_cases,
                   l.duration_sec,
                   (SELECT count(*) FROM perf p WHERE p.run_id = l.run_id)
                       AS success_cases,
                   (SELECT sd.device FROM system_devices sd
                     WHERE sd.run_id = l.run_id
                     ORDER BY sd.device_index LIMIT 1) AS gpu_name,
                   -- A run counts as clean only when pytest reported no
                   -- failure and no error; skips are expected on some rigs.
                   (SELECT strftime(r2.ts, '%Y%m%d_%H%M') FROM runs r2
                     WHERE r2.machine = l.machine{sub_where}
                       AND COALESCE(r2.total_tests, 0) > 0
                       AND COALESCE(r2.failed_tests, 0)
                         + COALESCE(r2.error_tests, 0) = 0
                     ORDER BY r2.ts DESC LIMIT 1) AS last_success_stamp,
                   (SELECT strftime(r2.ts, '%Y%m%d_%H%M') FROM runs r2
                     WHERE r2.machine = l.machine{sub_where}
                       AND COALESCE(r2.failed_tests, 0)
                         + COALESCE(r2.error_tests, 0) > 0
                     ORDER BY r2.ts DESC LIMIT 1) AS last_fail_stamp
                   {analysis_cols}
                   {issue_col}
            FROM l
            {analysis_join}
            ORDER BY l.machine
        """, [*params, *sub_params, *sub_params]).fetchdf()

        geo = con.execute(f"""            WITH ranked AS (
                SELECT r.machine, r.run_id,
                       ROW_NUMBER() OVER (PARTITION BY r.machine
                                          ORDER BY r.ts DESC) AS rn
                FROM runs r
                {where}
            ),
            scoped AS (SELECT * FROM ranked WHERE rn <= ?),
            vals AS (
                SELECT s.machine, s.rn, f.model, f.precision,
                       f.in_token, f.out_token, f.exec_mode, f.value
                FROM scoped s
                JOIN perf_flat f ON f.run_id = s.run_id
                WHERE f.value > 0 AND f.unit IN ('ms', 's')
            ),
            per_run AS (
                SELECT machine, count(DISTINCT rn) AS runs FROM vals
                GROUP BY machine
            ),
            common AS (
                SELECT v.machine, v.model, v.precision, v.in_token,
                       v.out_token, v.exec_mode
                FROM vals v
                JOIN per_run p USING (machine)
                GROUP BY v.machine, v.model, v.precision, v.in_token,
                         v.out_token, v.exec_mode, p.runs
                HAVING count(DISTINCT v.rn) = any_value(p.runs)
            ),
            geo AS (
                SELECT v.machine, v.rn, exp(avg(ln(v.value))) AS geomean
                FROM vals v
                JOIN common c USING (machine, model, precision,
                                     in_token, out_token, exec_mode)
                GROUP BY v.machine, v.rn
            )
            SELECT machine,
                   any_value(geomean) FILTER (WHERE rn = 1)  AS latest_geomean,
                   median(geomean)    FILTER (WHERE rn > 1)  AS history_geomean,
                   count(*)           FILTER (WHERE rn > 1)  AS history_runs
            FROM geo
            GROUP BY machine
        """, [*params, int(history_runs) + 1]).fetchdf()

    if latest.empty:
        return latest

    with _read_only(db_path) as con:
        # A run that broke early produces almost no series, so the machine's
        # own best recent run is the only available yardstick for how many
        # cases it should have produced.
        expected = con.execute(f"""
            WITH ranked AS (
                SELECT r.machine, r.run_id,
                       ROW_NUMBER() OVER (PARTITION BY r.machine
                                          ORDER BY r.ts DESC) AS rn
                FROM runs r
                {where}
            ),
            scoped AS (SELECT * FROM ranked WHERE rn <= ?),
            per_run AS (
                SELECT s.machine, s.run_id, count(*) AS cases
                FROM scoped s JOIN perf p ON p.run_id = s.run_id
                GROUP BY s.machine, s.run_id
            )
            SELECT machine, max(cases) AS expected_cases
            FROM per_run GROUP BY machine
        """, [*params, int(expected_window)]).fetchdf()

    out = latest.merge(geo, on="machine", how="left")
    out = out.merge(expected, on="machine", how="left")
    # Latency is lower-is-better, so a rise is a worsening.
    out["perf_pct"] = ((out["latest_geomean"] - out["history_geomean"])
                       / out["history_geomean"])
    out["age_hours"] = ((pd.Timestamp.now() - pd.to_datetime(out["ts"]))
                        .dt.total_seconds() / 3600.0)
    return out.sort_values("machine").reset_index(drop=True)
