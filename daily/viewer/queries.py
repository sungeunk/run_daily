"""DuckDB query helpers used by the Streamlit viewer.

All functions accept a path to the DuckDB file and open a read-only
connection per call. Streamlit caches at the dataframe level, so the
per-call connection cost is paid once per cache bucket.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _read_only(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def list_machines(db_path: Path) -> list[str]:
    with _read_only(db_path) as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT machine FROM runs ORDER BY machine").fetchall()]


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


def build_excel_matrix(db_path: Path, run_ids: list[str],
                       profile: str = "default") -> pd.DataFrame:
    """Return a wide dataframe: rows = display_rows order, columns = run stamps.

    Matching rules (in_spec / out_spec):
      'short' / 'long' / '0'  → match perf_with_buckets.(in|out)_bucket
      '*'                     → any
      otherwise               → exact numeric equality
    """
    if not run_ids:
        return pd.DataFrame()

    # Placeholder list for IN clause.
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
            p.value
        FROM display_rows d
        CROSS JOIN rs
        LEFT JOIN perf_with_buckets p
          ON p.run_id = rs.run_id
         AND p.model = d.model
         AND p.precision = d.precision
         AND p.exec_mode = d.exec_mode
         AND (
               d.in_spec = '*'
            OR d.in_spec = p.in_bucket
            OR TRY_CAST(d.in_spec AS INTEGER) = p.in_token
         )
         AND (
               d.out_spec = '*'
            OR d.out_spec = p.out_bucket
            OR TRY_CAST(d.out_spec AS INTEGER) = p.out_token
         )
        WHERE d.profile = ?
    )
    SELECT seq, d_model AS model, d_precision AS precision,
           in_spec, out_spec, d_exec AS exec_mode, label,
           stamp, value
    FROM joined
    ORDER BY seq, stamp
    """
    with _read_only(db_path) as con:
        df = con.execute(sql, [*run_ids, profile]).fetchdf()

    if df.empty:
        return df

    # Build one row per (seq, spec) with a column per run stamp. pivot_table
    # drops NaN index values and also explodes on large cross-products when
    # dropna=False, so we do it manually.
    spec_cols = ["seq", "model", "precision", "in_spec", "out_spec",
                 "exec_mode", "label"]
    specs = (df[spec_cols]
             .drop_duplicates(subset="seq")
             .sort_values("seq")
             .reset_index(drop=True))
    stamps = sorted(df["stamp"].dropna().unique())
    result = specs.copy()
    for st in stamps:
        vals = (df[df["stamp"] == st]
                .set_index("seq")["value"])
        result[st] = result["seq"].map(vals)
    return result.drop(columns=["seq"])


def extra_rows(db_path: Path, run_ids: list[str],
               profile: str = "default") -> pd.DataFrame:
    """Perf rows in the selected runs that match no display_row (QA/sanity)."""
    if not run_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(run_ids))
    sql = f"""
    WITH m AS (
        SELECT DISTINCT model, precision, in_token, out_token, exec_mode,
               CASE WHEN in_token = 0 THEN '0'
                    WHEN in_token < 100 THEN 'short' ELSE 'long' END AS in_bucket,
               CASE WHEN out_token = 0 THEN '0'
                    WHEN out_token < 100 THEN 'short' ELSE 'long' END AS out_bucket
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
        OR d.in_spec = m.in_bucket
        OR TRY_CAST(d.in_spec AS INTEGER) = m.in_token
     )
     AND (
           d.out_spec = '*'
        OR d.out_spec = m.out_bucket
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
                   days: int = 60) -> pd.DataFrame:
    """Time-series of one perf point with rolling baseline stats."""
    with _read_only(db_path) as con:
        return con.execute("""
            SELECT ts, date, ov_version, ov_build, ww,
                   value, unit,
                   win_median, win_mad, win_sigma, win_n,
                   z_score, pct_diff, cv
            FROM perf_stats
            WHERE machine = ?
              AND model = ? AND precision = ?
              AND in_token = ? AND out_token = ?
              AND exec_mode = ?
              AND ts >= current_date - (? || ' DAY')::INTERVAL
            ORDER BY ts
        """, [machine, model, precision, in_token, out_token, exec_mode, days]
                           ).fetchdf()


def regressions_for_run(db_path: Path, run_id: str,
                        z_threshold: float = 3.0,
                        pct_threshold: float = 0.05,
                        noisy_cv: float = 0.10) -> pd.DataFrame:
    """Return regressed perf points for a specific run.

    Direction handling: for 'ms' / 's' units, *up* is worse. For 'FPS' /
    'tps', *down* is worse. The SQL emits a ``direction`` column so the UI
    can colour accordingly.

    A row is flagged as regression when either:
      - |z_score| >= z_threshold AND sign matches a 'worse' move, OR
      - |pct_diff| >= pct_threshold AND sign matches a 'worse' move.
    Both conditions must pass when ``win_mad > 0``; if the series is
    constant (mad=0) we fall back to pct only.
    """
    with _read_only(db_path) as con:
        return con.execute("""
            WITH t AS (
                SELECT *,
                    CASE
                        WHEN unit IN ('ms', 's', '%') THEN 'lower_is_better'
                        ELSE 'higher_is_better'
                    END AS direction
                FROM perf_stats
                WHERE run_id = ?
            )
            SELECT
                machine, model, precision, in_token, out_token, exec_mode,
                value, unit, direction,
                win_median, win_sigma, win_n,
                z_score, pct_diff, cv,
                (cv IS NOT NULL AND cv >= ?) AS is_noisy,
                CASE
                    WHEN direction = 'lower_is_better' AND
                         ((z_score IS NOT NULL AND z_score >= ?)
                           OR (pct_diff IS NOT NULL AND pct_diff >= ?))
                      THEN 'regression'
                    WHEN direction = 'higher_is_better' AND
                         ((z_score IS NOT NULL AND z_score <= -?)
                           OR (pct_diff IS NOT NULL AND pct_diff <= -?))
                      THEN 'regression'
                    WHEN direction = 'lower_is_better' AND
                         z_score IS NOT NULL AND z_score <= -?
                      THEN 'improvement'
                    WHEN direction = 'higher_is_better' AND
                         z_score IS NOT NULL AND z_score >= ?
                      THEN 'improvement'
                    ELSE 'ok'
                END AS status
            FROM t
            ORDER BY
                CASE
                    WHEN direction = 'lower_is_better' THEN -COALESCE(z_score, 0)
                    ELSE COALESCE(z_score, 0)
                END DESC
        """, [run_id, noisy_cv,
              z_threshold, pct_threshold,
              z_threshold, pct_threshold,
              z_threshold, z_threshold]).fetchdf()


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
                      min_recent_points: int = 3,
                      min_baseline_points: int = 5) -> pd.DataFrame:
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
    sql = """
    WITH base AS (
        SELECT machine, model, precision, in_token, out_token, exec_mode, unit,
               ts, value
        FROM perf_flat
        WHERE machine = ?
          AND ts >= current_date - ((? + ?) || ' DAY')::INTERVAL
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
    recent_with_mad AS (
        -- DuckDB can't mix GROUP BY with a named window that references
        -- an aggregate. Compute recent MAD in a separate pass, per series,
        -- using a self-join on the recent window median.
        SELECT t.machine, t.model, t.precision, t.in_token, t.out_token,
               t.exec_mode,
               median(abs(t.value - rm.recent_median)) AS recent_mad
        FROM tagged t
        JOIN (
            SELECT machine, model, precision, in_token, out_token, exec_mode,
                   median(value) AS recent_median
            FROM tagged
            WHERE window_tag = 'recent'
            GROUP BY machine, model, precision, in_token, out_token, exec_mode
        ) rm USING (machine, model, precision, in_token, out_token, exec_mode)
        WHERE t.window_tag = 'recent'
        GROUP BY t.machine, t.model, t.precision, t.in_token, t.out_token, t.exec_mode
    ),
    agg AS (
        SELECT t.machine, t.model, t.precision, t.in_token, t.out_token,
               t.exec_mode, t.unit,
               median(t.value) FILTER (WHERE window_tag = 'recent')   AS recent_median,
               median(t.value) FILTER (WHERE window_tag = 'baseline') AS baseline_median,
               count(*)        FILTER (WHERE window_tag = 'recent')   AS recent_n,
               count(*)        FILTER (WHERE window_tag = 'baseline') AS baseline_n,
               any_value(rm.recent_mad) AS recent_mad
        FROM tagged t
        LEFT JOIN recent_with_mad rm
          USING (machine, model, precision, in_token, out_token, exec_mode)
        GROUP BY t.machine, t.model, t.precision, t.in_token, t.out_token,
                 t.exec_mode, t.unit
    )
    SELECT
        machine, model, precision, in_token, out_token, exec_mode, unit,
        recent_median, baseline_median, recent_n, baseline_n, recent_mad,
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
    params = [machine, recent_days, baseline_days, recent_days]
    with _read_only(db_path) as con:
        df = con.execute(sql, params).fetchdf()

    if df.empty:
        return df

    # Status derived in Python so we can thread sidebar thresholds through
    # without re-running SQL (viewer caches on threshold tuple).
    def _status(row: pd.Series) -> str:
        if (pd.isna(row["recent_median"]) or pd.isna(row["baseline_median"])
                or row["recent_n"] < min_recent_points
                or row["baseline_n"] < min_baseline_points):
            return "insufficient_data"
        return "ok"

    df["status"] = df.apply(_status, axis=1)
    return df.sort_values("worsening_pct", ascending=False,
                          na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Geomean trend: one number per run across a stable model set
# ---------------------------------------------------------------------------

def geomean_trend(db_path: Path, machine: str,
                  *, exec_mode: str = "2nd",
                  in_bucket: str | None = None,
                  out_bucket: str | None = None,
                  exclude_models: tuple[str, ...] = ("qwen_usage",),
                  days: int = 90) -> pd.DataFrame:
    """Geomean of ``value`` per run for a bucket of perf rows.

    ``exec_mode`` filters rows ('1st', '2nd', 'pipeline', ...).
    Bucket filters let the UI separate short-prompt from long-prompt trends,
    which is the usual way to read LLM 2nd-token latency.
    """
    filters = ["f.machine = ?", "f.exec_mode = ?"]
    params: list = [machine, exec_mode]
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
