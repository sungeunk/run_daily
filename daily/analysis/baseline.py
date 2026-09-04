"""Baseline run selection and last-known-good search.

All queries are read-only against the DuckDB connection supplied by the
caller; no writes happen here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

    from viewer.ingest.record import RunRecord

from .types import AnalysisConfig, BaselineInfo

log = logging.getLogger(__name__)

# Scheduled runs are tagged by the task scheduler with a purpose such as
# "daily_CB timer" or "daily2 timer"; only those are comparable reference runs.
REFERENCE_PURPOSE_LIKE = "%timer%"


def reference_purpose_sql(
    config: AnalysisConfig,
    column: str = "r.purpose",
) -> tuple[str, list]:
    """SQL predicate (and params) selecting runs eligible as reference."""
    values = _explicit_purposes(config)
    if values:
        placeholders = ", ".join("?" for _ in values)
        return f"COALESCE({column}, '') IN ({placeholders})", list(values)
    return f"lower(COALESCE({column}, '')) LIKE ?", [REFERENCE_PURPOSE_LIKE]


def reference_purpose_label(config: AnalysisConfig) -> str:
    """Human-readable form of the reference purpose filter."""
    values = _explicit_purposes(config)
    if values:
        return f"purpose in {', '.join(values)}"
    return f"purpose like '{REFERENCE_PURPOSE_LIKE}'"


def _explicit_purposes(config: AnalysisConfig) -> tuple[str, ...]:
    if not config.baseline_purpose:
        return ()
    return tuple(
        purpose.strip()
        for purpose in config.baseline_purpose.split(",")
        if purpose.strip()
    )


def select_baseline(
    con: "duckdb.DuckDBPyConnection",
    rec: "RunRecord",
    config: AnalysisConfig,
) -> BaselineInfo:
    """Return the most recent timer-scheduled run comparable with *rec*.

    Selection priority:

    1. same machine + same short_run + timer purpose + older timestamp
    2. same machine + timer purpose + older timestamp

    When *config.baseline_green_only* is True the query additionally
    requires ``overall_status = 'green'`` in the ``analysis_results``
    table.  If that table does not yet exist the flag is silently ignored.
    """
    green_join = _green_join(con) if config.baseline_green_only else ""
    label = reference_purpose_label(config)

    # --- priority 1: same short_run + reference purpose ---
    row = _query_baseline(
        con,
        rec=rec,
        config=config,
        green_join=green_join,
        include_short_run=True,
    )
    if row:
        return _make_info(row, f"latest run with same machine, short_run, {label}")

    # --- priority 2: same machine + reference purpose ---
    row = _query_baseline(
        con,
        rec=rec,
        config=config,
        green_join=green_join,
        include_short_run=False,
    )
    if row:
        return _make_info(row, f"latest run with same machine, {label}")

    return BaselineInfo(status="not_found")


# ---------------------------------------------------------------------------
# Last known good (bisect support)
# ---------------------------------------------------------------------------

def find_last_known_good(
    con: "duckdb.DuckDBPyConnection",
    rec: "RunRecord",
    config: AnalysisConfig | None = None,
) -> BaselineInfo:
    """Return the most recent run with overall_status = 'green'.

    Requires the ``analysis_results`` table to exist.  Returns
    ``BaselineInfo(status='not_found')`` if the table is absent or empty.
    """
    try:
        where_sql, params = _candidate_filters(
            rec,
            config=config or AnalysisConfig(),
            include_short_run=True,
            require_overlap=False,
        )
        row = con.execute(
            f"""
            SELECT r.run_id,
                   strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
                   COALESCE(r.ov_version, '') AS ov_version
            FROM runs r
            JOIN analysis_results ar USING (run_id)
            WHERE {where_sql}
              AND ar.overall_status = 'green'
            ORDER BY r.ts DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    except Exception:  # noqa: BLE001 — table may not exist yet
        return BaselineInfo(status="not_found")

    if row:
        return _make_info(row, "last known good (overall_status=green)")
    return BaselineInfo(status="not_found")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _green_join(con: "duckdb.DuckDBPyConnection") -> str:
    """Return a JOIN clause if analysis_results exists, else empty string."""
    try:
        con.execute("SELECT 1 FROM analysis_results LIMIT 0")
        return "JOIN analysis_results ar ON ar.run_id = r.run_id AND ar.overall_status = 'green'"
    except Exception:  # noqa: BLE001
        log.debug("analysis_results table not found; baseline_green_only ignored")
        return ""


def _query_baseline(
    con: "duckdb.DuckDBPyConnection",
    *,
    rec: "RunRecord",
    config: AnalysisConfig,
    green_join: str,
    include_short_run: bool,
) -> tuple | None:
    where_sql, params = _candidate_filters(
        rec,
        config=config,
        include_short_run=include_short_run,
        require_overlap=True,
    )
    sql = f"""
        SELECT r.run_id,
               strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
               COALESCE(r.ov_version, '') AS ov_version
        FROM runs r
        {green_join}
        WHERE {where_sql}
        ORDER BY r.ts DESC
        LIMIT 1
    """
    return con.execute(sql, params).fetchone()


def _candidate_filters(
    rec: "RunRecord",
    *,
    config: AnalysisConfig,
    include_short_run: bool,
    require_overlap: bool,
) -> tuple[str, list]:
    """Build shared candidate-policy predicates for baseline/LKG lookup."""
    clauses = [
        "r.machine = ?",
        "r.run_id <> ?",
        "r.ts < ?",
    ]
    params: list = [rec.machine, rec.run_id, rec.ts]

    if include_short_run:
        clauses.append("r.short_run IS NOT DISTINCT FROM ?")
        params.append(rec.short_run)

    purpose_sql, purpose_params = reference_purpose_sql(config)
    clauses.append(purpose_sql)
    params.extend(purpose_params)

    if require_overlap:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM perf c
                JOIN perf p
                  ON p.model     = c.model
                 AND p.precision = c.precision
                 AND p.in_token  = c.in_token
                 AND p.out_token = c.out_token
                 AND p.exec_mode = c.exec_mode
                WHERE c.run_id = ?
                  AND p.run_id = r.run_id
                LIMIT 1
            )
            """.strip()
        )
        params.append(rec.run_id)

    return "\n          AND ".join(clauses), params


def _make_info(row: tuple, reason: str) -> BaselineInfo:
    run_id, stamp, ov_version = row
    return BaselineInfo(
        status="found",
        run_id=run_id,
        stamp=stamp,
        ov_version=ov_version or None,
        selection_reason=reason,
    )
