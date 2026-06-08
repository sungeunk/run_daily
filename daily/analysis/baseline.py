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
REFERENCE_PURPOSES = ("daily_CB timer", "daily2 timer")


def _reference_purposes(config: AnalysisConfig, current_purpose: str | None) -> tuple[str, ...]:
    """Return the purpose values that count as the daily reference set."""
    if config.baseline_purpose:
        values = tuple(
            purpose.strip()
            for purpose in config.baseline_purpose.split(",")
            if purpose.strip()
        )
        if values:
            return values
    if current_purpose == "daily2 timer":
        return REFERENCE_PURPOSES
    if current_purpose:
        return (current_purpose,)
    return REFERENCE_PURPOSES


def select_baseline(
    con: "duckdb.DuckDBPyConnection",
    rec: "RunRecord",
    config: AnalysisConfig,
) -> BaselineInfo:
    """Return the most recent comparable baseline run for *rec*.

    Selection priority (matching the logic previously in ``run.py``):

    1. same machine + same short_run + same purpose + older timestamp
    2. same machine + same short_run + older timestamp
    3. same machine + older timestamp

    When *config.baseline_green_only* is True the query additionally
    requires ``overall_status = 'green'`` in the ``analysis_results``
    table.  If that table does not yet exist the flag is silently ignored.
    """
    green_join = _green_join(con) if config.baseline_green_only else ""

    reference_purposes = _reference_purposes(config, rec.purpose)

    # --- priority 1: same short_run + reference purpose ---
    row = _query_baseline(
        con,
        rec=rec,
        green_join=green_join,
        include_short_run=True,
        include_purpose=True,
        purpose_values=reference_purposes,
    )
    if row:
        return _make_info(row, f"same machine, short_run, purpose in {', '.join(reference_purposes)}")

    # --- priority 2: same machine + reference purpose ---
    row = _query_baseline(
        con,
        rec=rec,
        green_join=green_join,
        include_short_run=False,
        include_purpose=True,
        purpose_values=reference_purposes,
    )
    if row:
        return _make_info(row, f"same machine, purpose in {', '.join(reference_purposes)}")

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
            include_short_run=True,
            include_purpose=True,
            purpose_values=_reference_purposes(config or AnalysisConfig(), rec.purpose),
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
    green_join: str,
    include_short_run: bool,
    include_purpose: bool,
    purpose_values: tuple[str, ...] | None = None,
) -> tuple | None:
    where_sql, params = _candidate_filters(
        rec,
        include_short_run=include_short_run,
        include_purpose=include_purpose,
        purpose_values=purpose_values,
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
    include_short_run: bool,
    include_purpose: bool,
    purpose_values: tuple[str, ...] | None,
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
    if include_purpose:
        values = purpose_values or _reference_purposes(AnalysisConfig(), rec.purpose)
        if len(values) == 1:
            clauses.append("COALESCE(r.purpose, '') = COALESCE(?, '')")
            params.append(values[0])
        else:
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"COALESCE(r.purpose, '') IN ({placeholders})")
            params.extend(values)

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
