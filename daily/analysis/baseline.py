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
    common_params = [rec.machine, rec.run_id, rec.ts]
    green_join = _green_join(con) if config.baseline_green_only else ""

    # --- priority 1: same purpose ---
    if rec.purpose:
        row = _query_baseline(
            con,
            extra_where="AND COALESCE(r.purpose, '') = ?",
            extra_params=[*common_params, rec.short_run, rec.purpose],
            green_join=green_join,
        )
        if row:
            return _make_info(row, "same machine, short_run, purpose")

    # --- priority 2: same short_run ---
    row = _query_baseline(
        con,
        extra_where="",
        extra_params=[*common_params, rec.short_run],
        green_join=green_join,
    )
    if row:
        return _make_info(row, "same machine, short_run")

    # --- priority 3: same machine only ---
    row = _query_baseline(
        con,
        extra_where="",
        extra_params=common_params,
        green_join=green_join,
        skip_short_run=True,
    )
    if row:
        return _make_info(row, "same machine")

    return BaselineInfo(status="not_found")


# ---------------------------------------------------------------------------
# Last known good (bisect support)
# ---------------------------------------------------------------------------

def find_last_known_good(
    con: "duckdb.DuckDBPyConnection",
    rec: "RunRecord",
) -> BaselineInfo:
    """Return the most recent run with overall_status = 'green'.

    Requires the ``analysis_results`` table to exist.  Returns
    ``BaselineInfo(status='not_found')`` if the table is absent or empty.
    """
    try:
        row = con.execute(
            """
            SELECT r.run_id,
                   strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
                   COALESCE(r.ov_version, '') AS ov_version
            FROM runs r
            JOIN analysis_results ar USING (run_id)
            WHERE r.machine = ?
              AND r.run_id  <> ?
              AND r.ts      < ?
              AND ar.overall_status = 'green'
            ORDER BY r.ts DESC
            LIMIT 1
            """,
            [rec.machine, rec.run_id, rec.ts],
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
    extra_where: str,
    extra_params: list,
    green_join: str,
    skip_short_run: bool = False,
) -> tuple | None:
    short_run_clause = "" if skip_short_run else "AND r.short_run = ?"
    sql = f"""
        SELECT r.run_id,
               strftime(r.ts, '%Y%m%d_%H%M') AS stamp,
               COALESCE(r.ov_version, '') AS ov_version
        FROM runs r
        {green_join}
        WHERE r.machine  = ?
          AND r.run_id  <> ?
          AND r.ts       < ?
          {short_run_clause}
          {extra_where}
        ORDER BY r.ts DESC
        LIMIT 1
    """
    return con.execute(sql, extra_params).fetchone()


def _make_info(row: tuple, reason: str) -> BaselineInfo:
    run_id, stamp, ov_version = row
    return BaselineInfo(
        status="found",
        run_id=run_id,
        stamp=stamp,
        ov_version=ov_version or None,
        selection_reason=reason,
    )
