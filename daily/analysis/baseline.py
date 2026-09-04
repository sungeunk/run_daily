"""Last-known-good search over the local run history.

The reference run itself now comes from the central daily_results server (see
``analysis.remote``); what stays here is the bisect helper, which deliberately
looks at the *local* database because that is where this machine's own green
runs are recorded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

    from viewer.ingest.record import RunRecord

from .types import AnalysisConfig, BaselineInfo

log = logging.getLogger(__name__)


def find_last_known_good(
    con: "duckdb.DuckDBPyConnection",
    rec: "RunRecord",
    config: AnalysisConfig | None = None,
) -> BaselineInfo:
    """Return this machine's most recent run with overall_status = 'green'.

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
              AND r.run_id <> ?
              AND r.ts < ?
              AND r.short_run IS NOT DISTINCT FROM ?
              AND ar.overall_status = 'green'
            ORDER BY r.ts DESC
            LIMIT 1
            """,
            [rec.machine, rec.run_id, rec.ts, rec.short_run],
        ).fetchone()
    except Exception:  # noqa: BLE001 — table may not exist yet
        return BaselineInfo(status="not_found")

    if not row:
        return BaselineInfo(status="not_found")

    run_id, stamp, ov_version = row
    return BaselineInfo(
        status="found",
        run_id=run_id,
        stamp=stamp,
        ov_version=ov_version or None,
        machine=rec.machine,
        selection_reason="last known good (overall_status=green, local history)",
    )
