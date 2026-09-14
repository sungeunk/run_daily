"""Everything that changes the benchmark DB.

Ingest is the bulk of it and lives in :mod:`data.ingest`; this module is the
entry point, plus the handful of mutations that are not ingest -- currently
the viewer's manual run exclusions.

Those exclusions used to sit in ``queries.py``, a module whose every other
function opened a read-only connection. A reader that quietly writes is how
a layer stops meaning anything, so the read and write halves are separated
here even though the exclusion table is small.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .ingest.writer import (connect, ensure_schema, load_display_profile,
                            upsert_run)

__all__ = ["connect", "ensure_schema", "upsert_run", "load_display_profile",
           "add_exclusion", "remove_exclusion"]


# ---------------------------------------------------------------------------
# Manual run exclusions (viewer's Exclusions tab)
#
# A short-lived read-write connection: the ingest writer is the only other
# thing that opens the file for writing, and it never runs concurrently with
# a user clicking a button in the UI.
# ---------------------------------------------------------------------------

_RUN_EXCLUSIONS_DDL = """
    CREATE TABLE IF NOT EXISTS run_exclusions (
        run_id      TEXT PRIMARY KEY,
        machine     TEXT NOT NULL,
        stamp       TEXT NOT NULL,
        reason      TEXT,
        excluded_at TIMESTAMP DEFAULT now()
    )
"""


def add_exclusion(db_path: Path, run_id: str, machine: str, stamp: str,
                  reason: str) -> None:
    """Hide ``run_id`` from every cohort-based analysis until restored.

    ``reason`` is mandatory. Exclusions feed the rolling baseline
    (``perf_stats``) and the dashboard's latest-run pick, so an exclusion
    without a recorded reason leaves a silently shifted baseline that nobody
    can audit later. Enforced here rather than as a NOT NULL column because
    the DDL is ``IF NOT EXISTS`` and would not migrate existing rows.
    """
    reason = (reason or "").strip()
    if not reason:
        raise ValueError(
            f"a reason is required to exclude run {run_id!r} "
            "(it is the only audit trail for a shifted baseline)"
        )
    with duckdb.connect(str(db_path), read_only=False) as con:
        con.execute(_RUN_EXCLUSIONS_DDL)
        con.execute("""
            INSERT INTO run_exclusions (run_id, machine, stamp, reason)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                reason = excluded.reason, excluded_at = now()
        """, [run_id, machine, stamp, reason])


def remove_exclusion(db_path: Path, run_id: str) -> None:
    """Restore a previously excluded run."""
    with duckdb.connect(str(db_path), read_only=False) as con:
        con.execute(_RUN_EXCLUSIONS_DDL)
        con.execute("DELETE FROM run_exclusions WHERE run_id = ?", [run_id])
