"""Ingest package: turn daily-run artefacts into DuckDB rows.

Input format: ``daily.<stamp>.summary.json`` (the format run.py emits).
It is parsed into :class:`RunRecord`, which :mod:`writer` upserts into
DuckDB.
"""

from .record import RunRecord, DeviceRecord, PerfRow
from .writer import connect, ensure_schema, upsert_run, load_display_profile

__all__ = [
    "RunRecord",
    "DeviceRecord",
    "PerfRow",
    "connect",
    "ensure_schema",
    "upsert_run",
    "load_display_profile",
]
