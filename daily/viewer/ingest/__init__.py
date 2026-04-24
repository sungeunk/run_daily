"""Ingest package: turn daily-run artefacts into DuckDB rows.

Two input formats are supported:

* 'new' — ``daily.<stamp>.summary.json`` (the format run.py emits today).
* 'old' — ``daily.<stamp>.<version>.pickle`` plus the companion ``.report``
  text file. Produced by the legacy scripts/report.py pipeline.

Both paths converge on :class:`RunRecord`, which :mod:`writer` upserts into
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
