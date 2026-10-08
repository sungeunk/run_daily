"""Shared interprocess lock for central refresh and manual DB changes."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock


@contextmanager
def database_lock(db_path: Path, *, lock_path: Path | None = None,
                  timeout: float = 60.0) -> Iterator[None]:
    """Serialize snapshot replacement and edits to the live database."""
    path = lock_path or Path(os.environ.get(
        "INGEST_LOCK_FILE", str(Path(db_path).resolve().parent / ".ingest.lock")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path, timeout=timeout, mode=0o600):
        yield