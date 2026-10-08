"""Refresh subprocess and cache-version helpers, independent of Streamlit UI."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import cast

from data.ingest.refresh import RefreshResult

CacheVersion = tuple[str, int, int, int, int]
CACHE_MAX_ENTRIES = 32


def cache_version(db: Path, query_module: Path) -> CacheVersion:
    """Keep DB identity and query version as independently hashed fields."""
    try:
        stat = db.stat()
        inode, modified, size = stat.st_ino, stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        inode = modified = size = 0
    return str(db.resolve()), inode, modified, size, query_module.stat().st_mtime_ns


def run_refresh(script: Path, db: Path, lock: Path,
                on_line: Callable[[str], None]) -> RefreshResult:
    """Stream progress and return the structured outcome after successful exit."""
    environment = {**os.environ, "DAILY_DB_FILE": str(db.resolve()),
                   "INGEST_LOCK_FILE": str(lock.resolve()), "PYTHONUNBUFFERED": "1"}
    environment.pop("INGEST_LOCK_HELD", None)
    tail: deque[str] = deque(maxlen=40)
    payload = None
    with subprocess.Popen([str(script.resolve())], cwd=script.parent,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=environment) as process:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            tail.append(line)
            if line.startswith("REFRESH_RESULT "):
                payload = line.removeprefix("REFRESH_RESULT ")
            else:
                on_line(line)
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"Ingestion failed (exit {exit_code}):\n" + "\n".join(tail))
    return _parse_refresh_result(payload)


def _parse_refresh_result(payload: str | None) -> RefreshResult:
    """Reject incomplete or malformed subprocess results before the UI uses them."""
    if payload is None:
        raise RuntimeError("Ingestion returned no valid REFRESH_RESULT")
    try:
        result = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError("Ingestion returned invalid REFRESH_RESULT JSON") from exc
    if not isinstance(result, dict) or not isinstance(result.get("changed"), bool):
        raise RuntimeError("Ingestion returned no valid REFRESH_RESULT")
    for field in ("added", "skipped"):
        value = result.get(field)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"Invalid REFRESH_RESULT field: {field}")
    timings = result.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("Invalid REFRESH_RESULT field: timings")
    for name, elapsed in timings.items():
        try:
            valid = (isinstance(name, str) and type(elapsed) in (int, float)
                     and math.isfinite(elapsed) and elapsed >= 0)
        except OverflowError:
            valid = False
        if not valid:
            raise RuntimeError(f"Invalid REFRESH_RESULT timing: {name}")
    return cast(RefreshResult, result)