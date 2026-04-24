"""Shared helpers used by both old and new loaders."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

# Filename stamp: ``daily.<YYYYMMDD>_<HHMM(or HMM)>.*``
FILENAME_DT_RE = re.compile(r"\.(?P<date>\d{8})_(?P<time>\d{3,4})\.", re.ASCII)

# OpenVINO version string: "<ver>-<build>-<sha>" (sha 7-40 hex chars).
OV_VERSION_RE = re.compile(r"(?P<build>\d+)-(?P<sha>[0-9a-fA-F]{7,40})")


def parse_stamp_from_name(name: str) -> datetime | None:
    m = FILENAME_DT_RE.search(name)
    if not m:
        return None
    date_str = m.group("date")
    time_str = m.group("time").zfill(4)
    try:
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
    except ValueError:
        return None


def workweek_of(ts: datetime) -> str:
    iso = ts.isocalendar()
    return f"{iso.year}.WW{iso.week}.{iso.weekday}"


def split_ov_version(ov_version: str | None) -> tuple[str | None, str | None]:
    """Return ``(build, sha)`` from strings like ``2026.2.0-21664-ad5d8e0f99b``."""
    if not ov_version:
        return None, None
    m = OV_VERSION_RE.search(ov_version)
    if not m:
        return None, None
    return m.group("build"), m.group("sha").lower()


def file_hash(path: Path) -> str:
    """Content hash — stable regardless of filesystem metadata or location."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def run_id_of(machine: str, ts: datetime, report_file: str) -> str:
    """Stable run_id derived from machine + timestamp + file stem.

    Including ``ts`` means re-runs stamped minutes apart get distinct ids
    even if the file stem somehow collides. The report_file adds another
    layer so two machines that coincidentally ran at the same minute don't
    collapse.
    """
    key = f"{machine}|{ts.isoformat(timespec='minutes')}|{report_file}"
    return hashlib.sha1(key.encode()).hexdigest()[:20]
