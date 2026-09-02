#!/usr/bin/env python3
"""Artefact layout rules shared by the local output tree and the relay backup.

A run's files live under ``<machine>/<YYYY.MM>`` on both sides, so the same
relative path resolves locally and at ``http://<relay>/daily2/...``. The month
comes from the run stamp rather than the wall clock, so a re-publish lands in
the month the run happened.
"""

from __future__ import annotations

import platform
import re
from pathlib import Path


# Matches a run stamp on its own (``20260902_1222``) or inside an artefact
# name (``daily.20260902_1222.summary.json``).
_STAMP_RE = re.compile(r'(\d{4})(\d{2})\d{2}_\d{3,4}')


def machine_name() -> str:
    return platform.node()


def month_dir_for(name: str) -> str:
    """``YYYY.MM`` bucket for a stamp or artefact name, '' when it has none."""
    m = _STAMP_RE.search(name)
    return f'{m.group(1)}.{m.group(2)}' if m else ''


def relative_dir(name: str = '') -> str:
    """Path under the output/backup root that ``name`` belongs in."""
    month = month_dir_for(name)
    node = machine_name()
    return f'{node}/{month}' if month else node


def machine_dir(root: Path) -> Path:
    """Per-machine root; holds the viewer DB and the monthly run folders."""
    return Path(root) / machine_name()


def run_dir(root: Path, stamp: str) -> Path:
    """Directory holding every artefact of the run identified by ``stamp``."""
    return Path(root) / relative_dir(stamp)
