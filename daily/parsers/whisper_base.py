#!/usr/bin/env python3
"""Parser for the whisper base optimum-notebook runner.

Extracts the ``tps`` (tokens-per-second) figure. The runner may report more
than one; we keep all of them.
"""

from __future__ import annotations

import re
from typing import TypedDict


class WhisperItem(TypedDict, total=False):
    perf: list[float]     # [tps]


_RE_TPS = re.compile(r'tps : (\d+\.\d+)')


def parse_output(output: str) -> list[WhisperItem]:
    ret: list[WhisperItem] = []
    for line in output.splitlines():
        m = _RE_TPS.search(line)
        if m:
            ret.append({'perf': [float(m.group(1))]})
    return ret
