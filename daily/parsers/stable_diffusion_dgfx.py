#!/usr/bin/env python3
"""Parser for the DGfx_E2E_AI harness' ``testResult`` dict output.

The harness prints a single line containing a Python-repr'd dict per run,
prefixed with ``testResult``. We do not eval it — instead we pull the handful
of fields the daily report cares about via regexes against the substring
starting at ``testResult``.
"""

from __future__ import annotations

import re
from typing import TypedDict


class SdDgfxItem(TypedDict, total=False):
    pipeline_sec: float
    batch_size: int
    steps: int
    size: str


_RE_BATCH = re.compile(r"'batch_size': (\d+)")
_RE_SIZE = re.compile(r"'Width': (\d+), 'Height': (\d+)")
_RE_STEPS = re.compile(r"'Inference Steps': (\d+)")
_RE_PIPELINE = re.compile(
    r"'Seconds per image \(s/img\)': np\.float64\((\d+\.\d+)\)"
)


def parse_output(output: str) -> list[SdDgfxItem]:
    ret: list[SdDgfxItem] = []

    for line in output.splitlines():
        idx = line.find('testResult')
        if idx < 0:
            continue
        payload = line[idx:]

        item: SdDgfxItem = {}
        m = _RE_BATCH.search(payload)
        if m:
            item['batch_size'] = int(m.group(1))
        m = _RE_SIZE.search(payload)
        if m:
            item['size'] = f'{int(m.group(1))}x{int(m.group(2))}'
        m = _RE_STEPS.search(payload)
        if m:
            item['steps'] = int(m.group(1))
        m = _RE_PIPELINE.search(payload)
        if m:
            item['pipeline_sec'] = float(m.group(1))
            ret.append(item)

    return ret
