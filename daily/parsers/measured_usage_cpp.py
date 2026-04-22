#!/usr/bin/env python3
"""Parser for the qwen measured-usage C++ binary output.

Expected output shape (per prompt)::

    First inference took 123.45 ms
    ... generated text spanning multiple lines ...
    Average other token latency: 23.45 ms
    Input num tokens: 32, output num tokens: 256, ...
"""

from __future__ import annotations

import re
from typing import TypedDict


class QwenUsageItem(TypedDict, total=False):
    in_token: int
    out_token: int
    perf: list[float]     # [first_inf_ms, avg_other_ms]
    generated_text: str


_RE_FIRST_INF = re.compile(r'First inference took (\d+\.\d+) ms')
_RE_AVG_OTHER = re.compile(r'Average other token latency: (\d+\.\d+) ms')
_RE_COUNTS = re.compile(r'Input num tokens: (\d+), output num tokens: (\d+),')


def parse_output(output: str) -> list[QwenUsageItem]:
    ret: list[QwenUsageItem] = []
    first_inf = -1.0
    avg_other = -1.0
    in_sentence = False
    sentence: list[str] = []

    for line in output.splitlines():
        m = _RE_FIRST_INF.search(line)
        if m:
            first_inf = float(m.group(1))
            in_sentence = True
            continue

        m = _RE_AVG_OTHER.search(line)
        if m:
            avg_other = float(m.group(1))
            in_sentence = False
            continue

        m = _RE_COUNTS.search(line)
        if m:
            ret.append({
                'in_token': int(m.group(1)),
                'out_token': int(m.group(2)),
                'perf': [first_inf, avg_other],
                'generated_text': ''.join(sentence),
            })
            sentence = []
            continue

        if in_sentence:
            sentence.append(line)

    return ret
