#!/usr/bin/env python3
"""Parser for SD/whisper runs driven by llm_bench with ``--genai``.

Each sample line pair looks roughly like::

    Batch_size=1
    steps=20
    width=512
    height=512
    guidance_scale=7.5
    Input token size: 77
    Output size: 0
    Infer count: 1
    Generation Time: 3.42s
    [000][P0] start: ...

On seeing the ``[NNN][Pi] start:`` marker we flush the accumulated dict as
one data item. Items use a structured dict rather than a positional list so
downstream consumers don't have to know the index scheme.
"""

from __future__ import annotations

import re
from typing import TypedDict


class SdGenaiItem(TypedDict, total=False):
    generation_time_sec: float
    batch_size: int
    steps: int
    width: int
    height: int
    input_token_size: int
    output_token_size: int
    infer_count: int


_FIELDS: list[tuple[str, re.Pattern, type]] = [
    ('batch_size',         re.compile(r'Batch_size=(\d+)'), int),
    ('steps',              re.compile(r'steps=(\d+)'), int),
    ('width',              re.compile(r'width=(\d+)'), int),
    ('height',             re.compile(r'height=(\d+)'), int),
    ('guidance_scale',     re.compile(r'guidance_scale=(\d+\.\d+)'), float),
    ('input_token_size',   re.compile(r'Input token size: (\d+)'), int),
    ('output_token_size',  re.compile(r'Output size: (\d+)'), int),
    ('infer_count',        re.compile(r'Infer count: (\d+)'), int),
    ('generation_time_sec', re.compile(r'Generation Time: (\d+\.\d+)s'), float),
]

_RE_START = re.compile(r'\[(\d+)\]\[P(\d+)\] start: ')
_RE_WARMUP = re.compile(r'\[warm-up\]')


def parse_output(output: str) -> list[SdGenaiItem]:
    ret: list[SdGenaiItem] = []
    current: SdGenaiItem = {}

    for line in output.splitlines():
        if _RE_WARMUP.search(line):
            continue

        for key, pat, caster in _FIELDS:
            m = pat.search(line)
            if m:
                current[key] = caster(m.group(1))

        if _RE_START.search(line):
            if current:
                ret.append(current)
                current = {}

    return ret
