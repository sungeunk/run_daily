#!/usr/bin/env python3
"""Parser for ``openvino.tools.benchmark.main`` (benchmark_app) stdout.

Extracts a single Throughput (FPS) figure.
"""

from __future__ import annotations

import re
from typing import TypedDict


class BenchmarkAppItem(TypedDict, total=False):
    perf: list[float]


_RE_THROUGHPUT = re.compile(r'Throughput: +(\d+\.\d+) FPS')


def parse_output(output: str) -> list[BenchmarkAppItem]:
    for line in output.splitlines():
        m = _RE_THROUGHPUT.search(line)
        if m:
            return [{'perf': [float(m.group(1))]}]
    return []
