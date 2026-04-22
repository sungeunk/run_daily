#!/usr/bin/env python3
"""Parser for openvino.genai/tools/llm_bench/benchmark.py stdout.

Ported from scripts/test_cases/test_benchmark.py's parse_output. Returns
plain dicts so the result can be serialised to JSON directly (pytest's
user_properties channel).

Output line formats this parser recognises:

    prompt nums: 2
    [<iter>][p<idx>] Input token size: <in>, Output size: <out>
    [<iter>][p<idx>] First token latency: <1st> ms, other tokens latency: <2nd> ms
    [<iter>][p<idx>] First token latency: <1st> ms
    [warm-up][p<idx>] Generated:<text...>

Multiple iterations may report latency; we keep the fastest (lowest geomean)
to match the old behaviour.
"""

from __future__ import annotations

import re
from statistics import geometric_mean
from typing import TypedDict


class LlmDataItem(TypedDict, total=False):
    in_token: int
    out_token: int
    perf: list[float]
    generated_text: str


_RE_PROMPT_NUMS = re.compile(r'prompt nums: (\d+)')
_RE_TOKEN_SIZE = re.compile(r'\[\w(\d+)\] Input token size: (\d+), Output size: (\d+)')
_RE_LATENCY_FULL = re.compile(
    r'\[\d+\]\[\w(\d+)\] First token latency: (\d+\.\d+) ms, other tokens latency: (\d+\.\d+) ms'
)
_RE_LATENCY_FIRST_ONLY = re.compile(
    r'\[\d+\]\[\w(\d+)\] First token latency: (\d+\.\d+) ms'
)
_RE_GENERATED = re.compile(r'\[warm-up\]\[\w(\d+)\] Generated:([\S ]+)')
_RE_NEXT_SECTION = re.compile(r'\[ (\S+) \] ')


def parse_output(output: str) -> list[LlmDataItem]:
    """Parse llm_bench stdout into a list of per-prompt data items."""
    ret: list[LlmDataItem] = []
    generated_text: str | None = None
    prompt_id = 0

    for line in output.splitlines():
        # Continuing a multi-line generated-text capture until we see the
        # next section header like "[ foo ]".
        if generated_text is not None:
            if _RE_NEXT_SECTION.search(line):
                ret[prompt_id]['generated_text'] = generated_text
                generated_text = None
            else:
                generated_text += line
            continue

        m = _RE_PROMPT_NUMS.search(line)
        if m:
            for _ in range(int(m.group(1))):
                ret.append({})
            continue

        m = _RE_TOKEN_SIZE.search(line)
        if m:
            idx, in_tok, out_tok = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ret[idx]['in_token'] = in_tok
            ret[idx]['out_token'] = out_tok
            continue

        m_full = _RE_LATENCY_FULL.search(line)
        m_first = _RE_LATENCY_FIRST_ONLY.search(line)
        if m_full:
            idx = int(m_full.group(1))
            new_perf = [float(m_full.group(2)), float(m_full.group(3))]
        elif m_first:
            idx = int(m_first.group(1))
            new_perf = [float(m_first.group(2))]
        else:
            new_perf = None

        if new_perf is not None:
            old_perf = ret[idx].get('perf')
            if old_perf is None or geometric_mean(new_perf) < geometric_mean(old_perf):
                ret[idx]['perf'] = new_perf
            continue

        m = _RE_GENERATED.search(line)
        if m:
            prompt_id = int(m.group(1))
            generated_text = m.group(2)
            continue

    return ret
