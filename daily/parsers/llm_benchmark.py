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

Multiple iterations may report latency; warm-up is ignored and the fastest
measured iteration is reported so a single slow iteration does not distort
the result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import geometric_mean
from typing import TypedDict


class LlmDataItem(TypedDict, total=False):
    prompt_idx: int
    in_token: int
    out_token: int
    perf: list[float]
    generated_text: str
    # The share of `perf` counted in llm_bench's token_infer_durations. The remainder is
    # neither host-only nor device-only; see mm_embeddings_time for its largest known part.
    infer_perf: list[float]
    # Part of the perf/infer_perf gap on multimodal models; absent for text-only models.
    mm_embeddings_time: float
    # UTC ISO-8601, JSON report only; absent on llm_bench builds that predate them.
    start: str
    end: str
    token_timestamps: dict[str, str]


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


# The daily metric is end-to-end 1st/2nd token latency. `first_infer_latency` /
# `second_infer_avg_latency` only cover what llm_bench counts in token_infer_durations
# and understate TTFT by up to ~16x on VLMs, so they are reported separately as
# `infer_perf` rather than as `perf`.
PERF_FIELDS = ('first_latency', 'second_avg_latency')
INFER_PERF_FIELDS = ('first_infer_latency', 'second_infer_avg_latency')


def _row_values(row: dict, fields: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, (int, float)) and value >= 0:
            values.append(float(value))
    return values


def _row_perf(row: dict) -> list[float]:
    return _row_values(row, PERF_FIELDS)


def parse_json_report(report_json_path: Path | str) -> list[LlmDataItem]:
    """Parse benchmark.py JSON report into a list of per-prompt data items.

    Warm-up iteration 0 is excluded. Among the remaining iterations the
    fastest one per ``prompt_idx`` is reported, so an occasional slow
    iteration (thermal/scheduling noise) does not distort the daily numbers.
    Speed is ranked by the geometric mean of the perf values, matching the
    stdout parser contract.
    """
    with open(report_json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    perfdata = report.get('perfdata', {})
    results = perfdata.get('results', [])

    prompt_rows: dict[int, list[dict]] = {}
    for row in results:
        if row.get('iteration', -1) < 1:
            continue
        prompt_idx = row.get('prompt_idx', 0)
        if not isinstance(prompt_idx, int):
            prompt_idx = 0
        prompt_rows.setdefault(prompt_idx, []).append(row)

    if not prompt_rows:
        return []

    parsed: list[LlmDataItem] = []
    for prompt_idx in sorted(prompt_rows):
        candidates = [(row, _row_perf(row)) for row in prompt_rows[prompt_idx]]
        candidates = [(row, perf) for row, perf in candidates if perf]
        if not candidates:
            continue

        best_row, best_perf = min(candidates, key=lambda item: geometric_mean(item[1]))

        item: LlmDataItem = {
            'prompt_idx': prompt_idx,
            'in_token': best_row.get('input_size', 0),
            'out_token': best_row.get('infer_count', best_row.get('output_size', 0)),
            'perf': best_perf,
        }
        infer_perf = _row_values(best_row, INFER_PERF_FIELDS)
        if infer_perf:
            item['infer_perf'] = infer_perf
        mm_time = best_row.get('mm_embeddings_preparation_time')
        if isinstance(mm_time, (int, float)) and mm_time > 0:
            item['mm_embeddings_time'] = float(mm_time)
        if best_row.get('start'):
            item['start'] = best_row['start']
        if best_row.get('end'):
            item['end'] = best_row['end']
        if best_row.get('token_timestamps'):
            item['token_timestamps'] = best_row['token_timestamps']
        parsed.append(item)

    return parsed
