#!/usr/bin/env python3
"""LLM benchmark tests — one case per (model, precision, extra config).

Run all cases::

    pytest daily/tests/test_llm_benchmark.py

Run a subset by model name::

    pytest daily/tests/test_llm_benchmark.py -k llama

List without running::

    pytest daily/tests/test_llm_benchmark.py --collect-only -q
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path
from parsers.llm_benchmark import parse_output


# ---------------------------------------------------------------------------
# Constants (port of ModelConfig / PROMPT_TYPE_* from the old code)
# ---------------------------------------------------------------------------

OV_FP16_4BIT_DEFAULT = 'OV_FP16-4BIT_DEFAULT'
PROMPT_TYPE_32_1K = '32_1024'
PROMPT_TYPE_MULTIMODAL = 'multimodal'


@dataclass(frozen=True)
class BenchmarkCase:
    model: str
    precision: str
    apply_chat_template: bool = True
    prompt_type: str = PROMPT_TYPE_32_1K

    @property
    def test_id(self) -> str:
        return f'{self.model}-{self.precision}'


CASES: list[BenchmarkCase] = [
    BenchmarkCase('baichuan2-7b-chat',      OV_FP16_4BIT_DEFAULT, apply_chat_template=False),
    BenchmarkCase('chatglm3-6b',            OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('gemma-7b-it',            OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('glm-4-9b-chat-hf',       OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('llama-2-7b-chat-hf',     OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('llama-3.1-8b-instruct',  OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('minicpm-1b-sft',         OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('minicpm-v-2_6',          OV_FP16_4BIT_DEFAULT, prompt_type=PROMPT_TYPE_MULTIMODAL),
    BenchmarkCase('mistral-7b-instruct-v0.2', OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('phi-3-mini-4k-instruct', OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('phi-3.5-mini-instruct',  OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('phi-3.5-vision-instruct', OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('qwen2-7b-instruct',      OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('qwen2.5-7b-instruct',    OV_FP16_4BIT_DEFAULT),
]


def _build_cmd(cfg: DailyConfig, case: BenchmarkCase) -> str:
    model_path = convert_path(
        f'{cfg.model_dir}/{cfg.model_date}/{case.model}/pytorch/ov/{case.precision}'
    )
    prompt_path = convert_path(
        f'{cfg.prompts_dir}/{case.prompt_type}/{case.model}.jsonl'
    )

    parts = [
        f'python {cfg.llm_bench_script}',
        f'-m {model_path}',
        f'-d {cfg.device}',
        '-mc 1',
        f'-ic {cfg.out_token_length}',
        f'-n {cfg.benchmark_iter_num}',
    ]
    if case.apply_chat_template:
        parts.append('--apply_chat_template')
    parts.append('--disable_prompt_permutation')
    parts.append(f'--load_config {convert_path(str(cfg.wa_config_path))}')
    parts.append(f'-pf {prompt_path}')

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# The parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_llm_benchmark(case: BenchmarkCase, daily_config: DailyConfig,
                       run_subprocess, record_metrics):
    cmd = _build_cmd(daily_config, case)

    # Attach metadata even on failure so the report builder can still render
    # a row for this case. We overwrite with the full payload on success.
    record_metrics({
        'test_type': 'llm_benchmark',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'data': [],
    })

    result = run_subprocess(cmd)
    data = parse_output(result.output)

    record_metrics({
        'test_type': 'llm_benchmark',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        'data': data,
    })

    assert result.returncode == 0, (
        f'llm_bench exited with {result.returncode}; see raw log for details'
    )
    assert data, 'parser returned no data rows'
