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

from dataclasses import dataclass
from pathlib import Path
import platform
import time

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path
from parsers.llm_benchmark import parse_json_report


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
    task: str | None = None
    ptl_only: bool = False

    @property
    def test_id(self) -> str:
        return f'{self.model}-{self.precision}'


CASES: list[BenchmarkCase] = [
    BenchmarkCase('gemma-2-9b-it',          OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('gemma-3-4b-it',          OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('gemma-4-26b-a4b-it',     OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('gemma-4-e2b-it',         OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('gpt-oss-20b',            OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('llama-2-7b-chat-hf',     OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('llama-3.1-8b-instruct',  OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('llama-3.2-1b-instruct',  OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('minicpm4-0.5b',          OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('minicpm4-8b',            OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('mistral-7b-instruct-v0.2', OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('phi-3.5-mini-instruct',  OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('phi-3.5-vision-instruct', OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('phi-4-mini-instruct',    OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('phi-4-multimodal-instruct', OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('qwen3-8b',               OV_FP16_4BIT_DEFAULT),
    BenchmarkCase('qwen3-vl-4b-instruct',   OV_FP16_4BIT_DEFAULT, prompt_type=PROMPT_TYPE_MULTIMODAL),
    BenchmarkCase('qwen3.5-9b',             OV_FP16_4BIT_DEFAULT, task='visual_text_gen'),
    BenchmarkCase('qwen3.6-35b-a3b',        OV_FP16_4BIT_DEFAULT, task='visual_text_gen', ptl_only=True),
]


def _is_ptl_machine() -> bool:
    return 'PTL' in platform.node().upper()


def _build_cmd(cfg: DailyConfig, case: BenchmarkCase, json_report_path: Path) -> str:
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
    if case.task:
        parts.append(f'-t {case.task}')
    parts.append('--disable_prompt_permutation')
    parts.append(f'--load_config {convert_path(str(cfg.wa_config_path))}')
    parts.append(f'-pf {prompt_path}')
    parts.append(f'-rj {convert_path(str(json_report_path))}')

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# The parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_llm_benchmark(case: BenchmarkCase, daily_config: DailyConfig,
                       run_subprocess, record_metrics):
    if case.ptl_only and not _is_ptl_machine():
        pytest.skip(f'{case.model} runs only on PTL machines')

    # Generate JSON report filename with timestamp to avoid overwrites
    output_dir = Path(daily_config.output_dir)
    timestamp = int(time.time())
    json_report_path = output_dir / f'llm_bench_{case.test_id}_{timestamp}.json'
    
    cmd = _build_cmd(daily_config, case, json_report_path)

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
    
    # Verify benchmark completed successfully
    assert result.returncode == 0, (
        f'llm_bench exited with {result.returncode}; see raw log for details'
    )
    assert json_report_path.exists(), (
        f'JSON report not generated at {json_report_path}'
    )
    
    # Parse JSON report instead of console output
    data = parse_json_report(json_report_path)

    record_metrics({
        'test_type': 'llm_benchmark',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        'data': data,
    })

    assert data, 'JSON parser returned no data rows'
