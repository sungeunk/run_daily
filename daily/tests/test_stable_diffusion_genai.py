#!/usr/bin/env python3
"""Stable Diffusion / whisper benchmarks driven by llm_bench --genai."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path
from parsers.stable_diffusion_genai import parse_output


OV_FP16_4BIT_DEFAULT = 'OV_FP16-4BIT_DEFAULT'
FP16 = 'FP16'
PROMPT_TYPE_MULTIMODAL = 'multimodal'
PROMPT_TYPE_32_1K = '32_1024'


@dataclass(frozen=True)
class SdGenaiCase:
    model: str
    precision: str
    prompt_type: str = PROMPT_TYPE_32_1K

    @property
    def test_id(self) -> str:
        return f'{self.model}-{self.precision}'


CASES: list[SdGenaiCase] = [
    SdGenaiCase('stable-diffusion-v1-5', FP16, PROMPT_TYPE_MULTIMODAL),
    SdGenaiCase('stable-diffusion-v2-1', FP16, PROMPT_TYPE_MULTIMODAL),
    SdGenaiCase('lcm-dreamshaper-v7',    FP16, PROMPT_TYPE_MULTIMODAL),
    SdGenaiCase('flux.1-schnell',        OV_FP16_4BIT_DEFAULT),
    SdGenaiCase('whisper-large-v3',      OV_FP16_4BIT_DEFAULT, PROMPT_TYPE_MULTIMODAL),
]


def _build_cmd(cfg: DailyConfig, case: SdGenaiCase) -> str:
    model_path = convert_path(
        f'{cfg.model_dir}/{cfg.model_date}/{case.model}/pytorch/ov/{case.precision}'
    )
    prompt_path = convert_path(
        f'{cfg.prompts_dir}/{case.prompt_type}/{case.model}.jsonl'
    )
    return (
        f'python {cfg.llm_bench_script}'
        f' -m {model_path}'
        f' -d {cfg.device}'
        f' -mc 1 -n 1 --genai'
        f' --output_dir {cfg.output_dir}'
        f' -pf {prompt_path}'
    )


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_sd_genai(case: SdGenaiCase, daily_config: DailyConfig,
                  run_subprocess, record_metrics):
    cmd = _build_cmd(daily_config, case)
    record_metrics({
        'test_type': 'sd_genai',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'data': [],
    })

    result = run_subprocess(cmd)
    data = parse_output(result.output)

    record_metrics({
        'test_type': 'sd_genai',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        'data': data,
    })

    assert result.returncode == 0
    assert data, 'no pipeline metrics parsed'
