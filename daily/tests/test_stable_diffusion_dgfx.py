#!/usr/bin/env python3
"""Stable Diffusion v3.0 / XL benchmark via the DGfx_E2E_AI harness.

These tests `cd` into the harness directory because the harness script uses
paths relative to its own working directory.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path
from parsers.stable_diffusion_dgfx import parse_output


FP16 = 'FP16'


@dataclass(frozen=True)
class DgfxCase:
    model: str
    precision: str
    model_name: str   # "v3.0" or "xl" — DGfx harness CLI label
    width: int
    height: int

    @property
    def test_id(self) -> str:
        return f'{self.model}-{self.precision}'


CASES: list[DgfxCase] = [
    DgfxCase('stable-diffusion-v3.0', FP16, 'v3.0', width=512, height=512),
    DgfxCase('stable-diffusion-xl',   FP16, 'xl',   width=768, height=768),
]


def _build_cmd(cfg: DailyConfig, case: DgfxCase) -> str:
    app_path = convert_path('temp/base_sd.py')
    model_root = convert_path(str(cfg.model_dir))
    return (
        f'python {app_path}'
        f' --device {cfg.device}'
        f' --api openvino-nightly'
        f' --model {case.model_name}'
        f' --height {case.height} --width {case.width}'
        f' --num_warm 1 --num_iter 1'
        f' --model_root {model_root}'
    )


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_sd_dgfx(case: DgfxCase, daily_config: DailyConfig,
                 run_subprocess, record_metrics):
    cmd = _build_cmd(daily_config, case)
    work_dir = str(daily_config.dgfx_tests_dir)
    record_metrics({
        'test_type': 'sd_dgfx',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'work_dir': work_dir,
        'data': [],
    })

    result = run_subprocess(cmd, cwd=work_dir)
    data = parse_output(result.output)

    record_metrics({
        'test_type': 'sd_dgfx',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'work_dir': work_dir,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        'data': data,
    })

    assert result.returncode == 0
    assert data, 'no testResult line parsed from DGfx harness output'
