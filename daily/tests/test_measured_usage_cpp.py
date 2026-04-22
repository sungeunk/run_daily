#!/usr/bin/env python3
"""Qwen measured-usage C++ binary tests — tracks peak CPU/memory."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path, is_windows
from common.profiling import sizeof_fmt
from parsers.measured_usage_cpp import parse_output


INT8 = 'INT8'


@dataclass(frozen=True)
class QwenUsageCase:
    select_inputs: int

    @property
    def test_id(self) -> str:
        return f'qwen_usage-INT8-select{self.select_inputs}'


CASES: list[QwenUsageCase] = [QwenUsageCase(i) for i in range(8)]

# Whole-file skip: this suite depends on a locally-built ``bin/qwen/main``
# C++ binary that is not present on current daily rigs. Remove this marker
# once the binary is built or vendored in.
pytestmark = pytest.mark.skip(
    reason='qwen_usage binary not available on this host (bin/qwen/main)',
)


def _build_cmd(cfg: DailyConfig, case: QwenUsageCase) -> str:
    exe = 'main.exe' if is_windows() else 'main'
    app_path = convert_path(f'{cfg.bin_dir}/qwen/{exe}')
    model_path = convert_path(
        f'{cfg.model_dir}/ww52-qwen-bkm-stateful/modified_openvino_model.xml'
    )
    tokenizer_path = convert_path(
        f'{cfg.model_dir}/ww52-qwen-bkm-stateful/qwen.tiktoken'
    )
    out_token_len = 256
    return (
        f'{app_path} -m {model_path} -t {tokenizer_path}'
        f' -d {cfg.device} -l en --stateful -mcl {out_token_len} -f'
        f' --select_inputs {case.select_inputs}'
    )


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_qwen_usage(case: QwenUsageCase, daily_config: DailyConfig,
                    run_subprocess, record_metrics, hw_tracker):
    cmd = _build_cmd(daily_config, case)
    record_metrics({
        'test_type': 'qwen_usage',
        'model': 'qwen_usage',
        'precision': INT8,
        'select_inputs': case.select_inputs,
        'cmd': cmd,
        'data': [],
    })

    tracker = hw_tracker()
    tracker.start()
    result = run_subprocess(cmd)
    stats = tracker.stop()

    data = parse_output(result.output)

    record_metrics({
        'test_type': 'qwen_usage',
        'model': 'qwen_usage',
        'precision': INT8,
        'select_inputs': case.select_inputs,
        'cmd': cmd,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        'peak_cpu_percent': stats.cpu_usage_mean_percent,
        'peak_mem_bytes': stats.mem_delta_bytes,
        'peak_mem_size': sizeof_fmt(stats.mem_delta_bytes),
        'peak_mem_percent': stats.mem_delta_percent,
        'data': data,
    })

    assert result.returncode == 0
    assert data, 'no inference stats parsed from qwen binary output'
