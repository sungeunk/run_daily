#!/usr/bin/env python3
"""Whisper base tps benchmark via optimum-notebook runner."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path
from parsers.whisper_base import parse_output


@dataclass(frozen=True)
class WhisperCase:
    display_name: str
    model_subdir: str

    @property
    def test_id(self) -> str:
        return self.display_name.replace(' ', '_')


CASES: list[WhisperCase] = [
    WhisperCase('whisper-base', 'whisper-base-nonstateful'),
]

# Whole-file skip: the upstream runner hardcodes ``openai/whisper-base`` and
# requires a pre-populated HF cache that current daily rigs don't have. The
# failure is in scripts/whisper/.../run_model.py, not in the daily framework.
pytestmark = pytest.mark.skip(
    reason='whisper_base runner requires HF cache for openai/whisper-base',
)


def _build_cmd(cfg: DailyConfig, case: WhisperCase) -> str:
    model_path = convert_path(f'{cfg.model_dir}/{case.model_subdir}')
    app = convert_path(str(cfg.whisper_base_script))
    return f'python {app} -m {model_path} -d {cfg.device}'


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_whisper_base(case: WhisperCase, daily_config: DailyConfig,
                      run_subprocess, record_metrics):
    cmd = _build_cmd(daily_config, case)
    record_metrics({
        'test_type': 'whisper_base',
        'model': case.display_name,
        'precision': '',
        'cmd': cmd,
        'data': [],
    })

    result = run_subprocess(cmd)
    data = parse_output(result.output)

    record_metrics({
        'test_type': 'whisper_base',
        'model': case.display_name,
        'precision': '',
        'cmd': cmd,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        'data': data,
    })

    assert result.returncode == 0
    assert data, 'no tps line parsed from whisper runner output'
