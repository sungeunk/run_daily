#!/usr/bin/env python3
"""chat_sample smoke test.

The upstream script has no parseable perf output — we just assert it runs
cleanly and store its raw stdout in the metrics so the report builder can
embed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path


@dataclass(frozen=True)
class ChatSampleCase:
    model: str
    precision: str
    app_path: str

    @property
    def test_id(self) -> str:
        return f'{self.model}-{self.precision}'


CASES: list[ChatSampleCase] = [
    ChatSampleCase(
        'glm-4-9b-chat-hf',
        'OV_FP16-4BIT_DEFAULT',
        'openvino.genai/samples/python/text_generation/chat_sample.py',
    ),
]


def _build_cmd(cfg: DailyConfig, case: ChatSampleCase) -> str:
    app = convert_path(str(cfg.repo_root / case.app_path))
    model = convert_path(
        f'{cfg.model_dir}/{cfg.model_date}/{case.model}/pytorch/ov/{case.precision}'
    )
    return f'python {app} {model} {cfg.device}'


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_chat_sample(case: ChatSampleCase, daily_config: DailyConfig,
                     run_subprocess, record_metrics):
    cmd = _build_cmd(daily_config, case)
    record_metrics({
        'test_type': 'chat_sample',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'output': '',
    })

    result = run_subprocess(cmd)

    record_metrics({
        'test_type': 'chat_sample',
        'model': case.model,
        'precision': case.precision,
        'cmd': cmd,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        # We embed the full stdout so the report can reproduce the old
        # behaviour of printing raw chat output in the report.
        'output': result.output,
    })

    assert result.returncode == 0
