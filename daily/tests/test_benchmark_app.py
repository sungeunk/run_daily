#!/usr/bin/env python3
"""benchmark_app (Resnet50) throughput tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.config import DailyConfig
from common.fs_utils import convert_path
from parsers.benchmark_app import parse_output


@dataclass(frozen=True)
class BenchAppCase:
    model: str
    precision: str
    model_path: str
    batch: int

    @property
    def test_id(self) -> str:
        return f'{self.model}-{self.precision}-b{self.batch}'


CASES: list[BenchAppCase] = [
    BenchAppCase('Resnet50', 'INT8',
                 'models/resnet_v1.5_50/resnet_v1.5_50_i8.xml', batch=1),
    BenchAppCase('Resnet50', 'INT8',
                 'models/resnet_v1.5_50/resnet_v1.5_50_i8.xml', batch=64),
]


def _build_cmd(cfg: DailyConfig, case: BenchAppCase) -> list[str]:
    # benchmark_app is invoked via ``python -c`` because the CLI entry point
    # is not always on PATH in the daily environment. Returning a list avoids
    # shell-quoting headaches around the ``-c`` argument.
    model_abs = convert_path(str(cfg.repo_root / case.model_path))
    return [
        'python', '-c',
        'from openvino.tools.benchmark.main import main; main()',
        '-m', model_abs,
        '-b', str(case.batch),
        '-d', cfg.device,
        '-hint', 'none',
        '-nstreams', '2',
        '-nireq', '4',
        '-t', '10',
    ]


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.test_id)
def test_benchmark_app(case: BenchAppCase, daily_config: DailyConfig,
                       run_subprocess, record_metrics):
    cmd = _build_cmd(daily_config, case)
    cmd_str = ' '.join(cmd)
    record_metrics({
        'test_type': 'benchmark_app',
        'model': case.model,
        'precision': case.precision,
        'batch': case.batch,
        'cmd': cmd_str,
        'data': [],
    })

    result = run_subprocess(cmd)
    data = parse_output(result.output)

    record_metrics({
        'test_type': 'benchmark_app',
        'model': case.model,
        'precision': case.precision,
        'batch': case.batch,
        'cmd': cmd_str,
        'returncode': result.returncode,
        'duration_sec': result.duration_sec,
        'data': data,
    })

    assert result.returncode == 0
    assert data, 'no Throughput line parsed from benchmark_app output'
