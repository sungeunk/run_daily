#!/usr/bin/env python3
"""Daily suite configuration.

Replaces the Borg-style GlobalConfig with an explicit dataclass that is
constructed once (typically in conftest.py) and then threaded through
fixtures. Keeps values immutable for the duration of a test session.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path

from .fs_utils import convert_path


try:
    from openvino import get_version as _ov_get_version
except ImportError:
    def _ov_get_version() -> str:
        return 'none'


# Repo layout: <repo>/daily/common/config.py -> <repo> is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DailyConfig:
    repo_root: Path
    output_dir: Path
    cache_dir: Path
    model_dir: Path
    model_date: str
    device: str
    timeout_sec: int
    short_run: bool           # replaces --test: shorten tokens/iters
    tee_raw_log: bool
    out_token_length: int
    benchmark_iter_num: int
    ov_version: str
    now: str                  # YYYYMMDD_HHMM timestamp
    raw_log_path: Path
    json_report_path: Path

    @property
    def bin_dir(self) -> Path:
        return self.repo_root / 'bin'

    @property
    def prompts_dir(self) -> Path:
        return self.repo_root / 'prompts'

    @property
    def llm_bench_script(self) -> Path:
        return self.repo_root / 'openvino.genai' / 'tools' / 'llm_bench' / 'benchmark.py'

    @property
    def wa_config_path(self) -> Path:
        config_path = os.environ.get('LLM_BENCH_CONFIG')
        if config_path:
            return Path(convert_path(config_path))
        return self.repo_root / 'res' / 'config.enable_PA.json'


def build_config(
    *,
    output_dir: Path,
    cache_dir: Path,
    model_dir: Path,
    model_date: str,
    device: str = 'GPU',
    timeout_sec: int = 1800,
    short_run: bool = False,
    tee_raw_log: bool = False,
    now: str | None = None,
) -> DailyConfig:
    # run.py passes its own stamp so every artefact of one run shares it;
    # generating it here would drift by a minute across the process boundary.
    now = now or dt.datetime.now().strftime('%Y%m%d_%H%M')
    ov_version = _ov_get_version()
    stem = f'daily.{now}'

    out_token_length = 32 if short_run else 256
    benchmark_iter_num = 1 if short_run else 3

    output_dir = Path(convert_path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    return DailyConfig(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        cache_dir=Path(convert_path(cache_dir)),
        model_dir=Path(convert_path(model_dir)),
        model_date=model_date,
        device=device,
        timeout_sec=timeout_sec,
        short_run=short_run,
        tee_raw_log=tee_raw_log,
        out_token_length=out_token_length,
        benchmark_iter_num=benchmark_iter_num,
        ov_version=ov_version,
        now=now,
        raw_log_path=output_dir / f'{stem}.raw',
        json_report_path=output_dir / f'{stem}.pytest.json',
    )


def ensure_utf8_env() -> None:
    os.environ['PYTHONUTF8'] = '1'
