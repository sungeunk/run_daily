#!/usr/bin/env python3
"""Pytest configuration and shared fixtures for the daily suite.

Key design decisions
--------------------
* The session-level ``daily_config`` fixture is built from pytest CLI options
  so the same code path works whether pytest is invoked directly or through
  ``run.py``.
* ``raw_log`` is a single session-scoped file that every subprocess tees into
  — this satisfies "one raw log file for the whole run".
* ``isolate_test`` is ``autouse`` and runs the cache cleaner before and after
  every test — this satisfies "tests must not affect each other".
* Metrics are attached via ``record_metrics`` which appends to
  ``request.node.user_properties``; ``pytest-json-report`` then surfaces them
  in the JSON report, which is what downstream apps consume.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from common.cache import clear_caches
from common.cmd_runner import CmdResult, run_cmd
from common.config import DailyConfig, build_config, ensure_utf8_env
from common.profiling import HWResourceTracker, ResourceStats, sizeof_fmt


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def _default_model_dir() -> str:
    import platform
    if platform.system() == 'Linux':
        return '/var/www/html/models/daily'
    return 'c:/dev/models/daily'


def _default_device() -> str:
    import os
    return os.environ.get('DAILY_DEVICE', 'GPU')


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('daily')
    group.addoption('--device', default=_default_device(),
                    help='Target OpenVINO device (override: $DAILY_DEVICE)')
    group.addoption('--model-dir', default=_default_model_dir(),
                    help='Root directory for models')
    group.addoption('--model-date', default='WW45_llm-optimum_2025.4.0-20381-RC1',
                    help='Model cache subdirectory under --model-dir')
    group.addoption('--cache-dir', default=None,
                    help='OpenVINO cache directory (defaults to <repo>/llm-cache)')
    group.addoption('--output-dir', default=None,
                    help='Output directory (defaults to <repo>/output)')
    group.addoption('--daily-timeout', default=1800, type=int,
                    help='Per-subprocess timeout in seconds')
    group.addoption('--short-run', action='store_true',
                    help='Reduced token/iter counts for quick smoke runs')


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def daily_config(request: pytest.FixtureRequest) -> DailyConfig:
    ensure_utf8_env()

    opt = request.config.getoption
    repo_root = Path(__file__).resolve().parent.parent

    cache_dir = opt('--cache-dir') or (repo_root / 'llm-cache')
    output_dir = opt('--output-dir') or (repo_root / 'output')

    cfg = build_config(
        output_dir=Path(output_dir),
        cache_dir=Path(cache_dir),
        model_dir=Path(opt('--model-dir')),
        model_date=opt('--model-date'),
        device=opt('--device'),
        timeout_sec=opt('--daily-timeout'),
        short_run=opt('--short-run'),
    )
    return cfg


class RawLogSink:
    """Thread-safe append-only writer for the session raw log."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._fp = open(path, 'w', encoding='utf-8', buffering=1)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, text: str) -> None:
        with self._lock:
            self._fp.write(text)

    def section(self, title: str) -> None:
        banner = f'\n===== {title} =====\n'
        self.write(banner)

    def close(self) -> None:
        with self._lock:
            if not self._fp.closed:
                self._fp.flush()
                self._fp.close()


@pytest.fixture(scope='session')
def raw_log(daily_config: DailyConfig) -> RawLogSink:
    sink = RawLogSink(daily_config.raw_log_path)
    sink.write(f'# Daily run {daily_config.now}\n')
    sink.write(f'# OpenVINO: {daily_config.ov_version}\n')
    sink.write(f'# Device:   {daily_config.device}\n\n')
    yield sink
    sink.close()


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_test(daily_config: DailyConfig, raw_log: RawLogSink,
                 request: pytest.FixtureRequest):
    """Cache cleanup around every test so runs are independent."""
    raw_log.section(f'TEST START: {request.node.nodeid}')
    clear_caches(daily_config.cache_dir, daily_config.model_dir)
    yield
    clear_caches(daily_config.cache_dir, daily_config.model_dir)
    raw_log.section(f'TEST END: {request.node.nodeid}')


@pytest.fixture
def run_subprocess(daily_config: DailyConfig, raw_log: RawLogSink
                   ) -> Callable[..., CmdResult]:
    """Run a shell command, teeing stdout into the session raw log.

    ``cmd`` can be a string (shlex-split on POSIX) or a list (passed through).
    ``cwd`` optionally changes the working directory for the duration of the
    call — used by tests whose scripts rely on relative paths.
    """
    import os

    def _run(cmd, *, cwd: str | None = None) -> CmdResult:
        raw_log.write(f'[CMD] {cmd}\n')
        if cwd:
            raw_log.write(f'[CWD] {cwd}\n')

        old_cwd = os.getcwd() if cwd else None
        if cwd:
            os.chdir(cwd)
        try:
            result = run_cmd(cmd, timeout_sec=daily_config.timeout_sec,
                             log_sink=raw_log.write)
        finally:
            if old_cwd:
                os.chdir(old_cwd)

        raw_log.write(f'[RC ] {result.returncode} '
                      f'(duration {result.duration_sec:.1f}s)\n')
        return result
    return _run


@pytest.fixture
def hw_tracker():
    """Factory returning a fresh HWResourceTracker.

    Usage::

        tracker = hw_tracker()
        tracker.start()
        result = run_subprocess(cmd)
        stats = tracker.stop()   # ResourceStats
    """
    return HWResourceTracker


@pytest.fixture
def record_metrics(request: pytest.FixtureRequest) -> Callable[[dict], None]:
    """Attach a metrics dict to the current test's user_properties.

    pytest-json-report serialises user_properties into the JSON report, where
    each item is a ``[name, value]`` pair. We always use the key ``metrics``
    so the report builder can find it with a single lookup.
    """
    def _record(data: dict[str, Any]) -> None:
        # Enforce JSON-serialisability up-front so failures surface here,
        # not when the report plugin tries to write the file.
        json.dumps(data)
        request.node.user_properties.append(('metrics', data))
    return _record


# ---------------------------------------------------------------------------
# Logging: quieter pytest capture so raw log stays authoritative
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def _configure_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname).1s] %(message)s',
                        datefmt='%H:%M:%S')
    yield
