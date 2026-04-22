#!/usr/bin/env python3
"""System resource tracker used by tests that need peak-memory / CPU figures.

Ported from scripts/profiling.py. Kept simple on purpose — the tracker polls
psutil on a background thread and reports min/max/mean on stop.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass

import psutil


class _Key(enum.Enum):
    TIMESTAMP = 0
    MEM_USAGE_PERCENT = 1
    MEM_USAGE_SIZE = 2
    CPU_USAGE_PERCENT = 3


@dataclass
class ResourceStats:
    cpu_usage_mean_percent: float
    mem_delta_bytes: int
    mem_delta_percent: float


class HWResourceTracker(threading.Thread):
    """Polls psutil every `period_sec` and records memory/CPU usage.

    Usage::

        tracker = HWResourceTracker()
        tracker.start()
        ... run workload ...
        stats = tracker.stop()
    """

    def __init__(self, period_sec: float = 0.1):
        super().__init__(daemon=True)
        self._period = period_sec
        self._running = True
        self._lock = threading.Lock()
        self._samples: list[list[float]] = []

    def run(self) -> None:
        while self._running:
            time.sleep(self._period)
            ts_ms = round(time.time() * 1000)
            vm = psutil.virtual_memory()
            sample = [ts_ms, vm.percent, vm.used, psutil.cpu_percent()]
            with self._lock:
                self._samples.append(sample)

    def stop(self) -> ResourceStats:
        if self.is_alive():
            self._running = False
            self.join()

        with self._lock:
            samples = list(self._samples)

        if not samples:
            return ResourceStats(0.0, 0, 0.0)

        mem_sizes = [s[_Key.MEM_USAGE_SIZE.value] for s in samples]
        mem_pcts = [s[_Key.MEM_USAGE_PERCENT.value] for s in samples]
        cpu_pcts = [s[_Key.CPU_USAGE_PERCENT.value] for s in samples]

        return ResourceStats(
            cpu_usage_mean_percent=sum(cpu_pcts) / len(cpu_pcts),
            mem_delta_bytes=int(max(mem_sizes) - min(mem_sizes)),
            mem_delta_percent=max(mem_pcts) - min(mem_pcts),
        )


def sizeof_fmt(num: float) -> str:
    for unit in ('', 'KB', 'MB', 'GB', 'TB'):
        if abs(num) < 1024.0:
            return f'{num:3.1f} {unit}'.strip()
        num /= 1024.0
    return f'{num:.1f} PB'
