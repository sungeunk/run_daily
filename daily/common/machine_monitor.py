#!/usr/bin/env python3
"""Sample machine state while a benchmark subprocess runs, then reduce it to stats.

``monitor_machine.py`` is launched as its own process rather than a thread so
its WMI / Level Zero / sysfs probes — which occasionally stall for hundreds of
milliseconds — never perturb the measurement they are meant to explain.

Usage::

    monitor = MachineMonitor(out_path)
    monitor.start(child_pid)
    ...                       # benchmark runs
    stats = monitor.stop()    # per-test summary dict
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

MONITOR_SCRIPT = Path(__file__).resolve().parent / 'monitor_machine.py'

NA = 'N/A'

# Numeric series worth reducing to min/max/mean for a per-model comparison.
_STAT_FIELDS: tuple[str, ...] = (
    'cpu_clock_mhz',
    'cpu_temp_c',
    'cpu_usage_percent',
    'gpu_clock_mhz',
    'gpu_utilization_percent',
    'gpu_power_watts',
    'gpu_memory_used_mb',
    'host_memory_usage_percent',
    'host_memory_available_mb',
    'host_commit_used_mb',
    'process_page_faults_per_sec',
)

# Constant-per-run context; the last observed value is enough.
_CONTEXT_FIELDS: tuple[str, ...] = (
    'gpu_name',
    'gpu_source',
    'gpu_clock_max_mhz',
    'gpu_memory_total_mb',
    'host_memory_total_gb',
    'host_memory_speed_mts',
    'process_priority_class',
    'process_cpu_affinity_count',
    'process_power_throttling',
    'timer_resolution_current_ms',
)

# Sysman reports a bitmask, sysfs-less hosts report nothing at all.
_NOT_THROTTLED = {None, '', '0x0', '0', 'none'}


def _stats(values: list[float]) -> dict[str, float] | str:
    if not values:
        return NA
    return {
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'mean': round(sum(values) / len(values), 2),
    }


def _numeric(rows: Iterable[dict], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def summarize(path: Path) -> dict:
    """Reduce a monitor JSONL file to a compact, JSON-serialisable summary."""
    summary: dict = {'file': str(path), 'samples': 0}

    if not path.exists():
        summary['error'] = 'monitor produced no output'
        return summary

    rows: list[dict] = []
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    summary['samples'] = len(rows)
    if not rows:
        summary['error'] = 'monitor produced no samples'
        return summary

    start = rows[0].get('t_monotonic')
    end = rows[-1].get('t_monotonic')
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        summary['duration_sec'] = round(end - start, 1)
    summary['started_utc'] = rows[0].get('timestamp_utc')

    for field in _STAT_FIELDS:
        summary[field] = _stats(_numeric(rows, field))

    for field in _CONTEXT_FIELDS:
        value = next(
            (row.get(field) for row in reversed(rows) if row.get(field) is not None),
            None,
        )
        summary[field] = NA if value is None else value

    throttled = [
        str(row.get('gpu_throttle_reasons'))
        for row in rows
        if row.get('gpu_throttle_reasons') not in _NOT_THROTTLED
    ]
    if any(row.get('gpu_throttle_reasons') is not None for row in rows):
        summary['gpu_throttled_sample_ratio'] = round(len(throttled) / len(rows), 3)
        summary['gpu_throttle_reasons_seen'] = sorted(set(throttled)) or [NA]
    else:
        summary['gpu_throttled_sample_ratio'] = NA
        summary['gpu_throttle_reasons_seen'] = NA

    # A sample that took unusually long means the probe itself was blocked, so
    # the surrounding values should be read with suspicion.
    durations = _numeric(rows, 'sample_duration_ms')
    summary['sample_duration_ms'] = _stats(durations)

    return summary


class MachineMonitor:
    """Owns the monitor subprocess for one test."""

    def __init__(
        self,
        out_path: Path,
        *,
        interval_sec: float = 0.5,
        max_duration_sec: float = 3600.0,
        gpu_full: bool = False,
        log_sink: Callable[[str], None] | None = None,
    ):
        self.out_path = Path(out_path)
        self._interval_sec = interval_sec
        self._max_duration_sec = max_duration_sec
        self._gpu_full = gpu_full
        self._log = log_sink or (lambda _text: None)
        self._proc: subprocess.Popen | None = None
        self.summary: dict | None = None

    def start(self, pid: int) -> None:
        if self._proc is not None:
            return

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(MONITOR_SCRIPT),
            '--pid', str(pid),
            '--interval-sec', str(self._interval_sec),
            '--duration-sec', str(self._max_duration_sec),
            '--out', str(self.out_path),
            '--top-processes', '5',
        ]
        if self._gpu_full:
            cmd.append('--gpu-telemetry-full')

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='UTF-8',
                errors='ignore',
            )
        except OSError as exc:
            self._log(f'[monitor] spawn failed: {exc}\n')
            self._proc = None
            return

        self._log(f'[monitor] pid={pid} -> {self.out_path}\n')

    def stop(self) -> dict:
        """Terminate the monitor and return the summary. Safe to call twice."""
        if self.summary is not None:
            return self.summary

        proc = self._proc
        if proc is not None:
            self._proc = None
            if proc.poll() is None:
                proc.terminate()
            try:
                output, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                output, _ = proc.communicate()
            for line in (output or '').splitlines():
                # Only surface problems; the INFO banner repeats every test.
                if 'ERROR' in line or 'WARNING' in line:
                    self._log(f'[monitor] {line}\n')

        self.summary = summarize(self.out_path)
        return self.summary
