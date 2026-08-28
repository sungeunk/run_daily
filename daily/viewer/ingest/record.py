"""Normalised run record shared by old/new loaders.

The DB writer only sees ``RunRecord`` — loaders are responsible for
filling it from their respective on-disk formats. This keeps the writer
format-agnostic and makes unit testing trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DeviceRecord:
    device_index: int
    device: str | None = None
    driver: str | None = None
    eu: int | None = None
    clock_freq_mhz: float | None = None
    global_mem_size_gb: float | None = None


@dataclass
class PerfRow:
    model: str
    precision: str
    in_token: int
    out_token: int
    exec_mode: str
    value: float
    unit: str | None = None
    prompt_idx: int = 0


@dataclass
class MonitorRow:
    """Per-test machine telemetry summary from common/machine_monitor.py."""

    nodeid: str
    model: str | None = None
    precision: str | None = None
    samples: int = 0
    duration_sec: float | None = None
    gpu_clock_mhz_mean: float | None = None
    gpu_clock_mhz_min: float | None = None
    gpu_clock_mhz_max: float | None = None
    gpu_clock_max_mhz: float | None = None
    gpu_utilization_mean: float | None = None
    gpu_power_watts_mean: float | None = None
    gpu_power_watts_max: float | None = None
    gpu_temp_c_mean: float | None = None
    gpu_temp_c_max: float | None = None
    cpu_clock_mhz_mean: float | None = None
    cpu_usage_percent_mean: float | None = None
    cpu_temp_c_max: float | None = None
    host_memory_usage_mean: float | None = None
    page_faults_per_sec_mean: float | None = None
    throttled_sample_ratio: float | None = None
    throttle_reasons: str | None = None
    sample_duration_ms_max: float | None = None
    monitor_file: str | None = None


@dataclass
class RunRecord:
    run_id: str
    source_format: str                  # 'old' | 'new'
    report_file: str
    machine: str
    ts: datetime
    device: str | None = None
    purpose: str | None = None
    run_kind: str = "daily"             # 'daily' | 'pr' | 'test' | 'manual'
    description: str | None = None
    ww: str | None = None
    ov_version: str | None = None
    ov_build: str | None = None
    ov_sha: str | None = None
    host_info: str | None = None
    host_memory_size_gb: float | None = None
    host_memory_speed_mhz: float | None = None
    gpu_info: str | None = None
    gpu_driver_version: str | None = None
    genai_version: str | None = None
    genai_commit: str | None = None
    tok_commit: str | None = None
    short_run: bool = False
    total_tests: int | None = None
    passed_tests: int | None = None
    failed_tests: int | None = None
    error_tests: int | None = None
    skipped_tests: int | None = None
    skipped_cases: int | None = None
    duration_sec: float | None = None
    source_path: str | None = None
    build_url: str | None = None
    rawlog_path: str | None = None
    file_hash: str | None = None
    devices: list[DeviceRecord] = field(default_factory=list)
    perf: list[PerfRow] = field(default_factory=list)
    monitor: list[MonitorRow] = field(default_factory=list)
