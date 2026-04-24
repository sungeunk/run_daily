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


@dataclass
class RunRecord:
    run_id: str
    source_format: str                  # 'old' | 'new'
    report_file: str
    machine: str
    ts: datetime
    device: str | None = None
    purpose: str | None = None
    description: str | None = None
    ww: str | None = None
    ov_version: str | None = None
    ov_build: str | None = None
    ov_sha: str | None = None
    genai_version: str | None = None
    genai_commit: str | None = None
    tok_commit: str | None = None
    short_run: bool = False
    source_path: str | None = None
    rawlog_path: str | None = None
    file_hash: str | None = None
    devices: list[DeviceRecord] = field(default_factory=list)
    perf: list[PerfRow] = field(default_factory=list)
