"""Loader for the *new* summary.json output produced by daily/run.py.

Structure recap::

    {
      "generated_at": <epoch>,
      "duration_sec": <float>,
      "meta": {machine, device, workweek, ov_version, ov_build, ov_sha, ...},
      "totals": {passed, failed, ...},
      "tests": [
        {"nodeid", "outcome", "duration_sec", "failure",
         "metrics": {"test_type", "model", "precision", "data": [...]}
        }
      ]
    }

``meta`` was added mid-migration — early summary files don't have it. In
that case we fall back to the filename.
"""

from __future__ import annotations

import json
import logging
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ._common import (file_hash, parse_stamp_from_name, run_id_of,
                      split_ov_version, workweek_of)
from .record import MonitorRow, PerfRow, RunRecord

log = logging.getLogger(__name__)

# Free-form purpose text is the only hint about who launched a run, so map it
# to a fixed vocabulary the viewer can filter on. Order matters: an explicit
# PR marker wins over everything, and a run that calls itself daily stays
# daily even if its description also mentions validation.
# Word boundaries keep short tokens like 'ci' and 'pr' from matching inside
# words such as 'precision'.
_RUN_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pr", r"\bpr[-_# ]*\d+|\bpull[-_ ]?request\b|\bpre-?commit\b"),
    ("daily", r"\bdaily|\bnightly\b|\bweekly\b"),
    ("test", r"\btest|\btrial\b|\bdebug\b|\bexperiment|\bjenkins\b|\bci\b|\bvalidation\b"),
)


def classify_run_kind(purpose: str | None, description: str | None = None) -> str:
    """Map purpose/description text onto 'daily' | 'pr' | 'test' | 'manual'."""
    text = f"{purpose or ''} {description or ''}".strip().lower()
    if not text:
        return "manual"
    for kind, pattern in _RUN_KIND_PATTERNS:
        if re.search(pattern, text):
            return kind
    return "manual"


# ---------------------------------------------------------------------------
# Per-test-type perf extractors (raw tokens preserved — no bucketing)
# ---------------------------------------------------------------------------

def _llm_rows(m: dict) -> Iterable[PerfRow]:
    model = m.get("model", "")
    precision = m.get("precision", "")
    for d in m.get("data", []) or []:
        perf = d.get("perf") or []
        in_tok = int(d.get("in_token") or 0)
        out_tok = int(d.get("out_token") or 0)
        prompt_idx = int(d.get("prompt_idx") or 0)
        if len(perf) > 0 and perf[0] is not None:
            yield PerfRow(model, precision, in_tok, out_tok, "1st",
                          float(perf[0]), "ms", prompt_idx=prompt_idx)
        if len(perf) > 1 and perf[1] is not None:
            yield PerfRow(model, precision, in_tok, out_tok, "2nd",
                          float(perf[1]), "ms", prompt_idx=prompt_idx)


def _benchmark_app_rows(m: dict) -> Iterable[PerfRow]:
    model = m.get("model", "")
    precision = m.get("precision", "")
    batch = m.get("batch", 0)
    for d in m.get("data", []) or []:
        perf = d.get("perf") or []
        if perf:
            yield PerfRow(model, precision, 0, 0, f"batch:{batch}",
                          float(perf[0]), "FPS")


def _sd_genai_rows(m: dict) -> Iterable[PerfRow]:
    model = m.get("model", "")
    precision = m.get("precision", "")
    is_whisper = model.startswith("whisper")
    for d in m.get("data", []) or []:
        gen_sec = d.get("generation_time_sec")
        if gen_sec is None:
            continue
        if is_whisper:
            in_tok = 0
            out_tok = int(d.get("output_token_size") or 0)
        else:
            in_tok = int(d.get("input_token_size") or 0)
            out_tok = int(d.get("output_token_size") or 0)
        yield PerfRow(model, precision, in_tok, out_tok, "pipeline",
                      float(gen_sec), "s")


def _sd_dgfx_rows(m: dict) -> Iterable[PerfRow]:
    model = m.get("model", "")
    precision = m.get("precision", "")
    for d in m.get("data", []) or []:
        sec = d.get("pipeline_sec")
        if sec is None:
            continue
        yield PerfRow(model, precision, 0, 0, "pipeline", float(sec), "s")


_TYPE_HANDLERS = {
    "llm_benchmark": _llm_rows,
    "benchmark_app": _benchmark_app_rows,
    "image_generation": _sd_genai_rows,
    "sd_dgfx":       _sd_dgfx_rows,
    # chat_sample has no perf data.
}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _extract_meta_from_summary(summary: dict) -> dict:
    return summary.get("meta") or {}


def _guess_machine(path: Path) -> str:
    """Best-effort machine guess when ``meta`` is absent.

    Results live under ``<root>/<MACHINE>/<YYYY.MM>/...`` (older runs sit
    directly under ``<MACHINE>/``), so the month bucket is stepped over.
    """
    parent = path.parent
    if re.fullmatch(r"\d{4}\.\d{2}", parent.name):
        parent = parent.parent
    if parent.name in {"output", "viewer", "daily"}:
        return platform.node()
    return parent.name


def _raw_log_candidate(path: Path) -> Path | None:
    """Find the ``.raw`` file that went with this summary.json.

    New format: ``daily.<stamp>.raw``.
    Legacy formats: ``daily.<stamp>.<ov_version>.raw`` and
    ``daily.<stamp>.none.raw``.
    """
    stem = path.name.split(".summary.json")[0]  # 'daily.20260419_2339'
    direct = path.parent / f"{stem}.raw"
    if direct.exists():
        return direct

    matches = sorted(path.parent.glob(f"{stem}.*.raw"))
    return matches[0] if matches else None


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num == num else None


def _int_or_none(value) -> int | None:
    num = _float_or_none(value)
    return None if num is None else int(num)


def _skipped_cases(summary: dict) -> int:
    """Benchmark cases a run never attempted because its test was skipped."""
    return sum(
        int((test.get("metrics") or {}).get("expected_series") or 0)
        for test in summary.get("tests", []) or []
        if test.get("outcome") == "skipped"
    )


def _stat(machine: dict, field: str, key: str) -> float | None:
    """Read one min/max/mean entry; absent probes serialise as the string 'N/A'."""
    block = machine.get(field)
    if not isinstance(block, dict):
        return None
    return _float_or_none(block.get(key))


def _monitor_row(test: dict, metrics: dict) -> MonitorRow | None:
    machine = metrics.get("machine")
    if not isinstance(machine, dict) or not machine.get("samples"):
        return None

    reasons = machine.get("gpu_throttle_reasons_seen")
    if isinstance(reasons, list):
        reasons = ",".join(str(r) for r in reasons) or None
    elif reasons is not None and not isinstance(reasons, str):
        reasons = str(reasons)

    return MonitorRow(
        nodeid=test.get("nodeid", ""),
        model=metrics.get("model"),
        precision=metrics.get("precision"),
        samples=int(machine.get("samples") or 0),
        duration_sec=_float_or_none(machine.get("duration_sec")),
        gpu_clock_mhz_mean=_stat(machine, "gpu_clock_mhz", "mean"),
        gpu_clock_mhz_min=_stat(machine, "gpu_clock_mhz", "min"),
        gpu_clock_mhz_max=_stat(machine, "gpu_clock_mhz", "max"),
        gpu_clock_max_mhz=_float_or_none(machine.get("gpu_clock_max_mhz")),
        gpu_utilization_mean=_stat(machine, "gpu_utilization_percent", "mean"),
        gpu_power_watts_mean=_stat(machine, "gpu_power_watts", "mean"),
        gpu_power_watts_max=_stat(machine, "gpu_power_watts", "max"),
        gpu_temp_c_mean=_stat(machine, "lhm_gpu_temp_c", "mean"),
        gpu_temp_c_max=_stat(machine, "lhm_gpu_temp_c", "max"),
        cpu_clock_mhz_mean=_stat(machine, "cpu_clock_mhz", "mean"),
        cpu_usage_percent_mean=_stat(machine, "cpu_usage_percent", "mean"),
        cpu_temp_c_max=_stat(machine, "cpu_temp_c", "max"),
        host_memory_usage_mean=_stat(machine, "host_memory_usage_percent", "mean"),
        page_faults_per_sec_mean=_stat(machine, "process_page_faults_per_sec", "mean"),
        throttled_sample_ratio=_float_or_none(machine.get("gpu_throttled_sample_ratio")),
        throttle_reasons=reasons,
        sample_duration_ms_max=_stat(machine, "sample_duration_ms", "max"),
        monitor_file=machine.get("file"),
    )


def load_summary(path: Path) -> RunRecord:
    """Parse a summary.json into a RunRecord (raw tokens preserved)."""
    path = Path(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    meta = _extract_meta_from_summary(summary)

    # Timestamp: prefer the stamp from meta, fall back to filename, then mtime.
    ts: datetime | None = None
    stamp = meta.get("stamp")
    if stamp:
        try:
            ts = datetime.strptime(stamp, "%Y%m%d_%H%M")
        except ValueError:
            ts = None
    if ts is None:
        ts = parse_stamp_from_name(path.name)
    if ts is None:
        generated_at = summary.get("generated_at")
        if generated_at:
            ts = datetime.fromtimestamp(float(generated_at))
    if ts is None:
        ts = datetime.fromtimestamp(path.stat().st_mtime)

    machine = meta.get("machine") or _guess_machine(path)
    purpose = meta.get("purpose") or meta.get("description")
    ov_version = meta.get("ov_version")
    ov_build = meta.get("ov_build") or None
    ov_sha = meta.get("ov_sha") or None
    if not ov_build or not ov_sha:
        b, s = split_ov_version(ov_version)
        ov_build = ov_build or b
        ov_sha = ov_sha or s
    ww = meta.get("workweek") or workweek_of(ts)
    totals = summary.get("totals") or {}
    host_info = meta.get("host_info")
    host_memory_size_gb = _float_or_none(meta.get("host_memory_size_gb"))
    host_memory_speed_mhz = _float_or_none(meta.get("host_memory_speed_mhz"))
    gpu_info = meta.get("gpu_info") or meta.get("device")
    gpu_driver_version = meta.get("gpu_driver_version")
    devices = []

    rec = RunRecord(
        run_id=run_id_of(machine, ts, path.name),
        source_format="new",
        report_file=path.name,
        machine=machine,
        ts=ts,
        device=meta.get("device"),
        purpose=purpose,
        run_kind=classify_run_kind(purpose, meta.get("description")),
        description=purpose,
        ww=ww,
        ov_version=ov_version,
        ov_build=ov_build,
        ov_sha=ov_sha,
        host_info=host_info,
        host_memory_size_gb=host_memory_size_gb,
        host_memory_speed_mhz=host_memory_speed_mhz,
        gpu_info=gpu_info,
        gpu_driver_version=gpu_driver_version,
        genai_version=meta.get("genai_version") or None,
        genai_commit=meta.get("genai_commit") or None,
        tok_commit=meta.get("tok_commit") or None,
        short_run=bool(meta.get("short_run", False)),
        total_tests=_int_or_none(totals.get("total")),
        passed_tests=_int_or_none(totals.get("passed")),
        failed_tests=_int_or_none(totals.get("failed")),
        error_tests=_int_or_none(totals.get("error")),
        skipped_tests=_int_or_none(totals.get("skipped")),
        skipped_cases=_skipped_cases(summary),
        duration_sec=_float_or_none(summary.get("duration_sec")),
        source_path=str(path),
        rawlog_path=str(rawlog) if (rawlog := _raw_log_candidate(path)) else None,
        file_hash=file_hash(path),
        devices=devices,
    )

    for t in summary.get("tests", []):
        metrics = t.get("metrics") or {}
        # Telemetry is kept even for failed tests: a crash on a throttled or
        # overloaded machine is exactly the case worth seeing.
        if (monitor_row := _monitor_row(t, metrics)) is not None:
            rec.monitor.append(monitor_row)
        if t.get("outcome") != "passed":
            continue
        handler = _TYPE_HANDLERS.get(metrics.get("test_type"))
        if handler is None:
            continue
        rec.perf.extend(handler(metrics))

    return rec
