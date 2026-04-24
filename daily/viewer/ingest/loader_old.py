"""Loader for the legacy pickle + .report combo produced by scripts/report.py.

Pickle layout::

    {
      (model_name, precision, <TestClass>): [
        {
          "cmd":         str,
          "raw_log":     str,
          "return_code": int,
          "data_list":   [ {"in_token": int, "out_token": int,
                            "perf": [latency_1st_ms, latency_2nd_ms, ...],
                            "generated_text": str}, ...],
          "process_time": float,
          # TestBenchmarkapp extras
          "test_config": {"batch": 1} or {"mem_check": True},
          # TestMeasuredUsageCpp extras
          "peak_cpu_usage_percent": float,
          "peak_mem_usage_percent": float,
          "peak_mem_usage_size":    str ("8.5 GB"),
        }, ...
      ]
    }

The tuple key contains the test-class *object*, which pickle serialises by
fully-qualified name. To unpickle without pulling in the legacy test_cases
package, we install a ``find_class`` override that replaces unknown classes
with a lightweight stand-in carrying just the class name.

The ``.report`` sibling file supplies ``purpose`` and the OpenVINO commit
string, which aren't in the pickle.
"""

from __future__ import annotations

import io
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path

from ._common import (file_hash, parse_stamp_from_name, run_id_of,
                      split_ov_version, workweek_of)
from .record import PerfRow, RunRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pickle decoding without the original modules
# ---------------------------------------------------------------------------

class _UnknownClass:
    """Stand-in for legacy test-case classes we no longer import."""

    def __init__(self, qualname: str):
        self._qualname = qualname

    def __repr__(self) -> str:
        return f"<UnknownClass {self._qualname}>"

    @property
    def __name__(self) -> str:
        return self._qualname.rsplit(".", 1)[-1]


class _TolerantUnpickler(pickle.Unpickler):
    """Replace class references with _UnknownClass instances on the fly."""

    def find_class(self, module: str, name: str):
        # Built-ins pass through so core types (list, dict, tuple, datetime,
        # numpy floats, ...) still unpickle correctly.
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _UnknownClass(f"{module}.{name}")


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return _TolerantUnpickler(f).load()


# ---------------------------------------------------------------------------
# .report text parsing (purpose + commit)
# ---------------------------------------------------------------------------

_PURPOSE_TABLE_RE = re.compile(
    r"(?im)^\s*\|\s*Purpose\s*\|\s*(?P<purpose>[^|]+?)\s*\|")
_PURPOSE_FALLBACK_RE = re.compile(r"(?im)^\s*Purpose\s*:\s*(?P<purpose>.+?)\s*$")
_OPENVINO_LINE_RE = re.compile(
    r"(?is)OpenVINO[^\n\r]*?-(?P<build>\d+)-(?P<sha>[0-9a-fA-F]{7,40})")


def _parse_report_text(path: Path) -> tuple[str | None, str | None, str | None]:
    """Return ``(purpose, ov_build, ov_sha)``. Missing values come back as None."""
    if not path.exists():
        return None, None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("Failed to read %s: %s", path, e)
        return None, None, None

    purpose = None
    m = _PURPOSE_TABLE_RE.search(text)
    if m:
        purpose = m.group("purpose").strip()
    else:
        m = _PURPOSE_FALLBACK_RE.search(text)
        if m:
            purpose = m.group("purpose").strip()

    build = sha = None
    m = _OPENVINO_LINE_RE.search(text)
    if m:
        build, sha = m.group("build"), m.group("sha").lower()

    return purpose, build, sha


# ---------------------------------------------------------------------------
# OV version embedded in pickle filenames: daily.<stamp>.<ov_version>.pickle
# ---------------------------------------------------------------------------

_FILENAME_OV_RE = re.compile(
    r"^daily\.\d{8}_\d{3,4}\.(?P<ov>[^.]+(?:\.[^.]+)*)\.pickle$")


def _ov_version_from_filename(name: str) -> str | None:
    m = _FILENAME_OV_RE.match(name)
    return m.group("ov") if m else None


# ---------------------------------------------------------------------------
# Perf extractors per legacy TestClass
# ---------------------------------------------------------------------------

def _class_name(key_tuple: tuple) -> str:
    """Third element of the key is the TestClass; we only need its short name."""
    cls = key_tuple[2]
    return getattr(cls, "__name__", str(cls))


def _sizestr_to_gb(s: str | float | None) -> float | None:
    """Legacy memory size strings look like '8.5 GB' or '1024 MB'."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        # Old rigs wrote bytes; others wrote GB. Use magnitude to guess.
        return float(s) / (1024 ** 3) if float(s) > 1024 else float(s)
    m = re.match(r"\s*([\d.]+)\s*([KMGT]?B)?\s*$", str(s), re.IGNORECASE)
    if not m:
        return None
    n = float(m.group(1))
    unit = (m.group(2) or "GB").upper()
    factor = {"B": 1 / 1024 ** 3, "KB": 1 / 1024 ** 2, "MB": 1 / 1024,
              "GB": 1.0, "TB": 1024.0}.get(unit, 1.0)
    return n * factor


def _benchmark_perf(key: tuple, items: list) -> list[PerfRow]:
    model, precision, _ = key
    out: list[PerfRow] = []
    for cmd_item in items:
        if cmd_item.get("return_code", -1) != 0:
            continue
        for d in cmd_item.get("data_list", []) or []:
            in_tok = int(d.get("in_token") or 0)
            out_tok = int(d.get("out_token") or 0)
            perf = d.get("perf") or []
            if len(perf) > 0 and perf[0] is not None:
                out.append(PerfRow(model, precision, in_tok, out_tok,
                                   "1st", float(perf[0]), "ms"))
            if len(perf) > 1 and perf[1] is not None:
                out.append(PerfRow(model, precision, in_tok, out_tok,
                                   "2nd", float(perf[1]), "ms"))
    return out


def _benchmark_app_perf(key: tuple, items: list) -> list[PerfRow]:
    model, precision, _ = key
    out: list[PerfRow] = []
    for cmd_item in items:
        if cmd_item.get("return_code", -1) != 0:
            continue
        batch = (cmd_item.get("test_config") or {}).get("batch", 0)
        for d in cmd_item.get("data_list", []) or []:
            perf = d.get("perf") or []
            if perf:
                out.append(PerfRow(model, precision, 0, 0,
                                   f"batch:{batch}", float(perf[0]), "FPS"))
    return out


def _measured_usage_perf(key: tuple, items: list) -> list[PerfRow]:
    model, precision, _ = key
    out: list[PerfRow] = []
    for cmd_item in items:
        if cmd_item.get("return_code", -1) != 0:
            continue
        peak_mem_pct = cmd_item.get("peak_mem_usage_percent")
        peak_mem_sz = _sizestr_to_gb(cmd_item.get("peak_mem_usage_size"))
        peak_cpu_pct = cmd_item.get("peak_cpu_usage_percent")
        for d in cmd_item.get("data_list", []) or []:
            in_tok = int(d.get("in_token") or 0)
            out_tok = int(d.get("out_token") or 0)
            perf = d.get("perf") or []
            if len(perf) > 0 and perf[0] is not None:
                out.append(PerfRow(model, precision, in_tok, out_tok,
                                   "1st", float(perf[0]), "ms"))
            if len(perf) > 1 and perf[1] is not None:
                out.append(PerfRow(model, precision, in_tok, out_tok,
                                   "2nd", float(perf[1]), "ms"))
        if peak_mem_pct is not None:
            out.append(PerfRow(model, precision, 0, 0, "memory percent",
                               float(peak_mem_pct), "%"))
        if peak_mem_sz is not None:
            out.append(PerfRow(model, precision, 0, 0, "memory size",
                               float(peak_mem_sz), "GB"))
        if peak_cpu_pct is not None:
            out.append(PerfRow(model, precision, 0, 0, "cpu percent",
                               float(peak_cpu_pct), "%"))
    return out


def _sd_perf(key: tuple, items: list) -> list[PerfRow]:
    """Legacy SD perf rows: perf[0] is pipeline ms."""
    model, precision, _ = key
    out: list[PerfRow] = []
    for cmd_item in items:
        if cmd_item.get("return_code", -1) != 0:
            continue
        for d in cmd_item.get("data_list", []) or []:
            perf = d.get("perf") or []
            if not perf or perf[0] is None:
                continue
            # Old format kept pipeline latency in ms already (see report.py
            # sec_to_ms conversion). Some pickles stored it in seconds — if
            # the magnitude is suspiciously small, scale up.
            v = float(perf[0])
            unit = "ms"
            # SD in/out tokens were stored at perf[4]/perf[5] in very old
            # dumps; newer ones emit them at top level.
            in_tok = int(d.get("in_token") or (perf[4] if len(perf) > 4 else 0) or 0)
            out_tok = int(d.get("out_token") or (perf[5] if len(perf) > 5 else 0) or 0)
            out.append(PerfRow(model, precision, in_tok, out_tok,
                               "pipeline", v, unit))
    return out


def _whisper_perf(key: tuple, items: list) -> list[PerfRow]:
    model, precision, _ = key
    out: list[PerfRow] = []
    for cmd_item in items:
        if cmd_item.get("return_code", -1) != 0:
            continue
        for d in cmd_item.get("data_list", []) or []:
            perf = d.get("perf") or []
            if perf:
                out.append(PerfRow(model, precision, 0, 0, "tps",
                                   float(perf[0]), "tps"))
    return out


_CLASS_HANDLERS = {
    "TestBenchmark":               _benchmark_perf,
    "TestBenchmarkapp":            _benchmark_app_perf,
    "TestMeasuredUsageCpp":        _measured_usage_perf,
    "TestStableDiffusion":         _sd_perf,
    "TestStableDiffusionGenai":    _sd_perf,
    "TestStableDiffusionDGfxE2eAi": _sd_perf,
    "TestWhisperBase":             _whisper_perf,
    # TestChatSample: no numeric perf.
}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _raw_log_candidate(report_path: Path) -> Path | None:
    """Old pipeline emitted daily.<stamp>.<ov_version>.raw alongside .report."""
    candidate = report_path.with_suffix(".raw")
    return candidate if candidate.exists() else None


def load_report(report_path: Path) -> RunRecord:
    """Parse an old ``.report`` + sibling ``.pickle`` into a RunRecord.

    ``report_path`` may be either the ``.report`` or the ``.pickle`` — we
    resolve the sibling automatically.
    """
    report_path = Path(report_path)
    if report_path.suffix == ".pickle":
        pickle_path = report_path
        text_report = report_path.with_suffix(".report")
    else:
        text_report = report_path
        pickle_path = report_path.with_suffix(".pickle")

    if not pickle_path.exists():
        raise FileNotFoundError(f"pickle sibling missing for {report_path}")

    result_root = _load_pickle(pickle_path)

    ts = parse_stamp_from_name(pickle_path.name)
    if ts is None:
        ts = datetime.fromtimestamp(pickle_path.stat().st_mtime)

    machine = pickle_path.parent.name  # e.g. /var/www/html/daily/<MACHINE>/...

    purpose, r_build, r_sha = _parse_report_text(text_report)

    ov_version = _ov_version_from_filename(pickle_path.name)
    fn_build, fn_sha = split_ov_version(ov_version)
    ov_build = r_build or fn_build
    ov_sha = r_sha or fn_sha

    rec = RunRecord(
        run_id=run_id_of(machine, ts, pickle_path.name),
        source_format="old",
        report_file=pickle_path.name,
        machine=machine,
        ts=ts,
        purpose=purpose,
        description=purpose,
        ww=workweek_of(ts),
        ov_version=ov_version,
        ov_build=ov_build,
        ov_sha=ov_sha,
        source_path=str(pickle_path),
        rawlog_path=str(rawlog) if (rawlog := _raw_log_candidate(text_report)) else None,
        file_hash=file_hash(pickle_path),
    )

    for key_tuple, items in result_root.items():
        if not isinstance(key_tuple, tuple) or len(key_tuple) < 3:
            continue
        cls_name = _class_name(key_tuple)
        handler = _CLASS_HANDLERS.get(cls_name)
        if handler is None:
            continue
        rec.perf.extend(handler(key_tuple, items))

    return rec
