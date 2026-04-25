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
that case we fall back to the filename and a companion ``.report`` text
file the legacy loader also knows how to parse.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ._common import (file_hash, parse_stamp_from_name, run_id_of,
                      split_ov_version, workweek_of)
from .record import PerfRow, RunRecord

log = logging.getLogger(__name__)


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
        if len(perf) > 0 and perf[0] is not None:
            yield PerfRow(model, precision, in_tok, out_tok, "1st",
                          float(perf[0]), "ms")
        if len(perf) > 1 and perf[1] is not None:
            yield PerfRow(model, precision, in_tok, out_tok, "2nd",
                          float(perf[1]), "ms")


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
    "sd_genai":      _sd_genai_rows,
    "sd_dgfx":       _sd_dgfx_rows,
    # Deliberately dropped: qwen_usage (TestMeasuredUsageCpp) and
    # whisper_base (TestWhisperBase). No longer part of the daily signal.
    # chat_sample: no perf data.
}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _extract_meta_from_summary(summary: dict) -> dict:
    return summary.get("meta") or {}


def _guess_machine(path: Path) -> str:
    """Best-effort machine guess when ``meta`` is absent.

    When the results live under ``/var/www/html/daily/<MACHINE>/...`` the
    immediate parent directory is authoritative.
    """
    return path.parent.name


def _raw_log_candidate(path: Path) -> Path | None:
    """Find the ``.raw`` file that went with this summary.json.

    daily.<stamp>.<ov_version>.raw  OR  daily.<stamp>.none.raw
    """
    stem = path.name.split(".summary.json")[0]  # 'daily.20260419_2339'
    matches = sorted(path.parent.glob(f"{stem}.*.raw"))
    return matches[0] if matches else None


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
    ov_version = meta.get("ov_version")
    ov_build = meta.get("ov_build") or None
    ov_sha = meta.get("ov_sha") or None
    if not ov_build or not ov_sha:
        b, s = split_ov_version(ov_version)
        ov_build = ov_build or b
        ov_sha = ov_sha or s
    ww = meta.get("workweek") or workweek_of(ts)

    rec = RunRecord(
        run_id=run_id_of(machine, ts, path.name),
        source_format="new",
        report_file=path.name,
        machine=machine,
        ts=ts,
        device=meta.get("device"),
        purpose=meta.get("description"),
        description=meta.get("description"),
        ww=ww,
        ov_version=ov_version,
        ov_build=ov_build,
        ov_sha=ov_sha,
        short_run=bool(meta.get("short_run", False)),
        source_path=str(path),
        rawlog_path=str(rawlog) if (rawlog := _raw_log_candidate(path)) else None,
        file_hash=file_hash(path),
    )

    for t in summary.get("tests", []):
        if t.get("outcome") != "passed":
            continue
        metrics = t.get("metrics") or {}
        handler = _TYPE_HANDLERS.get(metrics.get("test_type"))
        if handler is None:
            continue
        rec.perf.extend(handler(metrics))

    return rec
