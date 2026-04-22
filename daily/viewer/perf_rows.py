#!/usr/bin/env python3
"""Flatten ``summary.json`` into one row per (key, value) — the shape the
xlsx writer and future DuckDB ingest both need.

Row schema::

    {
      'model':     str,
      'precision': str,
      'in_cat':    'short' | 'long' | int,   # category OR raw count
      'out_cat':   'short' | 'long' | int,
      'exec_mode': '1st' | '2nd' | 'pipeline' | 'batch:1' | 'batch:64' | ...
      'value':    float,                      # ms (llm/sd) or FPS (benchmark_app)
      'unit':     'ms' | 'FPS' | 'img/s' | 'tps',
    }

The key ``(model, precision, in_cat, out_cat, exec_mode)`` is the cell lookup
key the master xlsx uses — exactly mirroring the old ``FIXED_ROW_ORDER``
tuples so existing templates keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


# Token classification thresholds taken from the old viewer
# (scripts/run_daily_report_viewer_2.py::classify_token_size, threshold=100).
IN_TOKEN_THRESHOLD = 100
OUT_TOKEN_THRESHOLD = 100


def _classify(token: int | float | None, threshold: int) -> str | int:
    """Map raw token counts to the 'short'/'long' categories the xlsx uses."""
    if token is None or token == 0:
        return 0
    try:
        t = int(token)
    except (TypeError, ValueError):
        return 0
    return 'short' if t < threshold else 'long'


def _load(summary_path: Path | str) -> dict:
    if isinstance(summary_path, (str, Path)) and Path(summary_path).is_file():
        return json.loads(Path(summary_path).read_text(encoding='utf-8'))
    # Allow callers to pass an already-loaded dict.
    if isinstance(summary_path, dict):
        return summary_path
    raise FileNotFoundError(summary_path)


def _llm_benchmark_rows(m: dict) -> Iterable[dict]:
    """Emit two rows per data item: 1st + 2nd inference latency (ms)."""
    model = m.get('model', '')
    precision = m.get('precision', '')
    for d in m.get('data', []) or []:
        perf = d.get('perf') or []
        in_cat = _classify(d.get('in_token'), IN_TOKEN_THRESHOLD)
        # out_token for LLMs is always the request count (256/32 short-run).
        # The xlsx template uses the raw number here, not a category.
        out_tok = int(d.get('out_token') or 0)
        if len(perf) > 0 and perf[0] is not None:
            yield dict(model=model, precision=precision, in_cat=in_cat,
                       out_cat=out_tok, exec_mode='1st',
                       value=float(perf[0]), unit='ms')
        if len(perf) > 1 and perf[1] is not None:
            yield dict(model=model, precision=precision, in_cat=in_cat,
                       out_cat=out_tok, exec_mode='2nd',
                       value=float(perf[1]), unit='ms')


def _benchmark_app_rows(m: dict) -> Iterable[dict]:
    batch = m.get('batch', 0)
    for d in m.get('data', []) or []:
        perf = d.get('perf') or []
        if perf:
            yield dict(model=m.get('model', ''), precision=m.get('precision', ''),
                       in_cat=0, out_cat=0, exec_mode=f'batch:{batch}',
                       value=float(perf[0]), unit='FPS')


def _sd_genai_rows(m: dict) -> Iterable[dict]:
    """One pipeline row per data item. The xlsx lookup is token-agnostic for
    SD family (handled in the xlsx matcher), so we emit (0, 0) here too for
    non-whisper cases.

    whisper-large-v3 uses ``out_cat`` = 'short'/'long' to distinguish audio
    clip lengths, matching the old template.
    """
    model = m.get('model', '')
    precision = m.get('precision', '')
    is_whisper = model.startswith('whisper')
    for d in m.get('data', []) or []:
        gen_sec = d.get('generation_time_sec')
        if gen_sec is None:
            continue
        if is_whisper:
            out_cat = _classify(d.get('output_token_size'), OUT_TOKEN_THRESHOLD)
            in_cat = 0
        else:
            # Keep raw in_token for SD pipelines (32 in old template) so
            # template rows like ('stable-diffusion-v1-5', 'FP16', 32, 0, ...)
            # still match after migration.
            in_cat = int(d.get('input_token_size') or 0)
            out_cat = 0
        # Multiply by 1000 for ms convention? The old viewer kept SD in
        # seconds; we preserve that by emitting seconds with unit='s'.
        yield dict(model=model, precision=precision, in_cat=in_cat,
                   out_cat=out_cat, exec_mode='pipeline',
                   value=float(gen_sec), unit='s')


def _sd_dgfx_rows(m: dict) -> Iterable[dict]:
    for d in m.get('data', []) or []:
        sec = d.get('pipeline_sec')
        if sec is None:
            continue
        yield dict(model=m.get('model', ''), precision=m.get('precision', ''),
                   in_cat=0, out_cat=0, exec_mode='pipeline',
                   value=float(sec), unit='s')


def _qwen_usage_rows(m: dict) -> Iterable[dict]:
    # Disabled on current rigs; still define the mapping so the row is ready
    # when the binary comes back. Emits 1st/2nd inference plus memory.
    for d in m.get('data', []) or []:
        perf = d.get('perf') or []
        in_cat = _classify(d.get('in_token'), IN_TOKEN_THRESHOLD)
        out_tok = int(d.get('out_token') or 0)
        if len(perf) > 0 and perf[0] is not None:
            yield dict(model='qwen_usage', precision='INT8', in_cat=in_cat,
                       out_cat=out_tok, exec_mode='1st',
                       value=float(perf[0]), unit='ms')
        if len(perf) > 1 and perf[1] is not None:
            yield dict(model='qwen_usage', precision='INT8', in_cat=in_cat,
                       out_cat=out_tok, exec_mode='2nd',
                       value=float(perf[1]), unit='ms')


def _whisper_base_rows(m: dict) -> Iterable[dict]:
    for d in m.get('data', []) or []:
        perf = d.get('perf') or []
        if perf:
            yield dict(model=m.get('model', ''), precision=m.get('precision', ''),
                       in_cat=0, out_cat=0, exec_mode='tps',
                       value=float(perf[0]), unit='tps')


_TYPE_HANDLERS = {
    'llm_benchmark':  _llm_benchmark_rows,
    'benchmark_app':  _benchmark_app_rows,
    'sd_genai':       _sd_genai_rows,
    'sd_dgfx':        _sd_dgfx_rows,
    'qwen_usage':     _qwen_usage_rows,
    'whisper_base':   _whisper_base_rows,
    # chat_sample intentionally omitted — no perf data.
}


def flatten(summary: dict | Path | str) -> list[dict]:
    """Return a flat list of perf rows for every passed test in the summary.

    Failed/skipped tests are skipped — they have no reliable numbers to put
    into the master xlsx.
    """
    summary = _load(summary)
    rows: list[dict] = []
    for t in summary.get('tests', []):
        if t.get('outcome') != 'passed':
            continue
        metrics = t.get('metrics', {}) or {}
        handler = _TYPE_HANDLERS.get(metrics.get('test_type'))
        if handler is None:
            continue
        rows.extend(handler(metrics))
    return rows


def as_lookup(rows: list[dict]) -> dict[tuple, float]:
    """Convert rows to a ``{(model, precision, in_cat, out_cat, exec): value}``
    dict for O(1) xlsx cell fills."""
    return {
        (r['model'], r['precision'], r['in_cat'], r['out_cat'], r['exec_mode']):
        r['value']
        for r in rows
    }


if __name__ == '__main__':
    # Tiny CLI for ad-hoc debugging: python -m viewer.perf_rows path/to/summary.json
    import sys
    for row in flatten(sys.argv[1]):
        print(row)
