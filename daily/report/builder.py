#!/usr/bin/env python3
"""Report builder: consumes the pytest-json-report output and emits a human
readable text report plus a normalised JSON summary.

pytest-json-report schema we rely on::

    {
      "created": <float>,
      "duration": <float>,
      "summary": {"passed": N, "failed": N, ...},
      "tests": [
        {
          "nodeid": "...",
          "outcome": "passed" | "failed" | ...,
          "duration": <float>,
          "call": {"longrepr": "...", ...},
          "user_properties": [["metrics", {...}], ...]
        },
        ...
      ]
    }

Each test records the ``metrics`` property twice: a minimal payload before
running the subprocess and a full payload afterwards. We take the last one
so we always see the fullest view the test managed to assemble.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import geometric_mean
from typing import Any, Iterable

from tabulate import tabulate


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_metrics(test_entry: dict) -> dict | None:
    """Return the last ``metrics`` entry from a test's user_properties.

    pytest-json-report serialises user_properties in either of two shapes
    depending on version:

    * list of ``[name, value]`` pairs
    * list of ``{name: value}`` dicts  (≥ 1.5)

    We accept both so the report survives plugin upgrades.
    """
    for prop in reversed(test_entry.get('user_properties', [])):
        if isinstance(prop, dict):
            if 'metrics' in prop:
                return prop['metrics']
        elif isinstance(prop, (list, tuple)) and len(prop) == 2:
            name, value = prop
            if name == 'metrics':
                return value
    return None


def load_pytest_report(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def build_summary(pytest_report: dict, *, extra_meta: dict | None = None
                  ) -> dict[str, Any]:
    """Normalise pytest-json-report output into the daily summary schema.

    ``extra_meta`` is merged into the top-level ``meta`` block. Callers
    (run.py) pass the run-level metadata that the pytest plugin itself
    doesn't know about — OV version, machine hostname, workweek, purpose.
    """
    summary_block = pytest_report.get('summary', {})
    out = {
        'generated_at': pytest_report.get('created', 0.0),
        'duration_sec': pytest_report.get('duration', 0.0),
        'meta':         dict(extra_meta or {}),
        'totals': {
            'passed':  summary_block.get('passed', 0),
            'failed':  summary_block.get('failed', 0),
            'error':   summary_block.get('error', 0),
            'skipped': summary_block.get('skipped', 0),
            'total':   summary_block.get('total', 0),
        },
        'tests': [],
    }
    for entry in pytest_report.get('tests', []):
        call = entry.get('call', {}) or {}
        out['tests'].append({
            'nodeid':       entry.get('nodeid', ''),
            'outcome':      entry.get('outcome', 'unknown'),
            'duration_sec': entry.get('duration', 0.0),
            'failure':      call.get('longrepr'),
            'metrics':      _extract_metrics(entry) or {},
        })
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _filter(tests: list[dict], test_type: str) -> list[dict]:
    return [t for t in tests if t.get('metrics', {}).get('test_type') == test_type]


def _gm(values: Iterable[float]) -> str:
    values = list(values)
    return f'{geometric_mean(values):.2f}' if values else '-'


def _render_summary_header(summary: dict) -> str:
    t = summary['totals']
    return tabulate([
        ['Passed',       t['passed']],
        ['Failed',       t['failed']],
        ['Error',        t['error']],
        ['Skipped',      t['skipped']],
        ['Total',        t['total']],
        ['Duration (s)', f"{summary['duration_sec']:.1f}"],
    ], tablefmt='github', headers=['Metric', 'Value']) + '\n'


def _render_failures(tests: list[dict]) -> str:
    failed = [t for t in tests if t['outcome'] not in ('passed', 'skipped')]
    if not failed:
        return ''
    rows = [[t['nodeid'], t['outcome']] for t in failed]
    return '[ Failures ]\n' + tabulate(
        rows, headers=['nodeid', 'outcome'], tablefmt='github'
    ) + '\n'


# ---------------------------------------------------------------------------
# Per-test-type tables
# ---------------------------------------------------------------------------

def _render_llm_benchmark(tests: list[dict]) -> str:
    subset = _filter(tests, 'llm_benchmark')
    if not subset:
        return ''

    rows = []
    first_vals, second_vals = [], []
    for t in subset:
        m = t['metrics']
        if t['outcome'] != 'passed':
            rows.append([m.get('model', ''), m.get('precision', ''),
                         '', '', 'FAIL', 'FAIL'])
            continue
        for d in m.get('data', []):
            perf = d.get('perf', [])
            first  = f'{perf[0]:.2f}' if len(perf) > 0 else ''
            second = f'{perf[1]:.2f}' if len(perf) > 1 else ''
            if len(perf) > 0:
                first_vals.append(perf[0])
            if len(perf) > 1:
                second_vals.append(perf[1])
            rows.append([m['model'], m['precision'],
                         d.get('in_token', ''), d.get('out_token', ''),
                         first, second])

    rows.append(['', '', '', '', '-', '-'])
    rows.append(['geomean (1st)', '', '', '', _gm(first_vals), ''])
    rows.append(['geomean (2nd)', '', '', '', '', _gm(second_vals)])

    return '[RESULT] llm_benchmark\n' + tabulate(
        rows,
        headers=['model', 'precision', 'in token', 'out token',
                 '1st inf (ms)', '2nd inf (ms)'],
        tablefmt='github', stralign='right',
    ) + '\n'


def _render_benchmark_app(tests: list[dict]) -> str:
    subset = _filter(tests, 'benchmark_app')
    if not subset:
        return ''

    rows = []
    for t in subset:
        m = t['metrics']
        if t['outcome'] != 'passed':
            rows.append([m.get('model', ''), m.get('precision', ''),
                         m.get('batch', ''), 'FAIL'])
            continue
        for d in m.get('data', []):
            perf = d.get('perf', [])
            fps = f'{perf[0]:.2f}' if perf else ''
            rows.append([m['model'], m['precision'], m.get('batch', ''), fps])

    return '[RESULT] benchmark_app\n' + tabulate(
        rows,
        headers=['model', 'precision', 'batch', 'throughput (FPS)'],
        tablefmt='github', stralign='right',
    ) + '\n'


def _render_sd_genai(tests: list[dict]) -> str:
    subset = _filter(tests, 'sd_genai')
    if not subset:
        return ''

    rows = []
    for t in subset:
        m = t['metrics']
        if t['outcome'] != 'passed':
            rows.append([m.get('model', ''), m.get('precision', ''),
                         'FAIL', '', '', '', '', '', ''])
            continue
        for d in m.get('data', []):
            size = ''
            if d.get('width') and d.get('height'):
                size = f"{d['width']}x{d['height']}"
            rows.append([
                m['model'], m['precision'],
                f"{d.get('generation_time_sec', 0):.2f}" if d.get('generation_time_sec') is not None else '',
                d.get('batch_size', ''), d.get('steps', ''), size,
                d.get('input_token_size', ''), d.get('output_token_size', ''),
                d.get('infer_count', ''),
            ])

    return '[RESULT] stable_diffusion_genai\n' + tabulate(
        rows,
        headers=['model', 'precision', 'pipeline (s)', 'batch', 'steps',
                 'size', 'in tokens', 'out tokens', 'infer count'],
        tablefmt='github', stralign='right',
    ) + '\n'


def _render_sd_dgfx(tests: list[dict]) -> str:
    subset = _filter(tests, 'sd_dgfx')
    if not subset:
        return ''

    rows = []
    for t in subset:
        m = t['metrics']
        if t['outcome'] != 'passed':
            rows.append([m.get('model', ''), m.get('precision', ''),
                         'FAIL', '', '', ''])
            continue
        for d in m.get('data', []):
            rows.append([
                m['model'], m['precision'],
                f"{d.get('pipeline_sec', 0):.2f}" if d.get('pipeline_sec') is not None else '',
                d.get('batch_size', ''), d.get('steps', ''), d.get('size', ''),
            ])

    return '[RESULT] stable_diffusion_DGfx_E2E_AI\n' + tabulate(
        rows,
        headers=['model', 'precision', 'pipeline (s)', 'batch', 'steps', 'size'],
        tablefmt='github', stralign='right',
    ) + '\n'


def _render_qwen_usage(tests: list[dict]) -> str:
    subset = _filter(tests, 'qwen_usage')
    if not subset:
        return ''

    rows = []
    idx = 0
    for t in subset:
        m = t['metrics']
        if t['outcome'] != 'passed':
            rows.append([idx, '', '', 'FAIL', 'FAIL', '', '', ''])
            idx += 1
            continue
        for d in m.get('data', []):
            perf = d.get('perf', [])
            rows.append([
                idx,
                d.get('in_token', ''), d.get('out_token', ''),
                f'{perf[0]:.2f}' if len(perf) > 0 else '',
                f'{perf[1]:.2f}' if len(perf) > 1 else '',
                f"{m.get('peak_cpu_percent', 0):.2f}",
                m.get('peak_mem_size', ''),
                f"{m.get('peak_mem_percent', 0):.2f}",
            ])
            idx += 1

    return '[RESULT] measured_usage(cpp)\n' + tabulate(
        rows,
        headers=['index', 'in token', 'out token', '1st inf (ms)',
                 '2nd inf (ms)', 'CPU (%)', 'Memory', 'Memory (%)'],
        tablefmt='github', stralign='right',
    ) + '\n'


def _render_whisper_base(tests: list[dict]) -> str:
    subset = _filter(tests, 'whisper_base')
    if not subset:
        return ''

    rows = []
    for t in subset:
        m = t['metrics']
        if t['outcome'] != 'passed':
            rows.append([m.get('model', ''), m.get('precision', ''), 'FAIL'])
            continue
        for d in m.get('data', []):
            perf = d.get('perf', [])
            tps = f'{perf[0]:.2f}' if perf else ''
            rows.append([m['model'], m.get('precision', ''), tps])

    return '[RESULT] whisper_base\n' + tabulate(
        rows,
        headers=['model', 'precision', 'tps'],
        tablefmt='github', stralign='right',
    ) + '\n'


def _render_chat_sample(tests: list[dict]) -> str:
    subset = _filter(tests, 'chat_sample')
    if not subset:
        return ''

    parts = ['[RESULT] chat_sample']
    for t in subset:
        m = t['metrics']
        parts.append(f"--- {m.get('model','')} / {m.get('precision','')}"
                     f" (rc={m.get('returncode', '?')}) ---")
        parts.append(m.get('output', '').rstrip())
    return '\n'.join(parts) + '\n'


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

# Ordered so the report reads the same as the old one: summary, failures,
# then a block per test type.
_SECTION_RENDERERS = [
    _render_llm_benchmark,
    _render_benchmark_app,
    _render_sd_genai,
    _render_sd_dgfx,
    _render_qwen_usage,
    _render_whisper_base,
    _render_chat_sample,
]


def render_text_report(summary: dict) -> str:
    sections: list[str] = ['[ Summary ]', _render_summary_header(summary)]
    failures = _render_failures(summary['tests'])
    if failures:
        sections.append(failures)
    for render in _SECTION_RENDERERS:
        block = render(summary['tests'])
        if block:
            sections.append(block)
    return '\n'.join(s for s in sections if s).strip() + '\n'


def build_reports(pytest_json_path: Path, *, text_out: Path, summary_out: Path,
                  extra_meta: dict | None = None) -> dict:
    """Read pytest-json-report output, write text + normalised JSON reports.

    Returns the summary dict so callers can use it for mail titles etc.
    """
    pytest_report = load_pytest_report(pytest_json_path)
    summary = build_summary(pytest_report, extra_meta=extra_meta)

    summary_out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    text_out.write_text(render_text_report(summary), encoding='utf-8')
    return summary
