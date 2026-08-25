#!/usr/bin/env python3
"""Report builder: consumes the pytest-json-report output and emits a
normalised JSON summary.

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
from typing import Any


SUMMARY_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_metrics(test_entry: dict) -> dict | None:
    """Return the last ``metrics`` entry from a test's user_properties.

    pytest-json-report serialises user_properties in either of two shapes
    depending on version:

    * list of ``[name, value]`` pairs
    * list of ``{name: value}`` dicts  (â‰¥ 1.5)

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
    doesn't know about â€” OV version, machine hostname, workweek, purpose.
    """
    summary_block = pytest_report.get('summary', {})
    out = {
        'schema_version': SUMMARY_SCHEMA_VERSION,
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
# Public entry
# ---------------------------------------------------------------------------


def build_reports(pytest_json_path: Path, *, summary_out: Path,
                  extra_meta: dict | None = None) -> dict:
    """Read pytest-json-report output and write the normalised JSON summary.

    Returns the summary dict so callers can use it for mail titles etc.
    """
    pytest_report = load_pytest_report(pytest_json_path)
    summary = build_summary(pytest_report, extra_meta=extra_meta)

    summary_out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
