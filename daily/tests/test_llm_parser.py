from __future__ import annotations

import json
from pathlib import Path

from parsers.llm_benchmark import parse_json_report


def test_parse_json_report_uses_fastest_iteration(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps({
        'perfdata': {
            'results': [
                {
                    'iteration': 2,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_infer_latency': 200.0,
                    'second_infer_avg_latency': 20.0,
                },
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_infer_latency': 100.0,
                    'second_infer_avg_latency': 10.0,
                },
                {
                    'iteration': 3,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_infer_latency': 300.0,
                    'second_infer_avg_latency': 30.0,
                },
            ],
        },
    }), encoding='utf-8')

    assert parse_json_report(report_path) == [{
        'prompt_idx': 0,
        'in_token': 10,
        'out_token': 20,
        'perf': [100.0, 10.0],
    }]