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
                    'first_latency': 200.0,
                    'second_avg_latency': 20.0,
                },
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_latency': 100.0,
                    'second_avg_latency': 10.0,
                },
                {
                    'iteration': 3,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_latency': 300.0,
                    'second_avg_latency': 30.0,
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


def test_parse_json_report_reports_token_latency_not_infer_latency(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps({
        'perfdata': {
            'results': [
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 4937,
                    'infer_count': 256,
                    'first_latency': 1354.15,
                    'second_avg_latency': 32.04,
                    'first_infer_latency': 157.4,
                    'second_infer_avg_latency': 32.04,
                },
            ],
        },
    }), encoding='utf-8')

    item = parse_json_report(report_path)[0]
    assert item['perf'] == [1354.15, 32.04]
    assert item['infer_perf'] == [157.4, 32.04]


def test_parse_json_report_carries_mm_embeddings_time(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps({
        'perfdata': {
            'results': [
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 4937,
                    'infer_count': 256,
                    'first_latency': 1354.15,
                    'second_avg_latency': 32.04,
                    'first_infer_latency': 157.4,
                    'second_infer_avg_latency': 32.04,
                    'mm_embeddings_preparation_time': 1148.51,
                },
            ],
        },
    }), encoding='utf-8')

    assert parse_json_report(report_path)[0]['mm_embeddings_time'] == 1148.51


def test_parse_json_report_omits_mm_embeddings_time_for_text_models(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps({
        'perfdata': {
            'results': [
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_latency': 100.0,
                    'second_avg_latency': 10.0,
                },
            ],
        },
    }), encoding='utf-8')

    assert 'mm_embeddings_time' not in parse_json_report(report_path)[0]


def test_parse_json_report_skips_sentinel_latencies(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps({
        'perfdata': {
            'results': [
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_latency': 100.0,
                    'second_avg_latency': 10.0,
                    'first_infer_latency': -1,
                    'second_infer_avg_latency': '',
                },
            ],
        },
    }), encoding='utf-8')

    item = parse_json_report(report_path)[0]
    assert item['perf'] == [100.0, 10.0]
    assert 'infer_perf' not in item


def test_parse_json_report_carries_timestamps_of_selected_iteration(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps({
        'perfdata': {
            'results': [
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_latency': 100.0,
                    'second_avg_latency': 10.0,
                    'start': '2026-08-26T06:05:03.100000+00:00',
                    'end': '2026-08-26T06:05:06.200000+00:00',
                    'token_timestamps': {
                        'first_token_begin': '2026-08-26T06:05:03.200000+00:00',
                        'first_token_end': '2026-08-26T06:05:03.300000+00:00',
                        'second_token_begin': '2026-08-26T06:05:03.300000+00:00',
                        'second_token_end': '2026-08-26T06:05:03.310000+00:00',
                    },
                },
                {
                    'iteration': 2,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_latency': 200.0,
                    'second_avg_latency': 20.0,
                    'start': '2026-08-26T06:05:07.000000+00:00',
                    'end': '2026-08-26T06:05:10.000000+00:00',
                    'token_timestamps': {
                        'first_token_begin': '2026-08-26T06:05:07.100000+00:00',
                        'first_token_end': '2026-08-26T06:05:07.300000+00:00',
                    },
                },
            ],
        },
    }), encoding='utf-8')

    item = parse_json_report(report_path)[0]
    assert item['start'] == '2026-08-26T06:05:03.100000+00:00'
    assert item['end'] == '2026-08-26T06:05:06.200000+00:00'
    assert item['token_timestamps']['first_token_end'] == '2026-08-26T06:05:03.300000+00:00'


def test_parse_json_report_omits_timestamps_when_absent(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps({
        'perfdata': {
            'results': [
                {
                    'iteration': 1,
                    'prompt_idx': 0,
                    'input_size': 10,
                    'infer_count': 20,
                    'first_latency': 100.0,
                    'second_avg_latency': 10.0,
                },
            ],
        },
    }), encoding='utf-8')

    item = parse_json_report(report_path)[0]
    assert 'start' not in item
    assert 'end' not in item
    assert 'token_timestamps' not in item
