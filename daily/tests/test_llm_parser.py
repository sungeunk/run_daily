from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsers.llm_benchmark import parse_json_report
from common.output_capture import generated_texts, generation_fingerprint

pytestmark = pytest.mark.dev_only


def test_generated_outputs_keep_newlines_iterations_and_eof() -> None:
    records = generated_texts(
        "[ INFO ] [warm-up][P0] Generated: First\nsecond line\n"
        "[ INFO ] [1][P0] Generated: Actual\nlast line"
    )
    assert records == [
        {"prompt_idx": 0, "iteration": "warm-up", "generated_text": "First\nsecond line", "truncated": False},
        {"prompt_idx": 0, "iteration": "1", "generated_text": "Actual\nlast line", "truncated": False},
    ]


def test_generation_fingerprint_tracks_inputs(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.jsonl"
    script = tmp_path / "bench.py"
    prompt.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    script.write_text("pass\n", encoding="utf-8")
    original = generation_fingerprint(str(prompt), str(script), {"seed": 42})
    assert original
    assert original == generation_fingerprint(str(prompt), str(script), {"seed": 42})
    assert original != generation_fingerprint(str(prompt), str(script), {"seed": 43})
    prompt.write_text('{"prompt": "changed"}\n', encoding="utf-8")
    assert original != generation_fingerprint(str(prompt), str(script), {"seed": 42})
    prompt.write_text('{"prompt": "hello", "image": "unknown.png"}\n', encoding="utf-8")
    assert generation_fingerprint(str(prompt), str(script), {}) is None
    assert generation_fingerprint("missing.jsonl", str(script), {}) is None


@pytest.mark.parametrize("text", ["Normal generated answer.", "", "!!!!!!!!!!!!"])
def test_parse_json_report_preserves_selected_text(tmp_path: Path, text: str) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"perfdata": {"results": [
        {"iteration": 1, "prompt_idx": 0, "first_latency": 10.0, "generated_text": text},
        {"iteration": 2, "prompt_idx": 0, "first_latency": 20.0, "generated_text": "other"},
    ]}}), encoding="utf-8")
    assert parse_json_report(report_path)[0]["generated_text"] == text


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
                        'generate_begin': '2026-08-26T06:05:03.200000+00:00',
                        'first_token_end': '2026-08-26T06:05:03.300000+00:00',
                        'generate_end': '2026-08-26T06:05:06.100000+00:00',
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
                        'generate_begin': '2026-08-26T06:05:07.100000+00:00',
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
