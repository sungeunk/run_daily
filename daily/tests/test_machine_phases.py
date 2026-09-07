"""Tests for phase-scoped machine telemetry: windowing, extraction and ingest."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.dev_only

DAILY_DIR = Path(__file__).resolve().parent.parent
if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

import duckdb  # noqa: E402

from common.machine_monitor import summarize_window  # noqa: E402
from parsers.llm_benchmark import phase_windows  # noqa: E402
from viewer.ingest import writer  # noqa: E402
from viewer.ingest.loader_new import _phase_rows  # noqa: E402
from viewer.ingest.record import RunRecord  # noqa: E402

BASE = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _write_samples(path: Path, count: int, *, step_sec: float = 0.5,
                   clock_mhz: float = 2000.0) -> None:
    lines = []
    for i in range(count):
        ts = BASE + timedelta(seconds=i * step_sec)
        lines.append(json.dumps({
            'timestamp_utc': ts.strftime('%Y-%m-%dT%H:%M:%S') + f'.{ts.microsecond // 1000:03d}Z',
            't_monotonic': i * step_sec,
            'gpu_clock_mhz': clock_mhz + i,
            'gpu_utilization_percent': 90.0,
            'gpu_power_watts': 30.0,
            'gpu_throttle_reasons': '0x0',
            'sample_duration_ms': 5.0,
        }))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_window_selects_only_samples_inside_it(tmp_path: Path) -> None:
    jsonl = tmp_path / 'monitor.jsonl'
    _write_samples(jsonl, 20)

    window = summarize_window(jsonl, BASE.isoformat(),
                              (BASE + timedelta(seconds=2)).isoformat())

    assert window['samples'] == 5  # 0.0 .. 2.0 inclusive at 0.5 s spacing
    assert window['window_sec'] == 2.0
    assert window['gpu_clock_mhz']['min'] == 2000.0
    assert window['gpu_clock_mhz']['max'] == 2004.0


def test_window_shorter_than_sampling_interval_reports_zero_samples(tmp_path: Path) -> None:
    jsonl = tmp_path / 'monitor.jsonl'
    _write_samples(jsonl, 20)

    window = summarize_window(jsonl, (BASE + timedelta(seconds=0.6)).isoformat(),
                              (BASE + timedelta(seconds=0.9)).isoformat())

    assert window == {'samples': 0, 'window_sec': 0.3}


def test_window_is_empty_for_missing_or_inverted_bounds(tmp_path: Path) -> None:
    jsonl = tmp_path / 'monitor.jsonl'
    _write_samples(jsonl, 4)

    assert summarize_window(jsonl, None, BASE.isoformat()) == {}
    assert summarize_window(jsonl, BASE.isoformat(), None) == {}
    assert summarize_window(jsonl, (BASE + timedelta(seconds=5)).isoformat(),
                            BASE.isoformat()) == {}
    assert summarize_window(tmp_path / 'missing.jsonl', BASE.isoformat(),
                            (BASE + timedelta(seconds=1)).isoformat()) == {}


def test_phase_rows_map_every_phase_onto_an_exec_mode() -> None:
    def window(**over):
        return {'samples': 6, 'window_sec': 3.0,
                'gpu_clock_mhz': {'min': 900.0, 'max': 1100.0, 'mean': 1000.0},
                'gpu_throttled_sample_ratio': 0.0,
                'gpu_throttle_reasons_seen': 'N/A'} | over

    metrics = {
        'model': 'llama',
        'precision': 'INT4',
        'machine_phases': [
            dict(window(samples=40, window_sec=20.0), phase='compile',
                 in_token=0, out_token=0),
            dict(window(), phase='warmup', in_token=0, out_token=0),
            dict(window(), phase='idle', in_token=32, out_token=128),
            dict(window(gpu_throttled_sample_ratio=0.5,
                        gpu_throttle_reasons_seen=['0x4', '0x8']),
                 phase='first_token', in_token=32, out_token=128),
            dict(window(samples=12,
                        gpu_clock_mhz={'min': 1900.0, 'max': 2100.0, 'mean': 2000.0}),
                 phase='decode', in_token=32, out_token=128),
        ],
    }

    rows = {row.exec_mode: row for row in _phase_rows(metrics)}

    assert set(rows) == {'compile', 'warmup', 'idle', '1st', '2nd'}
    assert rows['1st'].throttled_sample_ratio == 0.5
    assert rows['1st'].throttle_reasons == '0x4,0x8'
    assert rows['2nd'].gpu_clock_mhz_mean == 2000.0
    assert rows['2nd'].samples == 12
    # compile and warm-up precede every prompt, so they belong to no series.
    for phase in ('compile', 'warmup'):
        assert (rows[phase].in_token, rows[phase].out_token) == (0, 0)
    assert (rows['idle'].in_token, rows['idle'].out_token) == (32, 128)


def test_phase_rows_absent_on_reports_without_machine_phases() -> None:
    metrics = {'model': 'llama', 'precision': 'INT4',
               'data': [{'in_token': 32, 'out_token': 128}]}
    assert list(_phase_rows(metrics)) == []


def test_phase_rows_skip_unknown_phase_names() -> None:
    metrics = {'model': 'llama', 'precision': 'INT4',
               'machine_phases': [{'phase': 'detokenize', 'samples': 1}]}
    assert list(_phase_rows(metrics)) == []


def test_phase_stats_are_ingested_and_join_perf_one_to_one(tmp_path: Path) -> None:
    from viewer.ingest.record import PerfRow, PhaseStatRow

    rec = RunRecord(run_id='run-1', source_format='new',
                    report_file='daily.1.summary.json', machine='TEST-01',
                    ts=datetime(2026, 1, 1, 12, 0))
    rec.perf.append(PerfRow('llama', 'INT4', 32, 128, '1st', 100.0, 'ms'))
    rec.perf.append(PerfRow('llama', 'INT4', 32, 128, '2nd', 10.0, 'ms'))
    rec.phase_stats.append(PhaseStatRow('llama', 'INT4', 32, 128, '1st',
                                        window_sec=1.0, samples=2,
                                        gpu_clock_mhz_mean=1000.0))
    rec.phase_stats.append(PhaseStatRow('llama', 'INT4', 32, 128, '2nd',
                                        window_sec=6.0, samples=12,
                                        gpu_clock_mhz_mean=2000.0))

    db = tmp_path / 'test.duckdb'
    con = writer.connect(db)
    writer.ensure_schema(con)
    writer.upsert_run(con, rec)
    con.close()

    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute("""
            SELECT p.exec_mode, p.value, s.gpu_clock_mhz_mean, s.samples
            FROM perf p
            JOIN perf_phase_stats s
              USING (run_id, model, precision, in_token, out_token, exec_mode)
            ORDER BY p.exec_mode
        """).fetchall()

    assert rows == [('1st', 100.0, 1000.0, 2), ('2nd', 10.0, 2000.0, 12)]


def _report(compile_end: float, rows: list[dict]) -> dict:
    return {'perfdata': {
        'compile_window': {'begin': _at(0), 'end': _at(compile_end)},
        'results': rows,
    }}


def _at(seconds: float) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def _result(iteration: int, prompt_idx: int, start: float, end: float,
            *, split: float | None = None) -> dict:
    row = {'iteration': iteration, 'prompt_idx': prompt_idx,
           'start': _at(start), 'end': _at(end)}
    if split is not None:
        row['token_timestamps'] = {
            'generate_begin': _at(start + 0.2),
            'first_token_end': _at(split),
            'generate_end': _at(end - 0.2),
        }
    return row


def test_phase_windows_name_every_stretch_of_a_run() -> None:
    warm0 = _result(0, 0, 10, 14)
    warm1 = _result(0, 1, 14, 18)
    run0 = _result(1, 0, 20, 26, split=21)
    report = _report(8, [warm0, warm1, run0])
    items = [{'in_token': 32, 'out_token': 128, 'start': run0['start'],
              'token_timestamps': run0['token_timestamps']}]

    windows = {w.phase: w for w in phase_windows(report, items)}

    assert set(windows) == {'compile', 'warmup', 'idle', 'first_token', 'decode'}
    assert (windows['compile'].begin, windows['compile'].end) == (_at(0), _at(8))
    # Warm-up spans both prompts of iteration 0.
    assert (windows['warmup'].begin, windows['warmup'].end) == (_at(10), _at(18))
    # Idle runs from the previous result's end to this prompt's generate.
    assert (windows['idle'].begin, windows['idle'].end) == (_at(18), _at(20.2))
    assert (windows['first_token'].begin, windows['first_token'].end) == (_at(20.2), _at(21))
    assert (windows['decode'].begin, windows['decode'].end) == (_at(21), _at(25.8))
    assert windows['first_token'].in_token == 32
    assert (windows['warmup'].in_token, windows['warmup'].out_token) == (0, 0)


def test_phase_windows_skip_the_idle_before_the_very_first_result() -> None:
    first = _result(0, 0, 10, 16, split=11)
    report = _report(8, [first])
    items = [{'in_token': 32, 'out_token': 128, 'start': first['start'],
              'token_timestamps': first['token_timestamps']}]

    phases = [w.phase for w in phase_windows(report, items)]

    assert 'idle' not in phases
    assert phases == ['compile', 'warmup', 'first_token', 'decode']


def test_phase_windows_keep_the_generate_whole_when_the_split_is_missing() -> None:
    row = _result(0, 0, 10, 16, split=11)
    del row['token_timestamps']['first_token_end']
    report = _report(8, [row])
    items = [{'in_token': 32, 'out_token': 128, 'start': row['start'],
              'token_timestamps': row['token_timestamps']}]

    windows = {w.phase: w for w in phase_windows(report, items)}

    assert 'first_token' not in windows and 'decode' not in windows
    assert (windows['generate'].begin, windows['generate'].end) == (_at(10.2), _at(15.8))


def test_phase_windows_are_empty_without_a_report() -> None:
    assert phase_windows({}, []) == []
