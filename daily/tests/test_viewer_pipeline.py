from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from common import delivery
import run
from viewer import app
from viewer import queries
from viewer.ingest._common import (file_hash, parse_stamp_from_name, run_id_of,
                                   split_ov_version, workweek_of)
from viewer.ingest import cli
from viewer.ingest.loader_new import load_summary
from viewer.ingest.loader_old import load_report
from viewer.ingest.record import PerfRow, RunRecord
from viewer.ingest.writer import connect, ensure_schema, load_display_profile, upsert_run
from analysis.report import render_analysis_summary
from analysis.types import (
    AnalysisResult,
    BaselineInfo,
    ComparisonRow,
    FunctionalResult,
    ModelSummary,
    PerformanceResult,
    SeriesKey,
)


class TestBenchmark:
    pass


class TestStableDiffusion:
    pass


def test_common_helpers_parse_versions_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("payload", encoding="utf-8")

    stamp = parse_stamp_from_name("daily.20260421_903.summary.json")

    assert stamp == datetime(2026, 4, 21, 9, 3)
    assert workweek_of(stamp) == "2026.WW17.2"
    assert split_ov_version("2026.2.0-21664-ad5d8e0f99b") == (
        "21664",
        "ad5d8e0f99b",
    )
    assert len(file_hash(path)) == 24
    assert run_id_of("LNL-03", stamp, "daily.report") == run_id_of(
        "LNL-03",
        stamp,
        "daily.report",
    )


def test_load_summary_extracts_metadata_artifacts_and_perf(tmp_path: Path) -> None:
    summary_path = tmp_path / "daily.20260421_2234.summary.json"
    raw_path = tmp_path / "daily.20260421_2234.2026.2.0-21664-ad5d8e0f99b.raw"
    raw_path.write_text("raw log", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": 1776810840.0,
                "duration_sec": 12.5,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260421_2234",
                    "workweek": "2026.WW17.2",
                    "ov_version": "2026.2.0-21664-ad5d8e0f99b",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [
                                {"in_token": 32, "out_token": 128, "perf": [11.0, 2.5]},
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rec = load_summary(summary_path)

    assert rec.source_format == "new"
    assert rec.machine == "LNL-03"
    assert rec.rawlog_path == str(raw_path)
    assert rec.ov_build == "21664"
    assert rec.ov_sha == "ad5d8e0f99b"
    assert [(p.exec_mode, p.value, p.unit) for p in rec.perf] == [
        ("1st", 11.0, "ms"),
        ("2nd", 2.5, "ms"),
    ]


def test_load_report_normalizes_legacy_sd_units(tmp_path: Path) -> None:
    run_dir = tmp_path / "LNL-03"
    run_dir.mkdir()
    pickle_path = run_dir / "daily.20260421_2234.2026.2.0-21664-ad5d8e0f99b.pickle"
    report_path = pickle_path.with_suffix(".report")
    raw_path = pickle_path.with_suffix(".raw")
    report_path.write_text(
        "| Purpose | daily_CB timer |\nOpenVINO 2026.2.0-21664-ad5d8e0f99b\n",
        encoding="utf-8",
    )
    raw_path.write_text("raw log", encoding="utf-8")
    payload = {
        ("stable-diffusion-xl", "FP16", TestStableDiffusion): [
            {
                "return_code": 0,
                "data_list": [
                    {"in_token": 0, "out_token": 0, "perf": [8000.0]},
                ],
            }
        ],
        ("llama", "FP16", TestBenchmark): [
            {
                "return_code": 0,
                "data_list": [
                    {"in_token": 32, "out_token": 128, "perf": [10.0, 2.0]},
                ],
            }
        ],
    }
    pickle_path.write_bytes(pickle.dumps(payload))

    rec = load_report(pickle_path)

    assert rec.source_format == "old"
    assert rec.machine == "LNL-03"
    assert rec.purpose == "daily_CB timer"
    assert rec.rawlog_path == str(raw_path)
    sd_rows = [p for p in rec.perf if p.model == "stable-diffusion-xl"]
    assert [(p.exec_mode, p.value, p.unit) for p in sd_rows] == [("pipeline", 8.0, "s")]

    overridden = load_report(pickle_path, machine_override="BMG-02")
    assert overridden.machine == "BMG-02"


def test_ingest_cli_loads_default_profile_additively(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    root = tmp_path / "empty"
    root.mkdir()
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
profile: default
rows:
  - model: llama
    precision: FP16
    in_spec: short
    out_spec: long
    exec_mode: 2nd
    label: first
""".strip(),
        encoding="utf-8",
    )

    assert cli.main(["--root", str(root), "--db", str(db_path), "--profile", str(profile_path)]) == 0
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace("label: first", "label: changed"),
        encoding="utf-8",
    )
    assert cli.main(["--root", str(root), "--db", str(db_path)]) == 0
    with connect(db_path, read_only=True) as con:
        label = con.execute("SELECT label FROM display_rows WHERE profile = 'default'").fetchone()[0]
    assert label == "first"

    assert cli.main(["--root", str(root), "--db", str(db_path), "--profile", str(profile_path)]) == 0
    with connect(db_path, read_only=True) as con:
        label = con.execute("SELECT label FROM display_rows WHERE profile = 'default'").fetchone()[0]
    assert label == "changed"


def test_writer_and_queries_cover_profile_matching_and_regressions(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    profile_path = tmp_path / "default.yaml"
    profile_path.write_text(
        """
profile: default
rows:
  - model: llama
    precision: FP16
    in_spec: short
    out_spec: long
    exec_mode: 2nd
    label: llama second token
""".strip(),
        encoding="utf-8",
    )
    con = queries.duckdb.connect(str(db_path))
    ensure_schema(con)
    assert load_display_profile(con, profile_path) == 1

    now = datetime.now().replace(microsecond=0)
    baseline_values = [10.0, 10.0, 10.0, 10.0, 10.0]
    recent_values = [12.0, 12.0, 12.0, 12.0, 12.0]
    offsets = [20, 19, 18, 17, 16, 5, 4, 3, 2, 1]
    for idx, (value, days_ago) in enumerate(zip(baseline_values + recent_values, offsets)):
        ts = now - timedelta(days=days_ago)
        rec = RunRecord(
            run_id=f"run-{idx}",
            source_format="new",
            report_file=f"daily.{idx}.summary.json",
            machine="LNL-03",
            ts=ts,
            purpose="daily_CB timer",
            description="daily_CB timer",
            ww=workweek_of(ts),
            ov_version="2026.2.0-21664-ad5d8e0f99b",
            source_path=str(tmp_path / f"daily.{idx}.summary.json"),
            file_hash=f"hash-{idx}",
            perf=[PerfRow("llama", "FP16", 32, 128, "2nd", value, "ms")],
        )
        upsert_run(con, rec)
    con.close()

    runs = queries.list_runs(db_path, "LNL-03")
    assert "source_path" in runs.columns

    matrix = queries.build_excel_matrix(db_path, ["run-0", "run-9"], "default")
    assert matrix.loc[0, "model"] == "llama"
    assert matrix.loc[0, "exec_mode"] == "2nd"
    assert queries.extra_rows(db_path, ["run-0"], "default").empty

    regressions = queries.trend_regressions(
        db_path,
        "LNL-03",
        recent_days=7,
        baseline_days=21,
        min_recent_points=3,
        min_baseline_points=3,
        purpose_filter="daily_CB timer",
    )

    row = regressions.iloc[0]
    assert row["status"] == "ok"
    assert row["worsening_pct"] == 0.2


def test_functional_queries_with_machine_filter_and_missing_category(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    with connect(db_path) as con:
        ensure_schema(con)

        ts_a = datetime(2026, 5, 9, 10, 0, 0)
        ts_b = datetime(2026, 5, 9, 11, 0, 0)

        rec_a = RunRecord(
            run_id="run-a",
            source_format="new",
            report_file="daily.a.summary.json",
            machine="LNL-03",
            ts=ts_a,
            purpose="daily_CB timer",
            description="daily",
            ww=workweek_of(ts_a),
            ov_version="2026.2.0-21664-ad5d8e0f99b",
            source_path=str(tmp_path / "daily.a.summary.json"),
            file_hash="hash-a",
            perf=[PerfRow("llama", "FP16", 32, 128, "2nd", 10.0, "ms")],
        )
        rec_b = RunRecord(
            run_id="run-b",
            source_format="new",
            report_file="daily.b.summary.json",
            machine="BMG-02",
            ts=ts_b,
            purpose="daily_CB timer",
            description="daily",
            ww=workweek_of(ts_b),
            ov_version="2026.2.0-21664-ad5d8e0f99b",
            source_path=str(tmp_path / "daily.b.summary.json"),
            file_hash="hash-b",
            perf=[PerfRow("llama", "FP16", 32, 128, "2nd", 11.0, "ms")],
        )
        upsert_run(con, rec_a)
        upsert_run(con, rec_b)

        con.execute(
            """
            INSERT INTO analysis_results (
                run_id, baseline_run_id, overall_status,
                compared_count, improved_count, same_count,
                regressed_count, functional_fail_count
            )
            VALUES (?, NULL, ?, 1, 0, 1, 0, ?)
            """,
            ["run-a", "red", 1],
        )
        con.execute(
            """
            INSERT INTO analysis_results (
                run_id, baseline_run_id, overall_status,
                compared_count, improved_count, same_count,
                regressed_count, functional_fail_count
            )
            VALUES (?, NULL, ?, 1, 1, 0, 0, ?)
            """,
            ["run-b", "green", 0],
        )

        con.execute(
            "INSERT INTO functional_issues (run_id, nodeid, outcome, message) VALUES (?, ?, ?, ?)",
            ["run-a", "test_case_a", "failed", "assertion"],
        )

    summary = queries.fetch_functional_summary(db_path, machine="LNL-03", days=30)
    assert summary["run_id"].tolist() == ["run-a"]
    assert int(summary.iloc[0]["functional_issue_count"]) == 1

    history = queries.fetch_functional_history(db_path, machine="LNL-03", days=30)
    assert history["run_id"].tolist() == ["run-a"]
    assert "category" in history.columns
    assert pd.isna(history.iloc[0]["category"])


def test_fetch_run_comparison_fallback_handles_unit_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    with connect(db_path) as con:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO perf (
                run_id, model, precision, in_token, out_token, exec_mode, value, unit
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "run-a", "llama", "FP16", 32, 128, "2nd", 10.0, "ms",
                "run-b", "llama", "FP16", 32, 128, "2nd", 100.0, "tps",
            ],
        )

    # No analysis_comparisons rows for this pair -> direct perf fallback path.
    df = queries.fetch_run_comparison(db_path, "run-a", "run-b")
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["improvement_pct"])
    assert df.iloc[0]["verdict"] == "unavailable"


def test_fetch_run_comparison_fallback_uses_coalesced_unit_direction(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    with connect(db_path) as con:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO perf (
                run_id, model, precision, in_token, out_token, exec_mode, value, unit
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                # Run A has NULL unit; Run B has latency unit -> must use latency direction.
                "run-a", "llama", "FP16", 32, 128, "2nd", 9.0, None,
                "run-b", "llama", "FP16", 32, 128, "2nd", 10.0, "ms",
            ],
        )

    df = queries.fetch_run_comparison(db_path, "run-a", "run-b")
    assert len(df) == 1
    # Latency improved from 10 -> 9 => +10%
    assert df.iloc[0]["improvement_pct"] == 0.1
    assert df.iloc[0]["verdict"] == "improved"


def test_fetch_run_comparison_fallback_applies_canonical_thresholds(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    with connect(db_path) as con:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO perf (
                run_id, model, precision, in_token, out_token, exec_mode, value, unit
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                # +4% throughput gain => below default 5% threshold => same
                "run-a", "llama", "FP16", 32, 128, "2nd", 104.0, "tps",
                "run-b", "llama", "FP16", 32, 128, "2nd", 100.0, "tps",
            ],
        )

    df = queries.fetch_run_comparison(db_path, "run-a", "run-b")
    assert len(df) == 1
    assert df.iloc[0]["improvement_pct"] == 0.04
    assert df.iloc[0]["verdict"] == "same"


def test_fetch_analysis_overview_returns_baseline_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    with connect(db_path) as con:
        ensure_schema(con)
        ts_base = datetime(2026, 5, 8, 9, 0, 0)
        ts_cur = datetime(2026, 5, 9, 9, 0, 0)
        base = RunRecord(
            run_id="run-base",
            source_format="new",
            report_file="daily.base.summary.json",
            machine="LNL-03",
            ts=ts_base,
            purpose="daily_CB timer",
            description="daily",
            ww=workweek_of(ts_base),
            ov_version="2026.2.0-11111-aaaaaaaaaaa",
            source_path=str(tmp_path / "daily.base.summary.json"),
            file_hash="hash-base",
            perf=[PerfRow("llama", "FP16", 32, 128, "2nd", 10.0, "ms")],
        )
        cur = RunRecord(
            run_id="run-cur",
            source_format="new",
            report_file="daily.cur.summary.json",
            machine="LNL-03",
            ts=ts_cur,
            purpose="daily_CB timer",
            description="daily",
            ww=workweek_of(ts_cur),
            ov_version="2026.2.0-22222-bbbbbbbbbbb",
            source_path=str(tmp_path / "daily.cur.summary.json"),
            file_hash="hash-cur",
            perf=[PerfRow("llama", "FP16", 32, 128, "2nd", 11.0, "ms")],
        )
        upsert_run(con, base)
        upsert_run(con, cur)

        con.execute(
            """
            INSERT INTO analysis_results (
                run_id, baseline_run_id, overall_status,
                compared_count, improved_count, same_count,
                regressed_count, functional_fail_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ["run-cur", "run-base", "yellow", 10, 1, 7, 2, 0],
        )

    df = queries.fetch_analysis_overview(db_path, "run-cur")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["overall_status"] == "yellow"
    assert row["regressed_count"] == 2
    assert row["functional_issue_count"] == 0
    assert row["baseline_run_id"] == "run-base"
    assert row["baseline_stamp"] == "20260508_0900"


def test_ensure_schema_upgrades_legacy_analysis_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.duckdb"
    with connect(db_path) as con:
        # Simulate older DB that already has analysis tables but misses new columns.
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                run_id TEXT PRIMARY KEY,
                baseline_run_id TEXT,
                overall_status TEXT NOT NULL,
                compared_count INTEGER NOT NULL DEFAULT 0,
                improved_count INTEGER NOT NULL DEFAULT 0,
                same_count INTEGER NOT NULL DEFAULT 0,
                regressed_count INTEGER NOT NULL DEFAULT 0,
                functional_fail_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_comparisons (
                run_id TEXT NOT NULL,
                baseline_run_id TEXT,
                model TEXT NOT NULL,
                precision TEXT NOT NULL,
                in_token INTEGER NOT NULL,
                out_token INTEGER NOT NULL,
                exec_mode TEXT NOT NULL,
                unit TEXT,
                current_value DOUBLE,
                baseline_value DOUBLE,
                improvement_pct DOUBLE,
                verdict TEXT NOT NULL,
                PRIMARY KEY (run_id, model, precision, in_token, out_token, exec_mode)
            )
            """
        )

        ensure_schema(con)

        cols_results = {
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'analysis_results'"
            ).fetchall()
        }
        cols_comparisons = {
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'analysis_comparisons'"
            ).fetchall()
        }

    assert "updated_at" in cols_results
    assert "threshold_pct" in cols_comparisons


def test_dashboard_artifact_helpers_use_source_path_as_anchor(tmp_path: Path) -> None:
    source_path = tmp_path / "daily.20260421_2234.summary.json"
    report_path = tmp_path / "daily.20260421_2234.report"
    pytest_path = tmp_path / "daily.20260421_2234.pytest.json"
    raw_path = tmp_path / "daily.20260421_2234.2026.2.0-21664-ad5d8e0f99b.raw"
    source_path.write_text("{}", encoding="utf-8")
    report_path.write_text("report", encoding="utf-8")
    pytest_path.write_text("{}", encoding="utf-8")
    raw_path.write_text("raw", encoding="utf-8")
    run = pd.Series(
        {
            "source_format": "new",
            "source_path": str(source_path),
            "report_file": "not-a-report-file.summary.json",
            "rawlog_path": str(raw_path),
        }
    )

    assert app._summary_path_for_run(run) == source_path
    assert app._report_path_for_run(run) == report_path
    assert app._pytest_json_path_for_run(run) == pytest_path
    assert app._rawlog_path_for_run(run) == raw_path
    assert app._metric_from_user_properties(
        {"user_properties": [["metrics", {"returncode": 7}]]},
        "returncode",
    ) == 7


def test_render_analysis_summary_includes_top_regressions() -> None:
    key = SeriesKey("llama", "FP16", 32, 128, "2nd")
    row = ComparisonRow(
        key=key,
        unit="ms",
        current_value=12.0,
        baseline_value=10.0,
        improvement_pct=-0.2,
        verdict="regressed",
    )
    result = AnalysisResult(
        overall_status="yellow",
        baseline=BaselineInfo(status="found", run_id="r0", stamp="20260501_0100"),
        functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=1, improved=0, same=0, regressed=1, unavailable=0),
        models=[ModelSummary(model="llama", avg_improvement_pct=-0.2, improved=0, same=0, regressed=1)],
        top_regressions=[row],
        rows=[row],
    )

    section = render_analysis_summary(result)

    assert "[ Analysis summary ]" in section
    assert "Top regressions:" in section
    assert "llama | FP16" in section


def test_render_analysis_summary_without_baseline() -> None:
    result = AnalysisResult(
        overall_status="gray",
        baseline=BaselineInfo(status="not_found"),
        functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[],
        top_regressions=[],
        rows=[],
    )

    section = render_analysis_summary(result)
    assert "Baseline comparison: no older run found for this machine." in section


def test_html_report_body_preserves_lines_and_escapes(tmp_path: Path) -> None:
    report = tmp_path / "daily.report"
    report.write_text("[ Summary ]\nA < B\n| col | value |\n", encoding="utf-8")

    body = delivery._html_report_body(report)

    assert "<pre" in body
    assert "[ Summary ]\nA &lt; B\n| col | value |" in body


def test_html_report_body_passthrough_for_html_files(tmp_path: Path) -> None:
    report = tmp_path / "daily.html"
    report.write_text("<html><body><h1>Hello</h1></body></html>", encoding="utf-8")

    body = delivery._html_report_body(report)

    assert body == "<html><body><h1>Hello</h1></body></html>"


def test_send_mail_pipes_preformatted_html_body(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "daily.html"
    report.write_text("<html><body><p>line 1</p><p>line 2</p></body></html>", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd: list[str], input: str, text: bool):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["text"] = text
        return _Result()

    monkeypatch.setattr(delivery.platform, "system", lambda: "Linux")
    monkeypatch.setattr(delivery.platform, "node", lambda: "TESTNODE")
    monkeypatch.setattr(delivery.subprocess, "run", _fake_run)

    ok = delivery.send_mail(report, "user@example.com", "daily report", now_stamp="20260505_2253")

    assert ok is True
    assert captured["cmd"] == [
        "mail",
        "--content-type=text/html",
        "-s",
        "[TESTNODE/20260505_2253] daily report",
        "user@example.com",
    ]
    assert captured["text"] is True
    assert "<pre" in str(captured["input"])
    assert "<p>line 1</p><p>line 2</p>" in str(captured["input"])


def test_send_mail_includes_analysis_summary_block(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "daily.report"
    report.write_text("line 1\n", encoding="utf-8")
    summary_json = tmp_path / "daily.summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "analysis": {
                    "overall_status": "yellow",
                    "baseline": {"status": "found", "stamp": "20260505_1200", "ov_version": "2026.2"},
                    "last_known_good": {"status": "found", "stamp": "20260501_1200", "ov_version": "2026.1"},
                    "bisect_delta": {
                        "status": "available",
                        "issue_stamp": "20260505_1200",
                        "issue_ov_version": "2026.2",
                        "last_good_stamp": "20260501_1200",
                        "last_good_ov_version": "2026.1",
                        "compared_count": 10,
                        "regressed_count": 2,
                        "functional_issue_count": 1,
                        "build_changed": True,
                        "sha_changed": True,
                    },
                    "functional": {"failed": 1, "error": 0},
                    "performance": {"compared": 10, "regressed": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd: list[str], input: str, text: bool):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["text"] = text
        return _Result()

    monkeypatch.setattr(delivery.platform, "system", lambda: "Linux")
    monkeypatch.setattr(delivery.platform, "node", lambda: "TESTNODE")
    monkeypatch.setattr(delivery.subprocess, "run", _fake_run)

    ok = delivery.send_mail(
        report,
        "user@example.com",
        "daily report",
        now_stamp="20260505_2253",
        summary_json=summary_json,
    )

    assert ok is True
    body = str(captured["input"])
    assert "Analysis summary" in body
    assert "overall: yellow" in body
    assert "last known good: 20260501_1200 / 2026.1" in body
    assert "bisect delta: issue=20260505_1200 / 2026.2" in body
    assert "functional_issues=1" in body
    assert "build=True sha=True" in body
    assert "functional: issues=1 failed=1 error=0" in body
    assert "performance: compared=10 regressed=2" in body


def test_send_mail_handles_malformed_analysis_values(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "daily.report"
    report.write_text("line 1\n", encoding="utf-8")
    summary_json = tmp_path / "daily.summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "analysis": {
                    "overall_status": "yellow",
                    "functional": {"failed": "N/A", "error": None},
                    "performance": {"compared": "oops", "regressed": "oops"},
                }
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd: list[str], input: str, text: bool):
        captured["input"] = input
        return _Result()

    monkeypatch.setattr(delivery.platform, "system", lambda: "Linux")
    monkeypatch.setattr(delivery.platform, "node", lambda: "TESTNODE")
    monkeypatch.setattr(delivery.subprocess, "run", _fake_run)

    ok = delivery.send_mail(
        report,
        "user@example.com",
        "daily report",
        now_stamp="20260505_2253",
        summary_json=summary_json,
    )

    assert ok is True
    body = str(captured["input"])
    assert "Analysis summary" in body
    assert "functional: issues=0 failed=0 error=0" in body
    assert "performance: compared=0 regressed=0" in body


def test_send_mail_handles_non_dict_json_payload(tmp_path: Path, monkeypatch) -> None:
    """summary.json that parses to a list (not dict) must not crash."""
    report = tmp_path / "daily.report"
    report.write_text("line 1\n", encoding="utf-8")
    summary_json = tmp_path / "daily.summary.json"
    summary_json.write_text("[]", encoding="utf-8")

    captured: dict[str, object] = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd: list[str], input: str, text: bool):
        captured["input"] = input
        return _Result()

    monkeypatch.setattr(delivery.platform, "system", lambda: "Linux")
    monkeypatch.setattr(delivery.platform, "node", lambda: "TESTNODE")
    monkeypatch.setattr(delivery.subprocess, "run", _fake_run)

    ok = delivery.send_mail(
        report,
        "user@example.com",
        "daily report",
        now_stamp="20260505_2253",
        summary_json=summary_json,
    )

    assert ok is True
    body = str(captured["input"])
    # No Analysis summary block when payload is not a dict.
    assert "Analysis summary" not in body


def test_run_analysis_compares_with_baseline(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bench.duckdb"
    assert not db_path.exists()
    monkeypatch.setattr(run, "VIEWER_DB", db_path)

    # Baseline run
    baseline_summary = tmp_path / "daily.20260501_0100.summary.json"
    baseline_summary.write_text(
        json.dumps(
            {
                "generated_at": 1776810000.0,
                "duration_sec": 10.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260501_0100",
                    "workweek": "2026.WW18.4",
                    "ov_version": "2026.2.0-21664-ad5d8e0f99b",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_base",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [{"in_token": 32, "out_token": 128, "perf": [10.0, 2.0]}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Current run (slower second-token latency => regression)
    current_summary = tmp_path / "daily.20260502_0100.summary.json"
    current_summary.write_text(
        json.dumps(
            {
                "generated_at": 1776896400.0,
                "duration_sec": 11.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260502_0100",
                    "workweek": "2026.WW18.5",
                    "ov_version": "2026.2.1-21680-bbbbbbb1234",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_cur",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [{"in_token": 32, "out_token": 128, "perf": [10.1, 2.4]}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Insert baseline first.
    report_base = tmp_path / "daily.20260501_0100.report"
    report_base.write_text("[ Summary ]\nbase\n", encoding="utf-8")
    run._run_analysis(report_base, baseline_summary)
    assert db_path.exists()
    baseline_data = json.loads(baseline_summary.read_text(encoding="utf-8"))
    assert "analysis" in baseline_data

    # Compare current against baseline.
    report_cur = tmp_path / "daily.20260502_0100.report"
    report_cur.write_text("[ Summary ]\ncurrent\n", encoding="utf-8")
    run._run_analysis(report_cur, current_summary)

    text = report_cur.read_text(encoding="utf-8")
    assert text.startswith("[ Analysis summary ]")
    assert "Model deltas:" in text
    assert "llama" in text
    assert "Overall verdict: Performance regression detected." in text

    current_data = json.loads(current_summary.read_text(encoding="utf-8"))
    assert current_data["analysis"]["overall_status"] == "yellow"
    assert "last_known_good" in current_data["analysis"]
    assert current_data["analysis"]["last_known_good"]["status"] == "not_found"


def test_run_analysis_writes_last_known_good_when_prior_green_exists(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bench.duckdb"
    monkeypatch.setattr(run, "VIEWER_DB", db_path)

    # run1: first run -> gray
    s1 = tmp_path / "daily.20260501_0100.summary.json"
    s1.write_text(
        json.dumps(
            {
                "generated_at": 1.0,
                "duration_sec": 10.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260501_0100",
                    "workweek": "2026.WW18.4",
                    "ov_version": "2026.2.0-11111-aaaaaaaaaaa",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_r1",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [{"in_token": 32, "out_token": 128, "perf": [10.0, 2.0]}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    r1 = tmp_path / "daily.20260501_0100.report"
    r1.write_text("[ Summary ]\nrun1\n", encoding="utf-8")
    run._run_analysis(r1, s1)

    # run2: same perf -> green
    s2 = tmp_path / "daily.20260502_0100.summary.json"
    s2.write_text(
        json.dumps(
            {
                "generated_at": 2.0,
                "duration_sec": 10.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260502_0100",
                    "workweek": "2026.WW18.5",
                    "ov_version": "2026.2.0-22222-bbbbbbbbbbb",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_r2",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [{"in_token": 32, "out_token": 128, "perf": [10.0, 2.0]}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    r2 = tmp_path / "daily.20260502_0100.report"
    r2.write_text("[ Summary ]\nrun2\n", encoding="utf-8")
    run._run_analysis(r2, s2)

    # run3: regression -> yellow and should carry LKG(found=run2)
    s3 = tmp_path / "daily.20260503_0100.summary.json"
    s3.write_text(
        json.dumps(
            {
                "generated_at": 3.0,
                "duration_sec": 10.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260503_0100",
                    "workweek": "2026.WW18.6",
                    "ov_version": "2026.2.1-33333-ccccccccccc",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_r3",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [{"in_token": 32, "out_token": 128, "perf": [10.0, 2.4]}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    r3 = tmp_path / "daily.20260503_0100.report"
    r3.write_text("[ Summary ]\nrun3\n", encoding="utf-8")
    run._run_analysis(r3, s3)

    d3 = json.loads(s3.read_text(encoding="utf-8"))
    assert d3["analysis"]["overall_status"] == "yellow"
    assert d3["analysis"]["last_known_good"]["status"] == "found"
    assert d3["analysis"]["last_known_good"]["stamp"] == "20260502_0100"
    assert d3["analysis"]["bisect_delta"]["status"] == "available"
    assert d3["analysis"]["bisect_delta"]["last_good_run_id"] is not None
    assert d3["analysis"]["bisect_delta"]["regressed_count"] >= 1


def test_run_analysis_reports_lkg_when_baseline_not_found(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bench.duckdb"
    monkeypatch.setattr(run, "VIEWER_DB", db_path)

    # run1: first perf run -> gray
    s1 = tmp_path / "daily.20260501_0100.summary.json"
    s1.write_text(
        json.dumps(
            {
                "generated_at": 1.0,
                "duration_sec": 10.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260501_0100",
                    "workweek": "2026.WW18.4",
                    "ov_version": "2026.2.0-11111-aaaaaaaaaaa",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_r1",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [{"in_token": 32, "out_token": 128, "perf": [10.0, 2.0]}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    r1 = tmp_path / "daily.20260501_0100.report"
    r1.write_text("[ Summary ]\nrun1\n", encoding="utf-8")
    run._run_analysis(r1, s1)

    # run2: comparable perf run -> green (becomes candidate LKG)
    s2 = tmp_path / "daily.20260502_0100.summary.json"
    s2.write_text(
        json.dumps(
            {
                "generated_at": 2.0,
                "duration_sec": 10.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260502_0100",
                    "workweek": "2026.WW18.5",
                    "ov_version": "2026.2.0-22222-bbbbbbbbbbb",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_r2",
                        "outcome": "passed",
                        "metrics": {
                            "test_type": "llm_benchmark",
                            "model": "llama",
                            "precision": "FP16",
                            "data": [{"in_token": 32, "out_token": 128, "perf": [10.0, 2.0]}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    r2 = tmp_path / "daily.20260502_0100.report"
    r2.write_text("[ Summary ]\nrun2\n", encoding="utf-8")
    run._run_analysis(r2, s2)

    # run3: functional issue with no perf rows -> red
    # Baseline must be not_found (no overlap), but LKG should still be found.
    s3 = tmp_path / "daily.20260503_0100.summary.json"
    s3.write_text(
        json.dumps(
            {
                "generated_at": 3.0,
                "duration_sec": 10.0,
                "meta": {
                    "machine": "LNL-03",
                    "stamp": "20260503_0100",
                    "workweek": "2026.WW18.6",
                    "ov_version": "2026.2.1-33333-ccccccccccc",
                    "description": "daily_CB timer",
                    "device": "GPU",
                },
                "totals": {"passed": 0, "failed": 1, "error": 0, "skipped": 0, "total": 1},
                "tests": [
                    {
                        "nodeid": "test_llm_r3",
                        "outcome": "failed",
                        "longrepr": "assert False",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    r3 = tmp_path / "daily.20260503_0100.report"
    r3.write_text("[ Summary ]\nrun3\n", encoding="utf-8")
    run._run_analysis(r3, s3)

    d3 = json.loads(s3.read_text(encoding="utf-8"))
    assert d3["analysis"]["overall_status"] == "red"
    assert d3["analysis"]["baseline"]["status"] == "not_found"
    assert d3["analysis"]["last_known_good"]["status"] == "found"
    assert d3["analysis"]["last_known_good"]["stamp"] == "20260502_0100"
    assert d3["analysis"]["bisect_delta"]["status"] == "unavailable"
    assert d3["analysis"]["bisect_delta"]["compared_count"] == 0
    assert d3["analysis"]["bisect_delta"]["functional_issue_count"] == 1

    text = r3.read_text(encoding="utf-8")
    assert "Baseline comparison: no older run found for this machine." in text
    assert "Last known good: stamp=20260502_0100" in text
