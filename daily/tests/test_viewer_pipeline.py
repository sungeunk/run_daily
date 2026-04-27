from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

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
                    {
                        "nodeid": "test_dropped",
                        "outcome": "passed",
                        "metrics": {"test_type": "qwen_usage", "data": []},
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


def test_format_regression_alerts_reports_threshold_hits() -> None:
    frame = pd.DataFrame(
        [
            {
                "status": "ok",
                "model": "llama",
                "precision": "FP16",
                "in_token": 32,
                "out_token": 128,
                "exec_mode": "2nd",
                "unit": "ms",
                "worsening_pct": 0.12,
                "worsening_z": 3.5,
                "recent_median": 12.0,
                "baseline_median": 10.0,
                "recent_cv": 0.04,
            },
            {
                "status": "ok",
                "model": "stable",
                "precision": "FP16",
                "in_token": 0,
                "out_token": 0,
                "exec_mode": "pipeline",
                "unit": "s",
                "worsening_pct": 0.01,
                "worsening_z": 0.5,
                "recent_median": 8.1,
                "baseline_median": 8.0,
                "recent_cv": 0.02,
            },
        ]
    )

    section = run._format_regression_alerts(frame)

    assert "[ Regression alerts ]" in section
    assert "llama | FP16" in section
    assert "stable | FP16" not in section


def test_format_regression_alerts_handles_no_hits() -> None:
    frame = pd.DataFrame(
        [
            {
                "status": "ok",
                "recent_cv": 0.02,
                "worsening_pct": 0.01,
                "worsening_z": 0.5,
            }
        ]
    )

    assert "No series exceeded" in run._format_regression_alerts(frame)
