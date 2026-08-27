"""Tests for the count-based viewer queries and machine-state attribution."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

DAILY_DIR = Path(__file__).resolve().parent.parent
if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

from analysis.functional import aggregate_functional  # noqa: E402
from viewer import queries as q  # noqa: E402
from viewer.ingest import writer  # noqa: E402
from viewer.ingest.loader_new import classify_run_kind  # noqa: E402
from viewer.ingest.record import MonitorRow, PerfRow, RunRecord  # noqa: E402

MACHINE = "TEST-01"
BASE_TS = datetime(2026, 1, 1, 12, 0)


def _record(idx: int, *, value: float, run_kind: str = "daily",
            model: str = "llama", monitor: MonitorRow | None = None,
            short_run: bool = False) -> RunRecord:
    ts = BASE_TS + timedelta(days=idx)
    rec = RunRecord(
        run_id=f"run-{idx:03d}",
        source_format="new",
        report_file=f"daily.{idx}.summary.json",
        machine=MACHINE,
        ts=ts,
        purpose=run_kind,
        run_kind=run_kind,
        short_run=short_run,
    )
    rec.perf.append(PerfRow(model, "INT4", 32, 128, "2nd", value, "ms"))
    if monitor is not None:
        rec.monitor.append(monitor)
    return rec


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "test.duckdb"
    con = writer.connect(path)
    writer.ensure_schema(con)
    con.close()
    return path


def _write(db_path: Path, records: list[RunRecord]) -> None:
    con = writer.connect(db_path)
    writer.ensure_schema(con)
    for rec in records:
        writer.upsert_run(con, rec)
    con.close()


class TestRunKind:
    def test_pr_and_test_runs_are_not_daily(self):
        assert classify_run_kind("PR-1234 validation") == "pr"
        assert classify_run_kind("jenkins test build") == "test"
        assert classify_run_kind("daily_CB") == "daily"
        assert classify_run_kind(None) == "manual"

    def test_pr_number_separators(self):
        for text in ("PR#36380 - SlideWindowSize sdpa", "pr 36380", "pr_36380"):
            assert classify_run_kind(text) == "pr", text

    def test_daily_variants_are_recognised(self):
        assert classify_run_kind("daily2 timer") == "daily"
        assert classify_run_kind("nightly") == "daily"

    def test_short_tokens_do_not_match_inside_words(self):
        # 'ci' must not match 'precision', 'pr' must not match 'preview'.
        assert classify_run_kind("precision sweep") == "manual"
        assert classify_run_kind("preview build") == "manual"


class TestCohort:
    def test_recent_runs_uses_count_not_days(self, db: Path):
        _write(db, [_record(i, value=100.0) for i in range(20)])
        cohort = q.recent_runs(db, MACHINE, limit=5)
        assert len(cohort) == 5
        # oldest first, newest last
        assert cohort.iloc[-1]["run_id"] == "run-019"
        assert cohort.iloc[0]["run_id"] == "run-015"

    def test_pr_runs_excluded_by_default(self, db: Path):
        records = [_record(i, value=100.0) for i in range(3)]
        records.append(_record(3, value=999.0, run_kind="pr"))
        _write(db, records)

        default_cohort = q.recent_runs(db, MACHINE, limit=10)
        assert "run-003" not in set(default_cohort["run_id"])

        both = q.recent_runs(db, MACHINE, limit=10, run_kinds=("daily", "pr"))
        assert "run-003" in set(both["run_id"])

    def test_short_runs_can_be_excluded(self, db: Path):
        records = [_record(i, value=100.0) for i in range(3)]
        records.append(_record(3, value=100.0, short_run=True))
        _write(db, records)

        cohort = q.recent_runs(db, MACHINE, limit=10, include_short_run=False)
        assert "run-003" not in set(cohort["run_id"])


class TestSeriesTrend:
    def test_regression_detected_against_run_history(self, db: Path):
        records = [_record(i, value=100.0) for i in range(10)]
        records += [_record(i, value=120.0) for i in range(10, 12)]
        _write(db, records)

        trend = q.series_trend(db, MACHINE, recent_runs_n=2, history_runs_n=10)
        assert len(trend) == 1
        row = trend.iloc[0]
        assert row["status"] == "ok"
        assert row["recent_n"] == 2
        assert row["history_n"] == 10
        # ms is lower-is-better, so a rise is a regression (positive worsening)
        assert row["worsening_pct"] == pytest.approx(0.20)

    def test_pr_run_does_not_pollute_history(self, db: Path):
        records = [_record(i, value=100.0) for i in range(10)]
        records.append(_record(10, value=500.0, run_kind="pr"))
        records.append(_record(11, value=100.0))
        _write(db, records)

        trend = q.series_trend(db, MACHINE, recent_runs_n=1, history_runs_n=10)
        row = trend.iloc[0]
        assert row["worsening_pct"] == pytest.approx(0.0)

    def test_model_filter_limits_series(self, db: Path):
        records = []
        for i in range(6):
            rec = _record(i, value=100.0, model="llama")
            rec.perf.append(PerfRow("qwen", "INT4", 32, 128, "2nd", 50.0, "ms"))
            records.append(rec)
        _write(db, records)

        trend = q.series_trend(db, MACHINE, recent_runs_n=1, history_runs_n=5,
                               models=["qwen"])
        assert set(trend["model"]) == {"qwen"}

    def test_insufficient_history_is_flagged(self, db: Path):
        _write(db, [_record(i, value=100.0) for i in range(3)])
        trend = q.series_trend(db, MACHINE, recent_runs_n=1, history_runs_n=2)
        assert trend.iloc[0]["status"] == "insufficient_data"


class TestGeomean:
    def test_common_series_only_ignores_partial_runs(self, db: Path):
        records = []
        for i in range(3):
            rec = _record(i, value=100.0, model="llama")
            if i < 2:
                rec.perf.append(
                    PerfRow("qwen", "INT4", 32, 128, "2nd", 400.0, "ms"))
            records.append(rec)
        _write(db, records)

        run_ids = q.recent_runs(db, MACHINE, limit=3)["run_id"].tolist()
        geo = q.geomean_for_runs(db, run_ids)
        # 'qwen' is missing from the newest run, so it must not move the curve.
        assert geo["geomean"].nunique() == 1


class TestTrendAwareCompare:
    def test_compare_reports_history_context(self, db: Path):
        records = [_record(i, value=100.0) for i in range(10)]
        records.append(_record(10, value=130.0))
        _write(db, records)

        df = q.compare_runs_with_trend(db, MACHINE, "run-010", "run-009",
                                       history_runs_n=10)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["value_a"] == 130.0
        assert row["value_b"] == 100.0
        assert row["median_a"] == pytest.approx(100.0)
        assert row["trend_context"] == "outside_history"

    def test_stable_series_is_within_history(self, db: Path):
        records = [_record(i, value=100.0 + (i % 2)) for i in range(11)]
        _write(db, records)

        df = q.compare_runs_with_trend(db, MACHINE, "run-010", "run-009",
                                       history_runs_n=10)
        assert df.iloc[0]["trend_context"] in {"within_history", "unknown"}


class TestMachineState:
    def test_monitor_summary_is_ingested(self, db: Path):
        monitor = MonitorRow(
            nodeid="tests/test_llm.py::test_llama",
            model="llama",
            samples=100,
            gpu_clock_mhz_mean=2000.0,
            gpu_clock_max_mhz=2400.0,
            throttled_sample_ratio=0.0,
        )
        _write(db, [_record(0, value=100.0, monitor=monitor)])

        health = q.machine_health_for_runs(db, ["run-000"])
        assert len(health) == 1
        assert health.iloc[0]["gpu_clock_ratio"] == pytest.approx(2000 / 2400)

    def test_throttled_run_is_classified(self, db: Path):
        monitor = MonitorRow(nodeid="t", model="llama", samples=10,
                             gpu_clock_mhz_mean=2000.0,
                             gpu_clock_max_mhz=2400.0,
                             throttled_sample_ratio=0.5)
        _write(db, [_record(0, value=100.0, monitor=monitor)])
        health = q.machine_health_for_runs(db, ["run-000"])
        assert q.classify_machine_state(health.iloc[0]) == "throttled"

    def test_low_clock_ratio_is_throttled(self, db: Path):
        monitor = MonitorRow(nodeid="t", model="llama", samples=10,
                             gpu_clock_mhz_mean=1000.0,
                             gpu_clock_max_mhz=2400.0,
                             throttled_sample_ratio=0.0)
        _write(db, [_record(0, value=100.0, monitor=monitor)])
        health = q.machine_health_for_runs(db, ["run-000"])
        assert q.classify_machine_state(health.iloc[0]) == "throttled"

    def test_stable_machine(self, db: Path):
        monitor = MonitorRow(nodeid="t", model="llama", samples=10,
                             gpu_clock_mhz_mean=2350.0,
                             gpu_clock_max_mhz=2400.0,
                             throttled_sample_ratio=0.0)
        _write(db, [_record(0, value=100.0, monitor=monitor)])
        health = q.machine_health_for_runs(db, ["run-000"])
        assert q.classify_machine_state(health.iloc[0]) == "stable"

    def test_missing_telemetry_is_unknown(self, db: Path):
        _write(db, [_record(0, value=100.0)])
        assert q.machine_health_for_runs(db, ["run-000"]).empty


class TestAttribution:
    def test_regression_on_disturbed_machine_is_machine_attributed(self):
        assert q.attribute_regression("regressed", "throttled") == "likely-machine"
        assert q.attribute_regression("regressed", "fluctuating") == "likely-machine"

    def test_regression_on_stable_machine_is_code_attributed(self):
        assert q.attribute_regression("regressed", "stable") == "likely-code"

    def test_unknown_machine_state_is_inconclusive(self):
        assert q.attribute_regression("regressed", "unknown") == "inconclusive"

    def test_non_regression_is_not_attributed(self):
        assert q.attribute_regression("same", "throttled") == "n/a"


class TestFunctionalModelColumn:
    def test_model_is_extracted_from_metrics(self):
        summary = {
            "totals": {"total": 1, "passed": 0, "failed": 1,
                       "error": 0, "skipped": 0},
            "tests": [{
                "nodeid": "tests/test_llm_benchmark.py::test_llm[case-1]",
                "outcome": "failed",
                "failure": "boom",
                # Recorded before the benchmark runs, so a failed test still
                # says which model it was exercising.
                "metrics": {"model": "llama-2-7b", "precision": "INT4"},
            }],
        }
        result = aggregate_functional(summary)
        assert len(result.issues) == 1
        assert result.issues[0].model == "llama-2-7b"
        assert result.issues[0].precision == "INT4"

    def test_missing_metrics_leaves_model_none(self):
        summary = {
            "totals": {"total": 1, "passed": 0, "failed": 1,
                       "error": 0, "skipped": 0},
            "tests": [{"nodeid": "t::x", "outcome": "failed"}],
        }
        issue = aggregate_functional(summary).issues[0]
        assert issue.model is None
        assert issue.precision is None

    def test_issues_round_trip_through_db_with_model(self, db: Path):
        _write(db, [_record(0, value=100.0)])
        con = writer.connect(db)
        con.execute(
            "INSERT INTO functional_issues "
            "(run_id, nodeid, outcome, message, model, precision) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ["run-000", "t::a", "failed", "boom", "qwen", "INT4"],
        )
        con.close()

        issues = q.functional_issues_for_runs(db, ["run-000"])
        assert issues.iloc[0]["model"] == "qwen"
        assert issues.iloc[0]["precision"] == "INT4"

    def test_model_filter_selects_matching_issues(self, db: Path):
        _write(db, [_record(0, value=100.0)])
        con = writer.connect(db)
        con.executemany(
            "INSERT INTO functional_issues "
            "(run_id, nodeid, outcome, message, model, precision) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [["run-000", "t::a", "failed", "", "qwen", "INT4"],
             ["run-000", "t::b", "failed", "", "llama", "INT4"]],
        )
        con.close()

        issues = q.functional_issues_for_runs(db, ["run-000"], models=["qwen"])
        assert set(issues["nodeid"]) == {"t::a"}

    def test_legacy_rows_without_model_stay_visible(self, db: Path):
        _write(db, [_record(0, value=100.0)])
        con = writer.connect(db)
        con.execute(
            "INSERT INTO functional_issues (run_id, nodeid, outcome, message) "
            "VALUES (?, ?, ?, ?)",
            ["run-000", "t::legacy", "failed", "old row"],
        )
        con.close()

        issues = q.functional_issues_for_runs(db, ["run-000"], models=["qwen"])
        assert "t::legacy" in set(issues["nodeid"])
