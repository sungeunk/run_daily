"""Tests for the count-based viewer queries and machine-state attribution."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.dev_only

DAILY_DIR = Path(__file__).resolve().parent.parent
if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

from analysis.functional import aggregate_functional  # noqa: E402
from viewer import queries as q  # noqa: E402
from viewer.ingest import writer  # noqa: E402
from viewer.ingest.loader_new import _cases, _skipped_cases, classify_run_kind  # noqa: E402
from viewer.ingest.record import IssueRow, MonitorRow, PerfRow, RunRecord  # noqa: E402

MACHINE = "TEST-01"
BASE_TS = datetime(2026, 1, 1, 12, 0)


def _record(idx: int, *, value: float, run_kind: str = "daily",
            model: str = "llama",
            monitor: MonitorRow | None = None) -> RunRecord:
    ts = BASE_TS + timedelta(days=idx)
    rec = RunRecord(
        run_id=f"run-{idx:03d}",
        source_format="new",
        report_file=f"daily.{idx}.summary.json",
        machine=MACHINE,
        ts=ts,
        purpose=run_kind,
        run_kind=run_kind,
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

    def test_runs_with_too_few_series_are_dropped(self, db: Path):
        # A partial run would otherwise collapse the common-series
        # intersection every cohort view is built on.
        records = []
        for i in range(3):
            rec = _record(i, value=100.0)
            for n in range(4):
                rec.perf.append(
                    PerfRow(f"m{n}", "INT4", 32, 128, "2nd", 50.0, "ms"))
            records.append(rec)
        records.append(_record(3, value=100.0))
        _write(db, records)

        cohort = q.recent_runs(db, MACHINE, limit=10, min_success_series=3)
        assert "run-003" not in set(cohort["run_id"])
        assert len(cohort) == 3

        unfiltered = q.recent_runs(db, MACHINE, limit=10)
        assert "run-003" in set(unfiltered["run_id"])


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
                             throttled_sample_ratio=0.9)
        _write(db, [_record(0, value=100.0, monitor=monitor)])
        health = q.machine_health_for_runs(db, ["run-000"])
        assert q.classify_machine_state(health.iloc[0]) == "throttled"

    def test_brief_throttling_is_not_flagged(self, db: Path):
        # Short throttle stretches show up in nearly every run, so one test
        # blipping must not label the whole run machine-limited.
        brief = MonitorRow(nodeid="a", model="llama", samples=100,
                           gpu_clock_mhz_mean=2350.0,
                           gpu_clock_max_mhz=2400.0,
                           throttled_sample_ratio=0.08)
        clean = MonitorRow(nodeid="b", model="qwen", samples=100,
                           gpu_clock_mhz_mean=2350.0,
                           gpu_clock_max_mhz=2400.0,
                           throttled_sample_ratio=0.0)
        rec = _record(0, value=100.0, monitor=brief)
        rec.monitor.append(clean)
        _write(db, [rec])
        health = q.machine_health_for_runs(db, ["run-000"])
        assert q.classify_machine_state(health.iloc[0]) == "stable"

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


class TestCaseCounts:
    def test_skipped_cases_sums_expected_series(self):
        summary = {
            "tests": [
                # One pytest test can stand for several benchmark cases.
                {"outcome": "skipped", "metrics": {"expected_series": 4}},
                {"outcome": "skipped", "metrics": {"expected_series": 2}},
                {"outcome": "passed", "metrics": {"expected_series": 8}},
            ],
        }
        assert _skipped_cases(summary) == 6

    def test_missing_expected_series_counts_as_zero(self):
        summary = {"tests": [{"outcome": "skipped", "metrics": {}}]}
        assert _skipped_cases(summary) == 0

    def test_expected_cases_counts_every_outcome(self):
        summary = {
            "tests": [
                {"outcome": "passed", "metrics": {"expected_series": 8}},
                {"outcome": "failed", "metrics": {"expected_series": 4}},
                {"outcome": "skipped", "metrics": {"expected_series": 2}},
            ],
        }
        assert _cases(summary) == 14

    def test_expected_cases_comes_from_the_run_itself(self, db: Path):
        # The run declares 4 cases but produced 1, so 3 are unaccounted for
        # even though the machine has never produced more than 1.
        rec = _record(0, value=100.0)
        rec.expected_cases = 4
        rec.skipped_cases = 1
        _write(db, [rec])

        row = q.machines_overview(db, [MACHINE]).iloc[0]
        assert row["success_cases"] == 1
        assert row["expected_cases"] == 4

    def test_expected_cases_falls_back_to_the_machines_best_run(self, db: Path):
        # Runs ingested before ``expected_cases`` existed: a run that broke
        # early must not lower what the machine is expected to produce,
        # otherwise its failures would vanish from the table.
        healthy = _record(0, value=100.0)
        healthy.perf.append(PerfRow("qwen", "INT4", 32, 128, "2nd", 50.0, "ms"))
        healthy.perf.append(PerfRow("qwen", "INT4", 64, 128, "2nd", 55.0, "ms"))
        broken = _record(1, value=100.0)
        _write(db, [healthy, broken])

        ov = q.machines_overview(db, [MACHINE])
        row = ov.iloc[0]
        assert row["success_cases"] == 1
        assert row["expected_cases"] == 3


class TestFleetStatus:
    def _full(self, idx: int) -> RunRecord:
        rec = _record(idx, value=100.0)
        rec.perf.append(PerfRow("qwen", "INT4", 32, 128, "2nd", 50.0, "ms"))
        rec.total_tests, rec.passed_tests = 2, 2
        rec.failed_tests = rec.error_tests = rec.skipped_tests = 0
        rec.expected_cases = 2
        return rec

    def test_partial_run_is_not_a_success(self, db: Path):
        # A run that only measured a fraction of the suite must not be
        # reported as the machine's last success.
        partial = self._full(1)
        partial.perf.pop()
        partial.total_tests = partial.passed_tests = 1
        partial.expected_cases = 1
        _write(db, [self._full(0), partial])

        row = q.machines_overview(db, [MACHINE]).iloc[0]
        assert row["latest_failed"]
        assert row["last_success_stamp"] == "20260101_1200"
        assert row["last_fail_stamp"] == "20260102_1200"

    def test_status_and_stamps_agree(self, db: Path):
        broken = self._full(1)
        broken.failed_tests, broken.passed_tests = 1, 1
        broken.perf.pop()
        _write(db, [self._full(0), broken])

        row = q.machines_overview(db, [MACHINE]).iloc[0]
        assert row["latest_failed"]
        assert row["last_fail_stamp"] == row["stamp"]

    def test_clean_run_is_not_a_failure(self, db: Path):
        _write(db, [self._full(0), self._full(1)])

        row = q.machines_overview(db, [MACHINE]).iloc[0]
        assert not row["latest_failed"]
        assert row["last_success_stamp"] == row["stamp"]
        assert row["last_fail_stamp"] is None

    def test_failed_models_are_named(self, db: Path):
        broken = self._full(1)
        broken.failed_tests, broken.passed_tests = 1, 1
        broken.perf.pop()
        broken.issues.append(IssueRow(
            nodeid="tests/test_llm_benchmark.py::test_llm_benchmark[qwen]",
            outcome="failed", model="qwen", precision="INT4"))
        _write(db, [self._full(0), broken])

        row = q.failing_models_overview(db, [MACHINE]).iloc[0]
        assert row["model"] == "qwen"
        assert not row["failed_before"]

    def test_a_model_failing_before_is_flagged(self, db: Path):
        records = []
        for i in range(2):
            rec = self._full(i)
            rec.failed_tests, rec.passed_tests = 1, 1
            rec.perf.pop()
            rec.issues.append(IssueRow(
                nodeid="tests/test_llm_benchmark.py::test_llm_benchmark[qwen]",
                outcome="failed", model="qwen", precision="INT4"))
            records.append(rec)
        _write(db, records)

        row = q.failing_models_overview(db, [MACHINE]).iloc[0]
        assert row["failed_before"]
        assert row["failed_runs"] == 2
        assert row["first_seen"] == "20260101_1200"

    def test_model_cache_change_is_flagged(self, db: Path):
        # The model passed on the old cache, then failed on the new one.
        passing = self._full(0)
        passing.model_cache = "WW24_llm-optimum_2026.3.0-22130"

        records = [passing]
        for i in (1, 2):
            rec = self._full(i)
            rec.model_cache = "WW35_llm-optimum_2026.4.0-22930-RC1"
            rec.failed_tests, rec.passed_tests = 1, 1
            rec.perf.pop()
            rec.issues.append(IssueRow(
                nodeid="tests/test_llm_benchmark.py::test_llm_benchmark[qwen]",
                outcome="failed", model="qwen", precision="INT4"))
            records.append(rec)
        _write(db, records)

        row = q.failing_models_overview(db, [MACHINE]).iloc[0]
        assert row["model_cache"] == "WW35_llm-optimum_2026.4.0-22930-RC1"
        # The immediately preceding run already used the new cache; the
        # comparison must reach back to the last run the model passed in.
        assert row["last_pass_model_cache"] == "WW24_llm-optimum_2026.3.0-22130"
        assert row["last_pass_stamp"] == "20260101_1200"
        assert row["model_cache_changed"]


class TestShortDeviceName:
    @pytest.mark.parametrize("full,expected", [
        ("Intel(R) Arc(TM) 140T GPU (16GB) (iGPU)", "140T"),
        ("Intel(R) Arc(TM) B580 Graphics (dGPU)", "B580"),
        ("Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)", "140V"),
        ("Intel(R) Arc(TM) B390 GPU (iGPU)", "B390"),
        ("Intel(R) Arc(TM) Pro B70 Graphics (dGPU)", "B70"),
        ("Intel(R) Arc(TM) A770 Graphics (dGPU)", "A770"),
    ])
    def test_model_token_is_extracted(self, full, expected):
        assert q.short_device_name(full) == expected

    def test_name_without_model_token_stays_readable(self):
        assert q.short_device_name("Intel(R) Arc(TM) Graphics (iGPU)") == "Arc"

    def test_missing_name_is_empty(self):
        assert q.short_device_name(None) == ""


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
