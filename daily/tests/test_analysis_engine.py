"""Unit tests for daily/analysis/*.

Covers:
- verdict boundary values
- metric direction (latency vs throughput)
- baseline selection priority
- no-baseline and no-comparable-rows paths
- functional fail → overall_status = red priority
- model-level aggregation
- analysis block written to summary.json
- report [ Analysis summary ] prepend
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from analysis.types import (
    AnalysisConfig,
    BaselineInfo,
    ComparisonRow,
    FunctionalResult,
    FunctionalIssue,
    ModelSummary,
    PerformanceResult,
    SeriesKey,
)
from analysis.verdict import improvement_pct, verdict_from_pct, verdict_from_signal, make_comparison_row
from analysis.functional import aggregate_functional
from analysis.engine import (
    _aggregate_models,
    _aggregate_performance,
    _fetch_comparison_rows,
    _overall_status,
    _top_regressions,
)
from analysis.report import render_analysis_summary, prepend_to_report
from analysis.persistence import write_analysis_to_summary, _result_to_dict, write_analysis_to_db
from analysis.baseline import find_last_known_good, select_baseline
from report.builder import build_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = AnalysisConfig(pct_threshold=0.05)

_KEY = SeriesKey("llama", "FP16", 32, 128, "2nd")


def _make_result(
    rows: list[ComparisonRow] | None = None,
    functional_failed: int = 0,
    baseline_found: bool = True,
):
    """Build a minimal AnalysisResult for testing helpers."""
    from analysis.engine import _aggregate_models, _aggregate_performance, _top_regressions, _overall_status
    from analysis.types import AnalysisResult

    rows = rows or []
    functional = FunctionalResult(
        total=10,
        passed=10 - functional_failed,
        failed=functional_failed,
        error=0,
        skipped=0,
    )
    performance = _aggregate_performance(rows)
    models = _aggregate_models(rows)
    top_reg = _top_regressions(rows, 5)
    baseline = (
        BaselineInfo(status="found", run_id="run-0", stamp="20260101_0000", ov_version="2026.0")
        if baseline_found
        else BaselineInfo(status="not_found")
    )
    status = _overall_status(functional, performance, baseline)
    return AnalysisResult(
        overall_status=status,
        baseline=baseline,
        functional=functional,
        performance=performance,
        models=models,
        top_regressions=top_reg,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# verdict.py
# ---------------------------------------------------------------------------

class TestImprovementPct:
    def test_throughput_higher_is_better(self):
        pct = improvement_pct(current=110.0, baseline=100.0, unit="tps")
        assert abs(pct - 0.10) < 1e-9

    def test_latency_lower_is_better(self):
        pct = improvement_pct(current=90.0, baseline=100.0, unit="ms")
        assert abs(pct - 0.10) < 1e-9   # current is better → positive

    def test_latency_regression(self):
        pct = improvement_pct(current=110.0, baseline=100.0, unit="ms")
        assert abs(pct - (-0.10)) < 1e-9  # current is worse → negative

    def test_zero_baseline_returns_none(self):
        assert improvement_pct(current=5.0, baseline=0.0, unit="ms") is None

    def test_none_baseline_returns_none(self):
        assert improvement_pct(current=5.0, baseline=None, unit="ms") is None  # type: ignore[arg-type]


class TestVerdictFromPct:
    cfg = _DEFAULT_CONFIG  # threshold = 5 %

    def test_exactly_at_threshold_improved(self):
        assert verdict_from_pct(0.05, self.cfg) == "improved"

    def test_just_below_threshold_same(self):
        assert verdict_from_pct(0.049, self.cfg) == "same"

    def test_exactly_at_negative_threshold_regressed(self):
        assert verdict_from_pct(-0.05, self.cfg) == "regressed"

    def test_just_above_negative_threshold_same(self):
        assert verdict_from_pct(-0.049, self.cfg) == "same"

    def test_none_returns_unavailable(self):
        assert verdict_from_pct(None, self.cfg) == "unavailable"

    def test_zero_returns_same(self):
        assert verdict_from_pct(0.0, self.cfg) == "same"

    def test_nan_returns_unavailable(self):
        assert verdict_from_pct(float("nan"), self.cfg) == "unavailable"


class TestVerdictFromSignal:
    cfg = AnalysisConfig(pct_threshold=0.05, z_threshold=3.0, noisy_cv_threshold=0.10)

    def test_noisy_when_cv_high(self):
        assert verdict_from_signal(-0.20, self.cfg, recent_cv=0.12) == "noisy"

    def test_insufficient_when_recent_points_low(self):
        assert verdict_from_signal(-0.20, self.cfg, recent_n=2, baseline_n=10) == "insufficient"

    def test_insufficient_when_baseline_points_low(self):
        assert verdict_from_signal(-0.20, self.cfg, recent_n=10, baseline_n=2) == "insufficient"

    def test_regression_requires_z_when_provided(self):
        assert verdict_from_signal(-0.20, self.cfg, worsening_z=2.5) == "same"
        assert verdict_from_signal(-0.20, self.cfg, worsening_z=3.0) == "regressed"

    def test_improvement_uses_pct_gate(self):
        assert verdict_from_signal(0.08, self.cfg, worsening_z=0.1) == "improved"


class TestMakeComparisonRow:
    def test_improved_throughput(self):
        row = make_comparison_row(_KEY, "tps", 110.0, 100.0, _DEFAULT_CONFIG)
        assert row.verdict == "improved"
        assert abs(row.improvement_pct - 0.10) < 1e-9

    def test_regressed_latency(self):
        row = make_comparison_row(_KEY, "ms", 120.0, 100.0, _DEFAULT_CONFIG)
        assert row.verdict == "regressed"

    def test_same(self):
        row = make_comparison_row(_KEY, "ms", 102.0, 100.0, _DEFAULT_CONFIG)
        assert row.verdict == "same"


# ---------------------------------------------------------------------------
# functional.py
# ---------------------------------------------------------------------------

class TestAggregateFunctional:
    def test_all_passed(self):
        summary = {"totals": {"total": 5, "passed": 5, "failed": 0, "error": 0, "skipped": 0}, "tests": []}
        result = aggregate_functional(summary)
        assert result.failed == 0
        assert result.issues == []

    def test_one_failure_captured(self):
        summary = {
            "totals": {"total": 2, "passed": 1, "failed": 1, "error": 0, "skipped": 0},
            "tests": [
                {"nodeid": "test_foo", "outcome": "failed", "longrepr": "AssertionError"},
                {"nodeid": "test_bar", "outcome": "passed"},
            ],
        }
        result = aggregate_functional(summary)
        assert result.failed == 1
        assert len(result.issues) == 1
        assert result.issues[0].nodeid == "test_foo"
        assert result.issues[0].outcome == "failed"

    def test_missing_totals(self):
        result = aggregate_functional({})
        assert result.total == 0
        assert result.issues == []

    def test_long_message_truncated(self):
        long_msg = "x" * 300
        summary = {
            "totals": {"total": 1, "passed": 0, "failed": 1, "error": 0, "skipped": 0},
            "tests": [{"nodeid": "t", "outcome": "failed", "longrepr": long_msg}],
        }
        result = aggregate_functional(summary)
        assert len(result.issues[0].message) <= 201  # 200 chars + ellipsis


# ---------------------------------------------------------------------------
# engine helpers
# ---------------------------------------------------------------------------

class TestAggregatePerformance:
    def _row(self, verdict):
        pct = {"improved": 0.1, "same": 0.0, "regressed": -0.1, "unavailable": None}[verdict]
        return ComparisonRow(_KEY, "ms", 1.0, 1.0, improvement_pct=pct, verdict=verdict)

    def test_counts(self):
        rows = [self._row("improved"), self._row("same"), self._row("regressed")]
        p = _aggregate_performance(rows)
        assert p.compared == 3
        assert p.improved == 1
        assert p.same == 1
        assert p.regressed == 1
        assert p.unavailable == 0


class TestAggregateModels:
    def test_avg_pct_and_sort(self):
        rows = [
            ComparisonRow(SeriesKey("bigmodel", "FP16", 32, 128, "2nd"), "ms", 1.0, 1.0, -0.20, "regressed"),
            ComparisonRow(SeriesKey("bigmodel", "FP16", 64, 128, "2nd"), "ms", 1.0, 1.0, -0.10, "regressed"),
            ComparisonRow(SeriesKey("smallmodel", "FP16", 32, 128, "2nd"), "ms", 1.0, 1.0, 0.02, "same"),
        ]
        models = _aggregate_models(rows)
        # bigmodel has bigger absolute avg → should be first
        assert models[0].model == "bigmodel"
        assert abs(models[0].avg_improvement_pct - (-0.15)) < 1e-9
        assert models[0].regressed == 2


class TestTopRegressions:
    def test_returns_only_regressed_rows(self):
        rows = [
            ComparisonRow(_KEY, "ms", 120.0, 100.0, -0.20, "regressed"),
            ComparisonRow(_KEY, "ms", 102.0, 100.0, -0.02, "same"),
            ComparisonRow(_KEY, "tps", 110.0, 100.0, 0.10, "improved"),
        ]

        top = _top_regressions(rows, 5)
        assert len(top) == 1
        assert top[0].verdict == "regressed"

    def test_empty_when_no_regressions(self):
        rows = [
            ComparisonRow(_KEY, "ms", 100.0, 100.0, 0.0, "same"),
            ComparisonRow(_KEY, "tps", 110.0, 100.0, 0.10, "improved"),
        ]

        assert _top_regressions(rows, 5) == []

    def test_sorted_and_limited_to_worst_n(self):
        rows = [
            ComparisonRow(_KEY, "ms", 130.0, 100.0, -0.30, "regressed"),
            ComparisonRow(_KEY, "ms", 120.0, 100.0, -0.20, "regressed"),
            ComparisonRow(_KEY, "ms", 110.0, 100.0, -0.10, "regressed"),
            ComparisonRow(_KEY, "ms", 100.0, 100.0, 0.0, "same"),
        ]

        top = _top_regressions(rows, 2)
        assert [r.improvement_pct for r in top] == [-0.30, -0.20]


class TestSelectBaseline:
    def test_green_only_uses_analysis_results_join(self):
        duckdb = pytest.importorskip("duckdb")
        from viewer.ingest.record import RunRecord

        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE runs (
                run_id TEXT,
                machine TEXT,
                ts TIMESTAMP,
                short_run BOOLEAN,
                purpose TEXT,
                ov_version TEXT
            )
            """
        )
        con.execute("CREATE TABLE analysis_results (run_id TEXT, overall_status TEXT)")
        con.execute(
            """
            CREATE TABLE perf (
                run_id TEXT,
                model TEXT,
                precision TEXT,
                in_token INTEGER,
                out_token INTEGER,
                exec_mode TEXT,
                value DOUBLE,
                unit TEXT
            )
            """
        )

        now = datetime(2026, 1, 2, 0, 0)
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ["old-green", "M1", now - timedelta(hours=2), True, "nightly", "ov-old-green"],
        )
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ["old-yellow", "M1", now - timedelta(hours=1), True, "nightly", "ov-old-yellow"],
        )
        con.execute("INSERT INTO analysis_results VALUES (?, ?)", ["old-green", "green"])
        con.execute("INSERT INTO analysis_results VALUES (?, ?)", ["old-yellow", "yellow"])
        con.executemany(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("old-green", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"),
                ("old-yellow", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"),
                ("current", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"),
            ],
        )

        rec = RunRecord(
            run_id="current",
            source_format="new",
            report_file="r",
            machine="M1",
            ts=now,
            short_run=True,
            purpose="nightly",
        )

        cfg = AnalysisConfig(baseline_green_only=True)
        baseline = select_baseline(con, rec, cfg)

        assert baseline.status == "found"
        assert baseline.run_id == "old-green"

    def test_skips_newer_run_without_comparable_series(self):
        duckdb = pytest.importorskip("duckdb")
        from viewer.ingest.record import RunRecord

        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE runs (
                run_id TEXT,
                machine TEXT,
                ts TIMESTAMP,
                short_run BOOLEAN,
                purpose TEXT,
                ov_version TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE perf (
                run_id TEXT,
                model TEXT,
                precision TEXT,
                in_token INTEGER,
                out_token INTEGER,
                exec_mode TEXT,
                value DOUBLE,
                unit TEXT
            )
            """
        )

        now = datetime(2026, 1, 2, 0, 0)
        con.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("old-match", "M1", now - timedelta(hours=3), True, "nightly", "ov-match"),
                ("old-no-overlap", "M1", now - timedelta(hours=1), True, "nightly", "ov-no-overlap"),
            ],
        )
        con.executemany(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("old-match", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"),
                ("old-no-overlap", "bert", "FP16", 32, 128, "2nd", 10.0, "ms"),
                ("current", "llama", "FP16", 32, 128, "2nd", 11.0, "ms"),
            ],
        )
        rec = RunRecord(
            run_id="current",
            source_format="new",
            report_file="r",
            machine="M1",
            ts=now,
            short_run=True,
            purpose="nightly",
        )

        baseline = select_baseline(con, rec, AnalysisConfig())

        assert baseline.status == "found"
        assert baseline.run_id == "old-match"


class TestFindLastKnownGood:
    def test_filters_same_run_profile(self):
        duckdb = pytest.importorskip("duckdb")
        from viewer.ingest.record import RunRecord

        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE runs (
                run_id TEXT,
                machine TEXT,
                ts TIMESTAMP,
                short_run BOOLEAN,
                purpose TEXT,
                ov_version TEXT
            )
            """
        )
        con.execute("CREATE TABLE analysis_results (run_id TEXT, overall_status TEXT)")
        con.execute(
            """
            CREATE TABLE perf (
                run_id TEXT,
                model TEXT,
                precision TEXT,
                in_token INTEGER,
                out_token INTEGER,
                exec_mode TEXT,
                value DOUBLE,
                unit TEXT
            )
            """
        )

        now = datetime(2026, 1, 2, 0, 0)
        # Should be ignored: wrong purpose.
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ["old-green-wrong-purpose", "M1", now - timedelta(hours=3), True, "adhoc", "ov-x"],
        )
        con.execute("INSERT INTO analysis_results VALUES (?, ?)", ["old-green-wrong-purpose", "green"])

        # Should be ignored: wrong short_run.
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ["old-green-wrong-short", "M1", now - timedelta(hours=2), False, "nightly", "ov-y"],
        )
        con.execute("INSERT INTO analysis_results VALUES (?, ?)", ["old-green-wrong-short", "green"])

        # Same profile + green candidates.
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ["old-green-match", "M1", now - timedelta(hours=1), True, "nightly", "ov-z"],
        )
        con.execute("INSERT INTO analysis_results VALUES (?, ?)", ["old-green-match", "green"])
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ["old-green-no-overlap", "M1", now - timedelta(minutes=30), True, "nightly", "ov-no-overlap"],
        )
        con.execute("INSERT INTO analysis_results VALUES (?, ?)", ["old-green-no-overlap", "green"])
        con.executemany(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # Current run series key
                ("current", "llama", "FP16", 32, 128, "2nd", 11.0, "ms"),
                # Comparable candidate
                ("old-green-match", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"),
                # Non-overlap candidate (newer but different model)
                ("old-green-no-overlap", "bert", "FP16", 32, 128, "2nd", 10.0, "ms"),
                # Other profile runs still comparable but filtered by profile policy
                ("old-green-wrong-purpose", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"),
                ("old-green-wrong-short", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"),
            ],
        )

        rec = RunRecord(
            run_id="current",
            source_format="new",
            report_file="r",
            machine="M1",
            ts=now,
            short_run=True,
            purpose="nightly",
        )

        lkg = find_last_known_good(con, rec)

        assert lkg.status == "found"
        assert lkg.run_id == "old-green-no-overlap"

    def test_allows_functional_only_runs_without_perf_overlap(self):
        duckdb = pytest.importorskip("duckdb")
        from viewer.ingest.record import RunRecord

        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE runs (
                run_id TEXT,
                machine TEXT,
                ts TIMESTAMP,
                short_run BOOLEAN,
                purpose TEXT,
                ov_version TEXT
            )
            """
        )
        con.execute("CREATE TABLE analysis_results (run_id TEXT, overall_status TEXT)")
        con.execute(
            """
            CREATE TABLE perf (
                run_id TEXT,
                model TEXT,
                precision TEXT,
                in_token INTEGER,
                out_token INTEGER,
                exec_mode TEXT,
                value DOUBLE,
                unit TEXT
            )
            """
        )

        now = datetime(2026, 1, 2, 0, 0)
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ["old-green", "M1", now - timedelta(hours=1), True, "nightly", "ov-green"],
        )
        con.execute("INSERT INTO analysis_results VALUES (?, ?)", ["old-green", "green"])

        rec = RunRecord(
            run_id="current-functional-only",
            source_format="new",
            report_file="r",
            machine="M1",
            ts=now,
            short_run=True,
            purpose="nightly",
        )

        lkg = find_last_known_good(con, rec)

        assert lkg.status == "found"
        assert lkg.run_id == "old-green"


class TestOverallStatus:
    def _perf(self, regressed=0, compared=5):
        return PerformanceResult(compared=compared, improved=0, same=max(compared - regressed, 0), regressed=regressed, unavailable=0)

    def _func(self, failed=0, issues=None):
        return FunctionalResult(total=5, passed=5 - failed, failed=failed, error=0, skipped=0, issues=issues or [])

    def _baseline(self, found=True):
        return BaselineInfo(status="found" if found else "not_found")

    def test_functional_fail_is_red(self):
        assert _overall_status(self._func(failed=1), self._perf(), self._baseline()) == "red"

    def test_regression_without_fail_is_yellow(self):
        assert _overall_status(self._func(), self._perf(regressed=1), self._baseline()) == "yellow"

    def test_no_baseline_is_gray(self):
        assert _overall_status(self._func(), self._perf(), self._baseline(found=False)) == "gray"

    def test_all_clear_is_green(self):
        assert _overall_status(self._func(), self._perf(), self._baseline()) == "green"

    def test_functional_fail_beats_regression(self):
        # Both issues present: red wins over yellow.
        assert _overall_status(self._func(failed=1), self._perf(regressed=2), self._baseline()) == "red"

    def test_timeout_issue_is_red(self):
        issues = [FunctionalIssue(nodeid="test_timeout", outcome="timeout", message="timed out")]
        assert _overall_status(self._func(issues=issues), self._perf(), self._baseline()) == "red"

    def test_baseline_found_but_no_comparison_is_gray(self):
        assert _overall_status(self._func(), self._perf(compared=0), self._baseline()) == "gray"


class TestFetchComparisonRows:
    def test_null_or_nan_values_marked_unavailable(self):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE perf (
                run_id TEXT,
                model TEXT,
                precision TEXT,
                in_token INTEGER,
                out_token INTEGER,
                exec_mode TEXT,
                value DOUBLE,
                unit TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["base", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"],
        )
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["cur-null", "llama", "FP16", 32, 128, "2nd", None, "ms"],
        )
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["cur-nan", "llama", "FP16", 32, 128, "2nd", float("nan"), "ms"],
        )

        baseline = BaselineInfo(status="found", run_id="base")
        cfg = AnalysisConfig()

        rows_null = _fetch_comparison_rows(con, "cur-null", baseline, cfg)
        assert len(rows_null) == 1
        assert rows_null[0].verdict == "unavailable"
        assert rows_null[0].improvement_pct is None
        assert math.isnan(rows_null[0].current_value)

        rows_nan = _fetch_comparison_rows(con, "cur-nan", baseline, cfg)
        assert len(rows_nan) == 1
        assert rows_nan[0].verdict == "unavailable"
        assert rows_nan[0].improvement_pct is None
        assert math.isnan(rows_nan[0].current_value)

    def test_unit_mismatch_is_not_compared(self):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE perf (
                run_id TEXT,
                model TEXT,
                precision TEXT,
                in_token INTEGER,
                out_token INTEGER,
                exec_mode TEXT,
                value DOUBLE,
                unit TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["base", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"],
        )
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["cur", "llama", "FP16", 32, 128, "2nd", 100.0, "tps"],
        )

        baseline = BaselineInfo(status="found", run_id="base")
        cfg = AnalysisConfig()
        rows = _fetch_comparison_rows(con, "cur", baseline, cfg)

        # Unit mismatch should produce exactly ONE unavailable row (merged).
        assert len(rows) == 1
        assert rows[0].verdict == "unavailable"

    def test_missing_current_series_is_marked_unavailable(self):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE perf (
                run_id TEXT,
                model TEXT,
                precision TEXT,
                in_token INTEGER,
                out_token INTEGER,
                exec_mode TEXT,
                value DOUBLE,
                unit TEXT
            )
            """
        )
        # Baseline has two series, current has only one.
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["base", "llama", "FP16", 32, 128, "2nd", 10.0, "ms"],
        )
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["base", "llama", "FP16", 64, 128, "2nd", 11.0, "ms"],
        )
        con.execute(
            "INSERT INTO perf VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["cur", "llama", "FP16", 32, 128, "2nd", 9.5, "ms"],
        )

        baseline = BaselineInfo(status="found", run_id="base")
        cfg = AnalysisConfig()
        rows = _fetch_comparison_rows(con, "cur", baseline, cfg)

        assert len(rows) == 2
        unavailable = [r for r in rows if r.verdict == "unavailable"]
        assert len(unavailable) == 1
        assert unavailable[0].key.in_token == 64


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

class TestRenderAnalysisSummary:
    def test_no_baseline_message(self):
        result = _make_result(baseline_found=False)
        text = render_analysis_summary(result)
        assert "[ Analysis summary ]" in text
        assert "no older run found" in text

    def test_regression_in_report(self):
        rows = [
            ComparisonRow(_KEY, "ms", 120.0, 100.0, -0.20, "regressed"),
        ]
        result = _make_result(rows=rows)
        text = render_analysis_summary(result)
        assert "regressed=1" in text
        assert "Top regressions" in text

    def test_functional_fail_verdict(self):
        result = _make_result(functional_failed=2)
        text = render_analysis_summary(result)
        assert "Functional issues detected" in text

    def test_prepend_places_block_before_existing(self, tmp_path):
        report = tmp_path / "test.report"
        report.write_text("[ Summary ]\nexisting content", encoding="utf-8")
        result = _make_result()
        prepend_to_report(report, result)
        content = report.read_text(encoding="utf-8")
        assert content.index("[ Analysis summary ]") < content.index("[ Summary ]")


# ---------------------------------------------------------------------------
# persistence.py
# ---------------------------------------------------------------------------

class TestWriteAnalysisToSummary:
    def test_analysis_block_written(self, tmp_path):
        summary_json = tmp_path / "daily.20260101_0000.summary.json"
        summary_json.write_text(
            json.dumps({"meta": {}, "totals": {}, "tests": []}),
            encoding="utf-8",
        )
        result = _make_result()
        write_analysis_to_summary(summary_json, result)

        data = json.loads(summary_json.read_text(encoding="utf-8"))
        assert "analysis" in data
        block = data["analysis"]
        assert "overall_status" in block
        assert "functional" in block
        assert "performance" in block
        assert "top_regressions" in block

    def test_config_snapshot_written(self, tmp_path):
        summary_json = tmp_path / "daily.20260101_0000.summary.json"
        summary_json.write_text(
            json.dumps({"meta": {}, "totals": {}, "tests": []}),
            encoding="utf-8",
        )
        result = _make_result()
        config = AnalysisConfig(pct_threshold=0.12, top_regressions=7)
        write_analysis_to_summary(summary_json, result, config=config)

        data = json.loads(summary_json.read_text(encoding="utf-8"))
        assert data["analysis"]["config_snapshot"]["pct_threshold"] == 0.12
        assert data["analysis"]["config_snapshot"]["top_regressions"] == 7

    def test_idempotent_overwrite(self, tmp_path):
        summary_json = tmp_path / "daily.20260101_0000.summary.json"
        summary_json.write_text(
            json.dumps({"meta": {}, "totals": {}, "tests": [], "analysis": {"old": True}}),
            encoding="utf-8",
        )
        result = _make_result()
        write_analysis_to_summary(summary_json, result)

        data = json.loads(summary_json.read_text(encoding="utf-8"))
        assert "old" not in data["analysis"]


class TestReportSummarySchema:
    def test_summary_schema_version_is_emitted(self):
        summary = build_summary({"created": 1.0, "duration": 2.0, "summary": {}, "tests": []})
        assert summary["schema_version"] == 1


class TestWriteAnalysisToDb:
    def _with_rows(self) -> "AnalysisResult":
        from analysis.types import AnalysisResult

        row = ComparisonRow(_KEY, "ms", 120.0, 100.0, -0.20, "regressed")
        return AnalysisResult(
            overall_status="red",
            baseline=BaselineInfo(status="found", run_id="base"),
            functional=FunctionalResult(
                total=1,
                passed=0,
                failed=1,
                error=0,
                skipped=0,
                issues=[FunctionalIssue(nodeid="t", outcome="failed", message="x")],
            ),
            performance=PerformanceResult(compared=1, improved=0, same=0, regressed=1, unavailable=0),
            models=[],
            top_regressions=[row],
            rows=[row],
        )

    def _empty(self) -> "AnalysisResult":
        from analysis.types import AnalysisResult

        return AnalysisResult(
            overall_status="gray",
            baseline=BaselineInfo(status="not_found"),
            functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0, issues=[]),
            performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
            models=[],
            top_regressions=[],
            rows=[],
        )

    def _timeout_only(self) -> "AnalysisResult":
        from analysis.types import AnalysisResult

        return AnalysisResult(
            overall_status="red",
            baseline=BaselineInfo(status="found", run_id="base"),
            functional=FunctionalResult(
                total=1,
                passed=0,
                failed=0,
                error=0,
                skipped=0,
                issues=[FunctionalIssue(nodeid="t", outcome="timeout", message="timed out")],
            ),
            performance=PerformanceResult(compared=1, improved=0, same=1, regressed=0, unavailable=0),
            models=[],
            top_regressions=[],
            rows=[ComparisonRow(_KEY, "ms", 10.0, 10.0, 0.0, "same")],
        )

    def test_cleans_stale_comparisons_on_empty_rerun(self, tmp_path):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        schema = (Path(__file__).resolve().parents[1] / "viewer" / "schema.sql").read_text(encoding="utf-8")
        con.execute(schema)

        write_analysis_to_db(con, "run-1", self._with_rows())
        write_analysis_to_db(con, "run-1", self._empty())

        count = con.execute(
            "SELECT count(*) FROM analysis_comparisons WHERE run_id = ?", ["run-1"]
        ).fetchone()[0]
        assert count == 0

    def test_cleans_stale_issues_on_empty_rerun(self, tmp_path):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        schema = (Path(__file__).resolve().parents[1] / "viewer" / "schema.sql").read_text(encoding="utf-8")
        con.execute(schema)

        write_analysis_to_db(con, "run-2", self._with_rows())
        write_analysis_to_db(con, "run-2", self._empty())

        count = con.execute(
            "SELECT count(*) FROM functional_issues WHERE run_id = ?", ["run-2"]
        ).fetchone()[0]
        assert count == 0

    def test_rolls_back_on_partial_failure(self):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        schema = (Path(__file__).resolve().parents[1] / "viewer" / "schema.sql").read_text(encoding="utf-8")
        con.execute(schema)

        dup_key = SeriesKey("m", "FP16", 1, 2, "2nd")
        r1 = ComparisonRow(dup_key, "ms", 12.0, 10.0, -0.2, "regressed")
        r2 = ComparisonRow(dup_key, "ms", 11.0, 10.0, -0.1, "regressed")
        from analysis.types import AnalysisResult

        broken = AnalysisResult(
            overall_status="yellow",
            baseline=BaselineInfo(status="found", run_id="base"),
            functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0, issues=[]),
            performance=PerformanceResult(compared=2, improved=0, same=0, regressed=2, unavailable=0),
            models=[],
            top_regressions=[r1, r2],
            rows=[r1, r2],
        )

        write_analysis_to_db(con, "run-rollback", broken, threshold_pct=0.05)

        results_count = con.execute(
            "SELECT count(*) FROM analysis_results WHERE run_id = ?", ["run-rollback"]
        ).fetchone()[0]
        cmp_count = con.execute(
            "SELECT count(*) FROM analysis_comparisons WHERE run_id = ?", ["run-rollback"]
        ).fetchone()[0]
        assert results_count == 0
        assert cmp_count == 0

    def test_persists_threshold_pct(self):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        schema = (Path(__file__).resolve().parents[1] / "viewer" / "schema.sql").read_text(encoding="utf-8")
        con.execute(schema)

        write_analysis_to_db(con, "run-threshold", self._with_rows(), threshold_pct=0.07)
        stored = con.execute(
            "SELECT threshold_pct FROM analysis_comparisons WHERE run_id = ? LIMIT 1", ["run-threshold"]
        ).fetchone()

        assert stored is not None
        assert abs(stored[0] - 0.07) < 1e-12

    def test_timeout_issue_counts_as_functional_issue(self):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect(":memory:")
        schema = (Path(__file__).resolve().parents[1] / "viewer" / "schema.sql").read_text(encoding="utf-8")
        con.execute(schema)

        write_analysis_to_db(con, "run-timeout", self._timeout_only(), threshold_pct=0.05)
        stored = con.execute(
            "SELECT functional_fail_count FROM analysis_results WHERE run_id = ?",
            ["run-timeout"],
        ).fetchone()

        assert stored is not None
        assert stored[0] == 1


# ---------------------------------------------------------------------------
# Integration: analyze_run end-to-end (uses tmp DuckDB)
# ---------------------------------------------------------------------------

class TestAnalyzeRunIntegration:
    """Smoke test: two runs ingested, second run analysis finds the first as baseline."""

    def _make_summary(
        self,
        tmp_path: Path,
        stamp: str,
        value: float,
        ov_version: str = "2026.0.0-100-abc1234",
    ) -> Path:
        path = tmp_path / f"daily.{stamp}.summary.json"
        path.write_text(
            json.dumps({
                "generated_at": 0.0,
                "duration_sec": 1.0,
                "meta": {
                    "machine": "TEST-01",
                    "stamp": stamp,
                    "workweek": "2026.WW01.1",
                    "ov_version": ov_version,
                    "description": "test",
                    "device": "GPU",
                },
                "totals": {"passed": 1, "failed": 0, "error": 0, "total": 1, "skipped": 0},
                "tests": [{
                    "nodeid": "test_llm",
                    "outcome": "passed",
                    "metrics": {
                        "test_type": "llm_benchmark",
                        "model": "llama",
                        "precision": "FP16",
                        "data": [{"in_token": 32, "out_token": 128, "perf": [value, value / 2]}],
                    },
                }],
            }),
            encoding="utf-8",
        )
        return path

    def test_baseline_found_and_compared(self, tmp_path):
        from analysis.engine import analyze_run

        db = tmp_path / "bench.duckdb"
        s1 = self._make_summary(tmp_path, "20260101_0000", value=100.0)
        s2 = self._make_summary(tmp_path, "20260102_0000", value=120.0)  # regression (ms unit)

        # Ingest baseline first so it exists in DB.
        analyze_run(s1, db)
        result = analyze_run(s2, db)

        assert result.baseline.status == "found"
        assert result.performance.compared > 0

    def test_no_baseline_gives_gray(self, tmp_path):
        from analysis.engine import analyze_run

        db = tmp_path / "bench.duckdb"
        s1 = self._make_summary(tmp_path, "20260101_0000", value=100.0)
        result = analyze_run(s1, db)

        assert result.baseline.status == "not_found"
        assert result.overall_status == "gray"

    def test_green_only_uses_last_green_run(self, tmp_path):
        from analysis.engine import analyze_run

        db = tmp_path / "bench.duckdb"
        cfg_green_only = AnalysisConfig(baseline_green_only=True)

        # run1: no baseline -> gray
        s1 = self._make_summary(tmp_path, "20260101_0000", value=100.0, ov_version="ov-r1")
        r1 = analyze_run(s1, db)
        assert r1.overall_status == "gray"

        # run2: same perf vs run1 -> green
        s2 = self._make_summary(tmp_path, "20260102_0000", value=100.0, ov_version="ov-r2")
        r2 = analyze_run(s2, db)
        assert r2.overall_status == "green"

        # run3: regression vs run2 -> yellow
        s3 = self._make_summary(tmp_path, "20260103_0000", value=130.0, ov_version="ov-r3")
        r3 = analyze_run(s3, db)
        assert r3.overall_status == "yellow"

        # run4: green-only baseline should skip yellow run3 and choose green run2
        s4 = self._make_summary(tmp_path, "20260104_0000", value=110.0, ov_version="ov-r4")
        r4 = analyze_run(s4, db, config=cfg_green_only)

        assert r4.baseline.status == "found"
        assert r4.baseline.ov_version == "ov-r2"

    def test_invalid_only_comparisons_do_not_turn_green(self, tmp_path):
        from analysis.engine import analyze_run

        db = tmp_path / "bench.duckdb"

        s1 = self._make_summary(tmp_path, "20260101_0000", value=100.0)
        analyze_run(s1, db)

        s2 = self._make_summary(tmp_path, "20260102_0000", value=100.0)
        data = json.loads(s2.read_text(encoding="utf-8"))
        data["tests"][0]["metrics"]["data"][0]["perf"] = ["nan", "nan"]
        s2.write_text(json.dumps(data), encoding="utf-8")

        r2 = analyze_run(s2, db)
        assert r2.baseline.status == "found"
        assert r2.performance.compared == 2
        assert r2.performance.unavailable == 2
        assert r2.overall_status == "gray"
