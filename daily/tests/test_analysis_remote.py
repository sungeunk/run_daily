from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from types import SimpleNamespace
from typing import Literal

import duckdb
import pytest

from analysis.engine import (
    _aggregate_performance,
    _fetch_comparison_rows,
    baseline_release_comparison_status,
    release_comparison_status,
)
from analysis.remote import (
    RELEASE_PURPOSE,
    RELEASE_RUN_IDS,
    _fetch_perf_rows,
    _reference_runs_sql,
    fetch_reference,
    fetch_release,
)
from analysis.types import AnalysisConfig, BaselineInfo, ComparisonRow, FunctionalResult, ReleaseInfo, SeriesKey
from common.mcp_client import McpError

pytestmark = pytest.mark.dev_only


@pytest.fixture
def perf_db() -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE perf (run_id VARCHAR, model VARCHAR, precision VARCHAR, "
            "in_token INTEGER, out_token INTEGER, exec_mode VARCHAR, unit VARCHAR, value DOUBLE)"
        )
        yield connection


def _populate(connection: duckdb.DuckDBPyConnection, count: int) -> None:
    connection.execute(
        "INSERT INTO perf SELECT 'reference', 'model-' || range, 'FP16', "
        "1024, 256, '2nd', 'ms', range::DOUBLE FROM range(?) ORDER BY range DESC",
        [count],
    )


def _query(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cursor = connection.execute(sql)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchmany(500)]


def test_reference_skips_failed_and_empty_timer_runs(
    perf_db: duckdb.DuckDBPyConnection,
) -> None:
    perf_db.execute(
        "CREATE TABLE runs_with_flags (run_id VARCHAR, machine VARCHAR, ts TIMESTAMP, "
        "ov_version VARCHAR, purpose VARCHAR, is_partial BOOLEAN, excluded BOOLEAN, "
        "total_tests INTEGER, passed_tests INTEGER, failed_tests INTEGER, error_tests INTEGER)"
    )
    perf_db.execute(
        "INSERT INTO runs_with_flags VALUES "
        "('good', 'machine', '2026-10-06 23:42:00', 'good-version', 'daily_pipeline timer', false, false, 37, 37, 0, 0), "
        "('empty', 'machine', '2026-10-07 12:00:00', 'empty-version', 'daily_pipeline timer', false, false, 37, 37, 0, 0), "
        "('skipped', 'machine', '2026-10-07 18:00:00', 'skipped-version', 'daily_pipeline timer', false, false, 37, 0, 0, 0), "
        "('failed', 'machine', '2026-10-07 23:50:00', 'failed-version', 'daily_pipeline timer', false, false, 37, 13, 24, 0), "
        "('manual', 'machine', '2026-10-08 08:00:00', 'manual-version', 'daily sungeunk', false, false, 37, 37, 0, 0)"
    )
    perf_db.execute(
        "INSERT INTO perf VALUES "
        "('good', 'model', 'FP16', 1024, 256, '2nd', 'ms', 10.0), "
        "('skipped', 'model', 'FP16', 1024, 256, '2nd', 'ms', 15.0), "
        "('failed', 'model', 'FP16', 1024, 256, '2nd', 'ms', 20.0), "
        "('manual', 'model', 'FP16', 1024, 256, '2nd', 'ms', 30.0)"
    )
    config = AnalysisConfig()
    record = SimpleNamespace(machine="machine", ts=datetime(2026, 10, 8, 9, 35))

    class Client:
        def run_sql(self, sql: str) -> list[dict]:
            return _query(perf_db, sql)

    assert [row["run_id"] for row in _query(perf_db, _reference_runs_sql(config, record))] == ["good"]
    reference = fetch_reference(config, record, client=Client())
    assert reference.info.run_id == "good"
    assert reference.values[("model", "FP16", 1024, 256, "2nd")] == (10.0, "ms")


@pytest.mark.parametrize("count", [0, 1, 199, 200, 201, 500, 600, 632])
def test_fetches_all_pages(
    perf_db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    _populate(perf_db, count)
    page_sizes: list[int] = []

    def run_sql(url: str, sql: str, *, timeout: float) -> list[dict]:
        rows = _query(perf_db, sql)
        if " OFFSET " in sql:
            assert "ORDER BY run_id, model, precision, in_token, out_token, exec_mode" in sql
            page_sizes.append(len(rows))
        return rows

    monkeypatch.setattr("common.mcp_client.run_sql", run_sql)
    rows = _fetch_perf_rows(AnalysisConfig(), ["reference"])
    assert len(rows) == count
    assert len({row["model"] for row in rows}) == count
    assert [row["model"] for row in rows] == sorted(row["model"] for row in rows)
    assert len(page_sizes) == (count + 199) // 200
    assert all(size <= 200 for size in page_sizes)


@pytest.mark.parametrize("failure", ["short", "duplicate", "changed_count", "request"])
@pytest.mark.parametrize("source", ["reference", "release"])
def test_incomplete_lookup_discards_partial_results(
    perf_db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch,
    failure: str, source: str,
) -> None:
    _populate(perf_db, 632)
    count_calls = 0

    def run_sql(url: str, sql: str, *, timeout: float) -> list[dict]:
        nonlocal count_calls
        if "FROM runs" in sql:
            return [{"run_id": "reference", "machine": "machine", "stamp": "20260908_2342"}]
        rows = _query(perf_db, sql)
        if "count(*) AS total" in sql:
            count_calls += 1
            if failure == "changed_count" and count_calls == 2:
                return [{"total": 633}]
        if "OFFSET 200" in sql:
            if failure == "short":
                return rows[:-1]
            if failure == "duplicate":
                rows[1] = rows[0]
            if failure == "request":
                raise McpError("page request failed")
        return rows

    monkeypatch.setattr("common.mcp_client.run_sql", run_sql)
    monkeypatch.setitem(RELEASE_RUN_IDS, "machine", "reference")
    config = AnalysisConfig()
    if source == "reference":
        result = fetch_reference(config, SimpleNamespace(machine="machine", ts=datetime(2026, 9, 9)))
        assert result.info.status == "unavailable"
        assert result.info.detail
        assert result.values == {}
        assert result.history == {}
    else:
        info, values = fetch_release(config, "machine")
        assert info.status == "unavailable"
        assert info.detail
        assert values == {}


@pytest.mark.parametrize("machine", ["ARLH-01", "BMG-02"])
@pytest.mark.parametrize("missing", [None, "run", "perf", "excluded", "partial", "purpose", "filter"])
def test_release_uses_only_pinned_run(
    perf_db: duckdb.DuckDBPyConnection, machine: str, missing: str | None,
) -> None:
    run_id = RELEASE_RUN_IDS[machine]
    perf_db.execute(
        "CREATE TABLE runs_with_flags (run_id VARCHAR, machine VARCHAR, ts TIMESTAMP, "
        "ov_version VARCHAR, purpose VARCHAR, is_partial BOOLEAN, excluded BOOLEAN)"
    )
    perf_db.execute(
        "INSERT INTO runs_with_flags VALUES (?, ?, '2026-10-05', '2026.4.1', ?, false, false), "
        "('newer', ?, '2026-10-07', '2026.4.1', ?, false, false)",
        [run_id, machine, RELEASE_PURPOSE, machine, RELEASE_PURPOSE],
    )
    perf_db.execute(
        "INSERT INTO perf VALUES (?, 'model', 'FP16', 1024, 256, '2nd', 'ms', 10), "
        "('newer', 'model', 'FP16', 1024, 256, '2nd', 'ms', 1)", [run_id],
    )
    config = AnalysisConfig()
    if missing == "run":
        perf_db.execute("DELETE FROM runs_with_flags WHERE run_id = ?", [run_id])
    elif missing == "perf":
        perf_db.execute("DELETE FROM perf WHERE run_id = ?", [run_id])
    elif missing == "excluded":
        perf_db.execute("UPDATE runs_with_flags SET excluded = true WHERE run_id = ?", [run_id])
    elif missing == "partial":
        perf_db.execute("UPDATE runs_with_flags SET is_partial = true WHERE run_id = ?", [run_id])
    elif missing == "purpose":
        perf_db.execute("UPDATE runs_with_flags SET purpose = 'another release' WHERE run_id = ?", [run_id])
    elif missing == "filter":
        config.release_purpose_like = "%2027%"

    queries: list[str] = []

    def run_sql(sql: str) -> list[dict]:
        queries.append(sql)
        return _query(perf_db, sql)

    info, values = fetch_release(config, machine, client=SimpleNamespace(run_sql=run_sql))
    assert info.run_id == run_id
    assert not any("ORDER BY ts" in sql for sql in queries)
    if missing:
        assert info.status == "not_found"
        assert run_id in info.detail
        assert values == {}
    else:
        assert info.status == "found"
        assert info.matched_count == 1
        assert values[("model", "FP16", 1024, 256, "2nd")] == (10.0, "ms")


@pytest.mark.parametrize("machine, enabled, status", [
    ("unregistered", True, "not_found"),
    (None, True, "not_found"),
    ("ARLH-01", False, "disabled"),
])
def test_release_without_pin_does_not_query(
    machine: str | None, enabled: bool, status: str,
) -> None:
    def run_sql(sql: str) -> list[dict]:
        pytest.fail(f"unexpected release query: {sql}")

    info, values = fetch_release(
        AnalysisConfig(release_enabled=enabled), machine, client=SimpleNamespace(run_sql=run_sql),
    )
    assert info.status == status
    assert values == {}


@pytest.mark.parametrize("delta, threshold, release_state, failures, expected", [
    (-0.06, 0.05, "found", 0, "yellow"),
    (-0.04, 0.05, "found", 0, "green"),
    (0.10, 0.05, "found", 0, "green"),
    (-0.06, 0.10, "found", 0, "green"),
    (None, 0.05, "found", 0, "gray"),
    (float("nan"), 0.05, "found", 0, "gray"),
    (float("inf"), 0.05, "found", 0, "gray"),
    (0.0, 0.05, "not_found", 0, "gray"),
    (0.0, 0.05, "unavailable", 0, "gray"),
    (0.0, 0.05, "found", 1, "red"),
    (None, 0.05, "not_found", 1, "red"),
    (0.0, 0.05, "disabled", 0, None),
])
def test_release_status_is_independent_of_baseline(
    delta: float | None, threshold: float,
    release_state: Literal["found", "not_found", "unavailable", "disabled"],
    failures: int, expected: str | None,
) -> None:
    row = ComparisonRow(
        key=SeriesKey("model", "FP16", 1024, 256, "2nd"),
        unit="ms", current_value=100.0, baseline_value=90.0,
        improvement_pct=-0.1, verdict="regressed", within_fluctuation=True,
        release_value=100.0, release_improvement_pct=delta,
    )
    functional = FunctionalResult(total=1, passed=1 - failures, failed=failures, error=0, skipped=0)
    assert release_comparison_status(
        functional, [row], ReleaseInfo(status=release_state),
        AnalysisConfig(pct_threshold=threshold),
    ) == expected
    assert row.verdict == "regressed"
    assert row.improvement_pct == -0.1


@pytest.mark.parametrize("value, release_value, unit, threshold, expected", [
    (106.0, 100.0, "ms", 0.05, "yellow"),
    (105.0, 100.0, "ms", 0.05, "yellow"),
    (104.0, 100.0, "ms", 0.05, "green"),
    (80.0, 100.0, "ms", 0.05, "green"),
    (80.0, 100.0, "FPS", 0.05, "yellow"),
    (120.0, 100.0, "FPS", 0.05, "green"),
    (106.0, 100.0, "ms", 0.10, "green"),
    (float("nan"), 100.0, "ms", 0.05, "gray"),
    (100.0, float("inf"), "ms", 0.05, "gray"),
    (100.0, 0.0, "ms", 0.05, "gray"),
])
def test_baseline_release_status_uses_baseline_values(
    value: float, release_value: float, unit: str, threshold: float, expected: str,
) -> None:
    key = ("model", "FP16", 1024, 256, "2nd")
    assert baseline_release_comparison_status(
        BaselineInfo(status="found"), {key: (value, unit)},
        ReleaseInfo(status="found"), {key: (release_value, unit)},
        AnalysisConfig(pct_threshold=threshold),
    ) == expected


@pytest.mark.parametrize("missing", [
    "baseline", "release", "baseline_values", "release_values", "key", "unit", "infer", "disabled",
])
def test_baseline_release_requires_comparable_data(missing: str) -> None:
    key = ("model", "FP16", 1024, 256, "2nd-infer" if missing == "infer" else "2nd")
    baseline = BaselineInfo(status="not_found" if missing == "baseline" else "found")
    release = ReleaseInfo(status="disabled" if missing == "disabled" else "found")
    baseline_values = {} if missing == "baseline_values" else {key: (100.0, "ms")}
    release_values = {} if missing == "release_values" else {key: (90.0, "ms")}
    if missing == "key":
        release_values = {("other", "FP16", 1024, 256, "2nd"): (90.0, "ms")}
    if missing == "unit":
        release_values = {key: (90.0, "s")}
    assert baseline_release_comparison_status(
        baseline, baseline_values, None if missing == "release" else release,
        release_values, AnalysisConfig(),
    ) == (None if missing == "disabled" else "gray")


def test_missing_reference_is_unavailable(
    perf_db: duckdb.DuckDBPyConnection,
) -> None:
    _populate(perf_db, 1)
    key = ("model-0", "FP16", 1024, 256, "2nd")
    rows = _fetch_comparison_rows(
        perf_db, "reference", AnalysisConfig(),
        reference_values={}, history_map={key: [22.0] * 6},
    )
    assert len(rows) == 1
    assert rows[0].reference_source == "no_baseline"
    assert rows[0].history_count == 6
    assert rows[0].verdict == "unavailable"
    performance = _aggregate_performance(rows)
    assert performance.unavailable == 1
    assert performance.same == 0
    assert performance.compared == 0