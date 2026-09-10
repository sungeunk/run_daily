from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from types import SimpleNamespace

import duckdb
import pytest

from analysis.engine import _aggregate_performance, _fetch_comparison_rows
from analysis.remote import _fetch_perf_rows, fetch_reference, fetch_release
from analysis.types import AnalysisConfig
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