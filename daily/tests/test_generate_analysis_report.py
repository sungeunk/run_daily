from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from analysis.types import (
    AnalysisConfig,
    AnalysisResult,
    BaselineInfo,
    FunctionalResult,
    PerformanceResult,
)


@pytest.fixture
def report_generator():
    script = Path(__file__).parents[2] / "scripts" / "generate_analysis_report.py"
    spec = importlib.util.spec_from_file_location("generate_analysis_report", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_generates_html_from_mcp_only(report_generator, monkeypatch, tmp_path: Path) -> None:
    selected_run = {
        "run_id": "run-1",
        "ts": "2026-09-14T00:11:00",
        "machine": "MTL-01",
    }
    calls: list[tuple[str | None, str | None, str | None]] = []

    def pick_run(config, *, machine, run_id, stamp, client):
        calls.append((machine, run_id, stamp))
        return selected_run

    monkeypatch.setattr(report_generator, "_pick_run_from_mcp", pick_run)
    result = AnalysisResult(
        overall_status="green",
        baseline=BaselineInfo(status="not_found"),
        functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[],
        top_regressions=[],
        rows=[],
    )
    monkeypatch.setattr(
        report_generator, "_analyze_run_from_mcp", lambda _config, _run, *, client: result
    )
    class Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr("common.mcp_client.McpHttpClient", Client)
    assert report_generator.main([
        "--mcp-url", "http://daily.example/mcp",
        "--machine", "MTL-01",
        "--out-dir", str(tmp_path),
    ]) == 0

    reports = list(tmp_path.glob("analysis.current_20260914_0011.generated_*.html"))
    assert len(reports) == 1
    html = reports[0].read_text(encoding="utf-8")
    assert "MCP report" not in html
    assert "<th>Success</th>" in html
    assert "<th>Fail</th>" in html
    assert ">1<" in html
    assert calls == [("MTL-01", None, None)]


def test_pick_run_queries_daily_results_mcp(report_generator, monkeypatch) -> None:
    queries: list[str] = []

    def run_sql(_url: str, sql: str, *, timeout: float) -> list[dict]:
        queries.append(sql)
        return [{"run_id": "run-1"}]

    monkeypatch.setattr("common.mcp_client.run_sql", run_sql)
    config = AnalysisConfig(mcp_url="http://daily.example/mcp")

    result = report_generator._pick_run_from_mcp(
        config, machine="LNL-03", run_id=None, stamp="20260913_2358"
    )

    assert result == {"run_id": "run-1"}
    assert len(queries) == 1
    assert "FROM runs" in queries[0]
    assert "machine = 'LNL-03'" in queries[0]
    assert "strftime(ts, '%Y%m%d_%H%M') = '20260913_2358'" in queries[0]