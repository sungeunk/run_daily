from __future__ import annotations

import importlib.util
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.types import (
    AnalysisConfig,
    AnalysisResult,
    BaselineInfo,
    FunctionalResult,
    PerformanceResult,
    ReleaseInfo,
    OutputQualityResult,
)

pytestmark = pytest.mark.dev_only


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


def test_mcp_analysis_inspects_selected_run_artifacts(report_generator, monkeypatch) -> None:
    from analysis.remote import ReferenceResult
    baseline = BaselineInfo(status="found", stamp="20261004_2349")
    release = ReleaseInfo(status="found", stamp="20261005_1032")
    monkeypatch.setattr(report_generator, "_build_functional_from_mcp", lambda *args, **kwargs: FunctionalResult(0, 0, 0, 0, 0))
    monkeypatch.setattr(report_generator, "_build_current_run_info", lambda *args, **kwargs: None)
    monkeypatch.setattr("analysis.remote.fetch_reference", lambda *args, **kwargs: ReferenceResult(baseline))
    monkeypatch.setattr("analysis.remote.fetch_release", lambda *args, **kwargs: (release, {}))
    monkeypatch.setattr("analysis.remote.fetch_series_values", lambda *args, **kwargs: {})
    calls: list[tuple] = []
    quality = OutputQualityResult(detail="test result")
    def inspect(*args):
        calls.append(args)
        return quality
    monkeypatch.setattr("analysis.output_quality.quality_for_runs", inspect)
    config = AnalysisConfig(output_text_iou_threshold=0.4)
    result = report_generator._analyze_run_from_mcp(
        config, {"run_id": "run", "machine": "PTLH-01", "ts": "2026-10-05T23:49:00"}, client=SimpleNamespace(),
    )
    assert calls == [(config, "PTLH-01", "20261005_2349", "20261004_2349", "20261005_1032")]
    assert result.output_quality is quality


@pytest.mark.parametrize("failure_mode", ["call", "import"])
def test_mcp_output_quality_failure_does_not_abort_report_analysis(
    report_generator, monkeypatch, failure_mode: str,
) -> None:
    from analysis.remote import ReferenceResult

    baseline = BaselineInfo(status="not_found")
    release = ReleaseInfo(status="not_found")
    monkeypatch.setattr(report_generator, "_build_functional_from_mcp", lambda *_args, **_kwargs: FunctionalResult(0, 0, 0, 0, 0))
    monkeypatch.setattr(report_generator, "_build_current_run_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("analysis.remote.fetch_reference", lambda *_args, **_kwargs: ReferenceResult(baseline))
    monkeypatch.setattr("analysis.remote.fetch_release", lambda *_args, **_kwargs: (release, {}))
    monkeypatch.setattr("analysis.remote.fetch_series_values", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("analysis.engine.build_comparison_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("analysis.engine._aggregate_performance", lambda _rows: PerformanceResult(0, 0, 0, 0, 0))
    monkeypatch.setattr("analysis.engine._aggregate_models", lambda _rows: [])
    monkeypatch.setattr("analysis.engine._top_regressions", lambda *_args: [])
    monkeypatch.setattr("analysis.engine._overall_status", lambda *_args: "green")
    if failure_mode == "import":
        monkeypatch.setitem(sys.modules, "analysis.output_quality", None)
    else:
        monkeypatch.setattr(
            "analysis.output_quality.quality_for_runs",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("archive unavailable")),
        )

    result = report_generator._analyze_run_from_mcp(
        AnalysisConfig(),
        {"run_id": "run-1", "machine": "LNL-03", "ts": "2026-10-05T23:49:00"},
        client=SimpleNamespace(),
    )

    assert result.overall_status == "green"
    assert result.output_quality is not None
    assert result.output_quality.detail == "Output-quality checks failed; see the run log for details."


@pytest.mark.parametrize("threshold", ["-1", "nan", "1.1"])
def test_cli_rejects_invalid_quality_threshold(report_generator, threshold: str) -> None:
    with pytest.raises(SystemExit) as failure:
        report_generator.main(["--output-text-iou-threshold", threshold])
    assert failure.value.code == 2


def test_cli_rejects_invalid_artifact_base_url(report_generator) -> None:
    with pytest.raises(SystemExit) as failure:
        report_generator.main(["--output-artifact-base-url", "file:///tmp/artifacts"])

    assert failure.value.code == 2




def test_daily_run_rejects_non_numeric_quality_threshold_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily_dir = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(daily_dir))
    monkeypatch.setenv("DAILY_OUTPUT_TEXT_IOU_THRESHOLD", "high")
    monkeypatch.setattr(sys, "argv", ["daily/run.py"])
    daily_run = importlib.import_module("run")

    with pytest.raises(SystemExit) as failure:
        daily_run._parse_args()

    assert failure.value.code == 2


def test_daily_run_rejects_invalid_artifact_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily_dir = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(daily_dir))
    monkeypatch.setenv("DAILY_OUTPUT_ARTIFACT_BASE_URL", "file:///tmp/artifacts")
    monkeypatch.setattr(sys, "argv", ["daily/run.py"])
    daily_run = importlib.import_module("run")

    with pytest.raises(SystemExit) as failure:
        daily_run._parse_args()

    assert failure.value.code == 2


@pytest.mark.parametrize("env_name", [
    "DAILY_OUTPUT_TEXT_IOU_THRESHOLD",
    "DAILY_OUTPUT_IMAGE_SSIM_THRESHOLD",
])
def test_cli_rejects_non_numeric_quality_threshold_env(
    report_generator, monkeypatch: pytest.MonkeyPatch, env_name: str,
) -> None:
    monkeypatch.setenv(env_name, "high")

    with pytest.raises(SystemExit) as failure:
        report_generator.main([])

    assert failure.value.code == 2