from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import analysis.engine as engine
from analysis.types import (
    AnalysisConfig,
    AnalysisResult,
    BaselineInfo,
    FunctionalResult,
    PerformanceResult,
    ReleaseInfo,
)

pytestmark = pytest.mark.dev_only


def test_run_analysis_config_uses_output_quality_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    monkeypatch.setattr(sys, "argv", ["daily/run.py"])
    import run

    args, _remaining = run._parse_args()
    config = run._analysis_config(args)

    assert config is not None
    assert config.output_text_iou_threshold == AnalysisConfig().output_text_iou_threshold
    assert config.output_image_ssim_threshold == AnalysisConfig().output_image_ssim_threshold
    assert config.output_artifact_base_url == AnalysisConfig().output_artifact_base_url


def test_analysis_config_fallbacks_work_without_output_quality_namespace_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    monkeypatch.setattr(sys, "argv", ["daily/run.py"])
    import run

    args = SimpleNamespace(
        mcp_url="http://mcp.example/mcp",
        reference_purpose_like="%timer%",
        release=False,
        release_purpose_like="%release%",
    )
    config = run._analysis_config(args)

    assert config.output_text_iou_threshold == AnalysisConfig().output_text_iou_threshold
    assert config.output_image_ssim_threshold == AnalysisConfig().output_image_ssim_threshold
    assert config.output_artifact_base_url == AnalysisConfig().output_artifact_base_url


@pytest.mark.parametrize("failure_mode", ["call", "import"])
def test_output_quality_failure_does_not_skip_analysis_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str,
) -> None:
    summary_path = tmp_path / "daily.20261005_2349.summary.json"
    summary_path.write_text('{"tests": []}', encoding="utf-8")
    rec = SimpleNamespace(
        run_id="run-1",
        machine="LNL-03",
        ts=datetime(2026, 10, 5, 23, 49),
    )
    reference = SimpleNamespace(info=BaselineInfo(status="not_found"), values={}, history={})
    active_connections = 0
    persisted: list[str] = []

    class Connection:
        def __enter__(self):
            nonlocal active_connections
            active_connections += 1
            return self

        def __exit__(self, *_args):
            nonlocal active_connections
            active_connections -= 1

    def connect(_path):
        return Connection()

    monkeypatch.setattr("data.ingest.writer.connect", connect)
    monkeypatch.setattr("data.ingest.writer.ensure_schema", lambda _con: None)
    monkeypatch.setattr("data.ingest.writer.upsert_run", lambda _con, _rec: None)
    monkeypatch.setattr("data.ingest.loader_new.load_summary", lambda _path: rec)
    monkeypatch.setattr(engine, "aggregate_functional", lambda _summary: FunctionalResult(1, 1, 0, 0, 0))
    monkeypatch.setattr(engine, "fetch_reference", lambda *_args: reference)
    monkeypatch.setattr(engine, "fetch_release", lambda *_args: (ReleaseInfo(status="not_found"), {}))
    monkeypatch.setattr(engine, "_fetch_comparison_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(engine, "_aggregate_performance", lambda _rows: PerformanceResult(0, 0, 0, 0, 0))
    monkeypatch.setattr(engine, "_aggregate_models", lambda _rows: [])
    monkeypatch.setattr(engine, "_top_regressions", lambda _rows, _limit: [])
    monkeypatch.setattr(engine, "_overall_status", lambda *_args: "green")
    monkeypatch.setattr(engine, "_build_current_run_info", lambda _rec: None)
    monkeypatch.setattr(engine, "release_comparison_status", lambda *_args: None)
    monkeypatch.setattr(engine, "baseline_release_comparison_status", lambda *_args: None)

    def fail_quality(*_args, **_kwargs):
        assert active_connections == 0
        raise RuntimeError("archive unavailable")

    if failure_mode == "import":
        monkeypatch.setitem(sys.modules, "analysis.output_quality", None)
    else:
        monkeypatch.setattr("analysis.output_quality.quality_for_runs", fail_quality)
    monkeypatch.setattr(
        "analysis.persistence.write_analysis_to_summary",
        lambda _path, result, **_kwargs: (
            persisted.append("summary")
            if result.output_quality.detail.startswith("Output-quality checks failed")
            else pytest.fail("sanitized quality failure detail was not persisted")
        ),
    )
    monkeypatch.setattr(
        "analysis.persistence.write_analysis_to_db",
        lambda _con, _run_id, result, **_kwargs: (
            persisted.append("database")
            if active_connections == 1 and result.output_quality is None
            else pytest.fail("core DB persistence should happen before quality checks")
        ),
    )

    result = engine.analyze_run(summary_path, tmp_path / "bench.duckdb", AnalysisConfig())

    assert result.output_quality is not None
    assert result.output_quality.detail == "Output-quality checks failed; see the run log for details."
    assert persisted == ["database", "summary"]
    assert active_connections == 0
