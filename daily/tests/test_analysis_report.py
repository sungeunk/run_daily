from __future__ import annotations

from html.parser import HTMLParser

import pytest

from analysis.report import render_analysis_html
from analysis.types import (
    AnalysisResult,
    BaselineInfo,
    ComparisonRow,
    FunctionalResult,
    PerformanceResult,
    OverallStatus,
    OutputQualityResult,
    OutputQualityRow,
    ReleaseInfo,
    SeriesKey,
)

pytestmark = pytest.mark.dev_only


class TableRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


@pytest.mark.parametrize("release_value, visible", [
    (None, False), (float("nan"), False), (float("inf"), False), (0.0, True), (22.5, True),
])
@pytest.mark.parametrize("with_rows", [False, True])
def test_release_columns_require_actual_data(
    release_value: float | None, visible: bool, with_rows: bool,
) -> None:
    rows = [
        ComparisonRow(
            key=SeriesKey("model-present", "FP16", 1024, 256, "2nd"),
            unit="ms", current_value=26.0, baseline_value=22.0,
            improvement_pct=0.0, verdict="same", release_value=release_value,
        ),
        ComparisonRow(
            key=SeriesKey("model-missing", "FP16", 274, 256, "1st"),
            unit="ms", current_value=280.0, baseline_value=276.0,
            improvement_pct=0.0, verdict="same",
        ),
    ] if with_rows else []
    result = AnalysisResult(
        overall_status="green", baseline=BaselineInfo(status="found"),
        functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=len(rows), improved=0, same=len(rows), regressed=0, unavailable=0),
        models=[], top_regressions=[], rows=rows,
        release=ReleaseInfo(status="found", matched_count=10),
    )
    rendered = render_analysis_html(result)
    parser = TableRows()
    parser.feed(rendered)
    visible = visible and with_rows
    width = 13 if visible else 11
    headers = [row for row in parser.rows if row and row[0] == "Model (?)"]
    assert len(headers) == 3
    for header in headers:
        assert len(header) == width
        assert ("Release (?)" in header) == visible
        assert ("\u0394 Release (?)" in header) == visible
    for row in parser.rows:
        if row and row[0].startswith("model-"):
            assert len(row) == width
            if visible and row[0] == "model-missing":
                assert row[7:9] == ["n/a", "n/a"]
    assert f"colspan='{width}'" in rendered
    assert f"colspan='{11 if visible else 13}'" not in rendered
    assert ("Relative change vs the release build" in rendered) == visible


@pytest.mark.parametrize("baseline_status, release_status", [
    ("green", "yellow"), ("yellow", "green"), ("red", "red"), ("green", "gray"),
])
@pytest.mark.parametrize("baseline_release_status", ["green", "yellow", "gray"])
def test_baseline_and_release_status_badges_are_separate(
    baseline_status: OverallStatus, release_status: OverallStatus,
    baseline_release_status: OverallStatus,
) -> None:
    result = AnalysisResult(
        overall_status=baseline_status, baseline=BaselineInfo(status="found"),
        functional=FunctionalResult(total=0, passed=0, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[], top_regressions=[], rows=[],
        release=ReleaseInfo(status="found"), release_status=release_status,
        baseline_release_status=baseline_release_status,
    )
    rendered = render_analysis_html(result)
    assert "Current vs Baseline" in rendered
    assert "Current vs Release" in rendered
    assert "Baseline vs Release" in rendered
    assert f'data-comparison="baseline" data-status="{baseline_status}"' in rendered
    assert f'data-comparison="release" data-status="{release_status}"' in rendered
    assert f'data-comparison="baseline-release" data-status="{baseline_release_status}"' in rendered
    assert result.overall_status == baseline_status


@pytest.mark.parametrize("disabled", [False, True])
def test_unavailable_or_disabled_release_status(disabled: bool) -> None:
    result = AnalysisResult(
        overall_status="green", baseline=BaselineInfo(status="found"),
        functional=FunctionalResult(total=0, passed=0, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[], top_regressions=[], rows=[],
        release=ReleaseInfo(status="disabled") if disabled else None,
    )
    rendered = render_analysis_html(result)
    assert "Current vs Baseline" in rendered
    assert ("Current vs Release" in rendered) == (not disabled)
    assert ("Baseline vs Release" in rendered) == (not disabled)
    if not disabled:
        assert 'data-comparison="release" data-status="gray"' in rendered
        assert 'data-comparison="baseline-release" data-status="gray"' in rendered


def test_unexpected_release_status_renders_fallback_badge() -> None:
    result = AnalysisResult(
        overall_status="green", baseline=BaselineInfo(status="found"),
        functional=FunctionalResult(total=0, passed=0, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[], top_regressions=[], rows=[], release=ReleaseInfo(status="found"),
        release_status="unexpected", baseline_release_status="unexpected",
    )

    rendered = render_analysis_html(result)

    assert 'data-comparison="release" data-status="unexpected"' in rendered
    assert 'data-comparison="baseline-release" data-status="unexpected"' in rendered
    assert 'background:#475467' in rendered


def test_quality_table_position_scores_and_escaped_preview() -> None:
    result = AnalysisResult(
        overall_status="green", baseline=BaselineInfo(status="found"),
        functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[], top_regressions=[], rows=[],
        output_quality=OutputQualityResult(rows=[
            OutputQualityRow("bad-model", "FP16", "0 / 1", "text", "warning", ["low <similarity>"],
                             baseline_score=0.1, release_score=0.2, current_preview="<script>bad()</script>"),
            OutputQualityRow("good-model", "FP16", "1 / 1", "text", "pass"),
            OutputQualityRow("missing-model", "FP16", "2 / 1", "image", "unavailable", ["image missing"]),
        ]),
    )
    rendered = render_analysis_html(result)
    assert rendered.index("<h2>Failed Tests") < rendered.index("<h2>Output Quality Checks") < rendered.index("<h2>Top Regressions")
    assert "IoU 0.100" in rendered and "IoU 0.200" in rendered
    assert "SSIM N/A" in rendered and "good-model" not in rendered
    assert "&lt;script&gt;" in rendered and "<script>bad()" not in rendered
    assert "low &lt;similarity&gt;" in rendered
    assert "PASS: 1 | WARNING: 1 | UNAVAILABLE: 1" in rendered
    assert result.overall_status == "green"


def test_quality_reference_scores_are_visible_without_overriding_output_status() -> None:
    result = AnalysisResult(
        overall_status="green", baseline=BaselineInfo(status="found"),
        functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[], top_regressions=[], rows=[],
        output_quality=OutputQualityResult(rows=[
            OutputQualityRow("legacy-model", "FP16", "0 / 1", "text", "pass", ["conditions unverified"],
                             baseline_score=1.0, release_score=0.1, baseline_comparison="unverified",
                             release_comparison="different", release_warning=True),
            OutputQualityRow("image-model", "FP16", "0 / 1", "image", "pass", ["comparison missing"]),
        ]),
    )
    rendered = render_analysis_html(result)
    assert "<th>Output Status</th>" in rendered
    assert rendered.count('data-quality-status="pass"') == 2
    assert "IoU 1.000" in rendered and "IoU 0.100" in rendered
    assert "Unverified conditions; reference only" in rendered
    assert "Different conditions; reference only" in rendered
    assert "Low similarity: 1" in rendered
    assert "SSIM N/A" in rendered and "Comparison unavailable" in rendered
    assert "PASS: 2 | WARNING: 0 | UNAVAILABLE: 0" in rendered


def test_missing_pinned_release_reason_is_visible() -> None:
    result = AnalysisResult(
        overall_status="gray", baseline=BaselineInfo(status="not_found"),
        functional=FunctionalResult(total=0, passed=0, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=0, improved=0, same=0, regressed=0, unavailable=0),
        models=[], top_regressions=[], rows=[],
        release=ReleaseInfo(status="not_found", detail="pinned release <missing>"),
    )
    rendered = render_analysis_html(result)
    assert "pinned release &lt;missing&gt;" in rendered
    assert "no release run published yet" not in rendered


def test_measurement_unit_is_html_escaped() -> None:
    result = AnalysisResult(
        overall_status="green", baseline=BaselineInfo(status="not_found"),
        functional=FunctionalResult(total=1, passed=1, failed=0, error=0, skipped=0),
        performance=PerformanceResult(compared=1, improved=0, same=1, regressed=0, unavailable=0),
        models=[], top_regressions=[],
        rows=[ComparisonRow(
            key=SeriesKey("model", "FP16", 1, 1, "1st"),
            unit='<img src=x onerror=alert(1)>', current_value=1.0, baseline_value=1.0,
            improvement_pct=0.0, verdict="same",
        )],
    )

    rendered = render_analysis_html(result)

    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<img src=x onerror=alert(1)>" not in rendered