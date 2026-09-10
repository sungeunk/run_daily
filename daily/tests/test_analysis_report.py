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