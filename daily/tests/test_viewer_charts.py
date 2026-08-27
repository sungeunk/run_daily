"""Tests for the viewer's chart y-axis scaling."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

DAILY_DIR = Path(__file__).resolve().parent.parent
if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

pytestmark = pytest.mark.dev_only

# app.py runs Streamlit at import time only under `streamlit run`; importing the
# module itself is safe and gives access to the pure helpers.
from viewer.app import (DEFAULT_Y_SCALE, Y_SCALE_OPTIONS,  # noqa: E402
                        _newest_first, _stable_y_range)


class TestStableYRange:
    def test_flat_series_gets_minimum_span(self):
        # 998..1003 on a 1000 ms series is scatter, not a regression: the axis
        # must not zoom in until it fills the plot.
        values = pd.Series([998.0, 1003.0, 1000.0, 1001.0])
        low, high = _stable_y_range(values, 0.05)
        assert high - low == pytest.approx(1000.5 * 0.10, rel=1e-6)
        assert low < 998.0 and high > 1003.0

    def test_real_move_expands_beyond_minimum(self):
        values = pd.Series([1000.0, 1400.0])
        low, high = _stable_y_range(values, 0.05)
        # Data span (400) dominates the 10% floor (120).
        assert high - low == pytest.approx(400 * 1.10, rel=1e-6)
        assert low <= 1000.0 and high >= 1400.0

    def test_tighter_scale_gives_smaller_span(self):
        values = pd.Series([1000.0, 1001.0])
        tight = _stable_y_range(values, 0.01)
        wide = _stable_y_range(values, 0.25)
        assert (tight[1] - tight[0]) < (wide[1] - wide[0])

    def test_auto_defers_to_plotly(self):
        assert _stable_y_range(pd.Series([1.0, 2.0]), "auto") is None

    def test_zero_based_starts_at_zero(self):
        low, high = _stable_y_range(pd.Series([10.0, 20.0]), "zero")
        assert low == 0.0
        assert high >= 20.0

    def test_positive_series_never_goes_negative(self):
        low, _ = _stable_y_range(pd.Series([1.0, 1.1]), 0.25)
        assert low >= 0.0

    def test_empty_series_has_no_range(self):
        assert _stable_y_range(pd.Series([], dtype="float64"), 0.05) is None

    def test_all_nan_series_has_no_range(self):
        assert _stable_y_range(pd.Series([None, None], dtype="float64"),
                               0.05) is None


class TestScaleOptions:
    def test_default_is_available(self):
        assert DEFAULT_Y_SCALE in Y_SCALE_OPTIONS

    def test_every_option_produces_a_usable_setting(self):
        values = pd.Series([100.0, 101.0])
        for label, scale in Y_SCALE_OPTIONS.items():
            result = _stable_y_range(values, scale)
            if scale == "auto":
                assert result is None, label
            else:
                assert result is not None and result[1] > result[0], label


class TestNewestFirst:
    def test_orders_by_timestamp_descending(self):
        frame = pd.DataFrame({
            "ts": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-02"]),
            "stamp": ["a", "c", "b"],
        })
        assert list(_newest_first(frame)["stamp"]) == ["c", "b", "a"]

    def test_falls_back_to_stamp_when_ts_absent(self):
        frame = pd.DataFrame({"stamp": ["20260101_0100", "20260103_0100",
                                        "20260102_0100"]})
        assert list(_newest_first(frame)["stamp"]) == [
            "20260103_0100", "20260102_0100", "20260101_0100"]

    def test_reverses_when_no_time_column(self):
        frame = pd.DataFrame({"x": [1, 2, 3]})
        assert list(_newest_first(frame)["x"]) == [3, 2, 1]

    def test_empty_frame_is_returned_unchanged(self):
        frame = pd.DataFrame(columns=["ts"])
        assert _newest_first(frame).empty
