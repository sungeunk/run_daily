"""Verdict assignment for individual performance series.

Extracted from ``run.py::_trend_verdict``.  Two modes:

* **Simple threshold** (MVP): purely pct-based, always available.
* **Dual gate** (future): requires worsening_z from a trend query; used
  when enough historical points exist.
"""

from __future__ import annotations

from data.verdict import (Verdict, improvement_pct, verdict_from_pct,
                          verdict_from_signal)

from .types import AnalysisConfig, ComparisonRow, SeriesKey


__all__ = ["Verdict", "improvement_pct", "verdict_from_pct",
           "verdict_from_signal", "make_comparison_row"]


def make_comparison_row(
    key: SeriesKey,
    unit: str | None,
    current_value: float,
    baseline_value: float,
    config: AnalysisConfig,
    *,
    worsening_z: float | None = None,
    recent_cv: float | None = None,
    recent_n: int | None = None,
    baseline_n: int | None = None,
) -> ComparisonRow:
    """Build a fully populated :class:`ComparisonRow` for one series."""
    pct = improvement_pct(current_value, baseline_value, unit)
    return ComparisonRow(
        key=key,
        unit=unit,
        current_value=current_value,
        baseline_value=baseline_value,
        improvement_pct=pct,
        verdict=verdict_from_signal(
            pct,
            config,
            worsening_z=worsening_z,
            recent_cv=recent_cv,
            recent_n=recent_n,
            baseline_n=baseline_n,
        ),
    )
