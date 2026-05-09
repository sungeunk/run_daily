"""Verdict assignment for individual performance series.

Extracted from ``run.py::_trend_verdict``.  Two modes:

* **Simple threshold** (MVP): purely pct-based, always available.
* **Dual gate** (future): requires worsening_z from a trend query; used
  when enough historical points exist.
"""

from __future__ import annotations

import math

from .types import AnalysisConfig, ComparisonRow, SeriesKey, Verdict

# Units where *lower* measured value means *better* performance.
_LOWER_IS_BETTER = {"ms", "s", "%"}


def improvement_pct(
    current: float | None,
    baseline: float | None,
    unit: str | None,
) -> float | None:
    """Return signed improvement percentage.

    Positive means the current run is *better* than baseline regardless
    of the metric direction.  Returns *None* when baseline is zero or
    unavailable.
    """
    if current is None or baseline is None:
        return None
    if not math.isfinite(current) or not math.isfinite(baseline):
        return None
    if baseline == 0.0:
        return None
    ratio = (current - baseline) / baseline
    if unit in _LOWER_IS_BETTER:
        return -ratio   # lower current -> positive improvement
    return ratio        # higher current -> positive improvement


def verdict_from_pct(pct: float | None, config: AnalysisConfig) -> Verdict:
    """Classify *pct* into improved / same / regressed using simple threshold."""
    if pct is None or not math.isfinite(pct):
        return "unavailable"
    if pct >= config.pct_threshold:
        return "improved"
    if pct <= -config.pct_threshold:
        return "regressed"
    return "same"


def make_comparison_row(
    key: SeriesKey,
    unit: str | None,
    current_value: float,
    baseline_value: float,
    config: AnalysisConfig,
) -> ComparisonRow:
    """Build a fully populated :class:`ComparisonRow` for one series."""
    pct = improvement_pct(current_value, baseline_value, unit)
    return ComparisonRow(
        key=key,
        unit=unit,
        current_value=current_value,
        baseline_value=baseline_value,
        improvement_pct=pct,
        verdict=verdict_from_pct(pct, config),
    )
