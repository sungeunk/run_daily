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
    return verdict_from_signal(pct, config)


def verdict_from_signal(
    pct: float | None,
    config: AnalysisConfig,
    *,
    worsening_z: float | None = None,
    recent_cv: float | None = None,
    recent_n: int | None = None,
    baseline_n: int | None = None,
) -> Verdict:
    """Classify one series using threshold + optional dual-gate signals.

    Rules:
    - invalid pct -> unavailable
    - improvement (pct >= threshold) -> improved (bypasses dual-gate gates)
    - regression (pct <= -threshold):
      - if high CV -> noisy
      - if insufficient points -> insufficient
      - if z provided: regressed if z >= threshold, else same
      - else: regressed
    - neutral (|pct| < threshold) -> same
    """
    if pct is None or not math.isfinite(pct):
        return "unavailable"

    # Improvement bypasses all other gates (pct-only rule)
    if pct >= config.pct_threshold:
        return "improved"

    # For regressions, check dual-gate signals
    if pct <= -config.pct_threshold:
        # Check points gates first (most restrictive)
        if recent_n is not None and recent_n < config.min_recent_points:
            return "insufficient"
        if baseline_n is not None and baseline_n < config.min_baseline_points:
            return "insufficient"

        # Check noisy gate
        if recent_cv is not None and math.isfinite(recent_cv) and recent_cv >= config.noisy_cv_threshold:
            return "noisy"

        # Apply z-threshold if available
        if worsening_z is not None and math.isfinite(worsening_z):
            return "regressed" if worsening_z >= config.z_threshold else "same"
        return "regressed"

    # Neutral: |pct| < threshold
    return "same"


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
