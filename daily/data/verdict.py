"""What counts as improved, same, or regressed.

The viewer and the analysis engine both label series, and they have to agree
or the dashboard and the mail contradict each other about the same run. The
rule therefore lives here rather than in either consumer.

``config`` is accepted structurally: anything exposing the five threshold
attributes works, so ``analysis.types.AnalysisConfig`` can keep its own home
and this module stays a leaf that imports nothing from its callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from .series import lower_is_better

Verdict = Literal["improved", "same", "regressed", "noisy", "insufficient",
                  "unavailable"]


class VerdictConfig(Protocol):
    """The thresholds a verdict needs. AnalysisConfig satisfies this."""

    pct_threshold: float
    z_threshold: float
    noisy_cv_threshold: float
    min_recent_points: int
    min_baseline_points: int


@dataclass(frozen=True, slots=True)
class VerdictThresholds:
    """Standalone defaults, for callers with no AnalysisConfig to hand.

    Values mirror ``analysis.types.AnalysisConfig``; that class remains the
    one the daily run tunes from the command line.
    """

    pct_threshold: float = 0.05
    z_threshold: float = 3.0
    noisy_cv_threshold: float = 0.10
    min_recent_points: int = 5
    min_baseline_points: int = 7


def improvement_pct(current: float | None, baseline: float | None,
                    unit: str | None) -> float | None:
    """Signed improvement, positive when the current run is *better*.

    The mirror image of :func:`data.series.worsening_pct`, which is positive
    when it is worse. Both exist because the analysis layer reports
    improvement and the trend layer sorts by worst-first; keeping them
    adjacent is what stops one of them being re-derived with the wrong sign.
    """
    if current is None or baseline is None:
        return None
    if not math.isfinite(current) or not math.isfinite(baseline):
        return None
    if baseline == 0.0:
        return None
    ratio = (current - baseline) / baseline
    return -ratio if lower_is_better(unit) else ratio


def verdict_from_pct(pct: float | None, config: VerdictConfig) -> Verdict:
    """Classify on the percentage alone."""
    return verdict_from_signal(pct, config)


def verdict_from_signal(
    pct: float | None,
    config: VerdictConfig,
    *,
    worsening_z: float | None = None,
    recent_cv: float | None = None,
    recent_n: int | None = None,
    baseline_n: int | None = None,
) -> Verdict:
    """Classify one series using the threshold plus optional dual-gate signals.

    Rules:
    - invalid pct -> unavailable
    - improvement (pct >= threshold) -> improved (bypasses the gates)
    - regression (pct <= -threshold):
      - too few points -> insufficient
      - high CV -> noisy
      - z provided -> regressed if z >= threshold, else same
      - otherwise -> regressed
    - neutral (|pct| < threshold) -> same

    ``recent_cv`` must be the 1.4826-scaled ``data.stats.robust_cv``, matching
    ``noisy_cv_threshold``. Passing ``mad_ratio`` -- what
    ``trend_regressions`` reports in its ``recent_cv`` column -- understates
    the dispersion by that factor and under-reports noisy series.
    """
    if pct is None or not math.isfinite(pct):
        return "unavailable"

    # Improvement bypasses all other gates (pct-only rule).
    if pct >= config.pct_threshold:
        return "improved"

    if pct <= -config.pct_threshold:
        # Points gates first: they are the most restrictive.
        if recent_n is not None and recent_n < config.min_recent_points:
            return "insufficient"
        if baseline_n is not None and baseline_n < config.min_baseline_points:
            return "insufficient"

        if (recent_cv is not None and math.isfinite(recent_cv)
                and recent_cv >= config.noisy_cv_threshold):
            return "noisy"

        if worsening_z is not None and math.isfinite(worsening_z):
            return "regressed" if worsening_z >= config.z_threshold else "same"
        return "regressed"

    return "same"
