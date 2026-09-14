"""The statistics the daily signal is built from.

Robust estimators throughout: a benchmark fleet produces occasional wild
single points, and a median/MAD pair ignores them where a mean/stddev pair
would report them as a regression.

Two things this module exists to stop:

*Geomean was written four times* -- ``statistics.geometric_mean`` in
``common/delivery.py``, SQL ``exp(avg(ln(value)))`` in
``queries.legacy_geomean_summary``, ``np.exp(np.log(v).mean())`` in
``queries.geomean_for_runs``, and again in ``parsers/llm_benchmark.py``.

*Two different quantities were both called "cv"* --
``analysis/engine.py`` used ``1.4826 * mad / |median|`` while
``queries.trend_regressions`` used ``mad / median``. They differ by a factor
of 1.4826, they were compared against the same thresholds, and nothing in
either name said which one you had. They are both kept below, under names
that cannot be confused, so a caller has to choose deliberately.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Sequence

from .series import lower_is_better

#: Scale factor making the MAD a consistent estimator of the standard
#: deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826


def _finite(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


# ---------------------------------------------------------------------------
# Central tendency
# ---------------------------------------------------------------------------

def geomean(values: Sequence[float]) -> float | None:
    """Geometric mean of the positive, finite values, or None if there are none.

    Non-positive values are dropped rather than poisoning the log: a latency
    of 0 or -1 is llm_bench's "not measured", not a real measurement.
    """
    usable = [v for v in _finite(values) if v > 0.0]
    if not usable:
        return None
    return math.exp(sum(math.log(v) for v in usable) / len(usable))


def geomean_sql(column: str = "value") -> str:
    """SQL form of :func:`geomean`, with the same positive-only guard."""
    return f"exp(avg(ln({column}))) FILTER (WHERE {column} > 0)"


def safe_median(values: Sequence[float]) -> float | None:
    usable = _finite(values)
    return median(usable) if usable else None


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

def mad(values: Sequence[float]) -> float | None:
    """Median absolute deviation."""
    usable = _finite(values)
    if not usable:
        return None
    centre = median(usable)
    return median([abs(v - centre) for v in usable])


def robust_sigma(values: Sequence[float]) -> float | None:
    """MAD rescaled to a standard-deviation-like quantity."""
    dispersion = mad(values)
    return None if dispersion is None else MAD_TO_SIGMA * dispersion


def mad_ratio(values: Sequence[float]) -> float | None:
    """``MAD / median`` -- dispersion as a fraction of the typical value.

    This is what ``trend_regressions`` reports in its ``recent_cv`` column and
    what the exploratory trend workflow thresholds on. Despite the historical
    column name it is NOT a coefficient of variation: no 1.4826, no stddev.
    Use :func:`robust_cv` if you want the rescaled one.
    """
    centre = safe_median(values)
    dispersion = mad(values)
    if centre in (None, 0.0) or dispersion is None:
        return None
    return dispersion / abs(centre)


def robust_cv(values: Sequence[float]) -> float | None:
    """``1.4826 * MAD / |median|`` -- the stddev-scaled coefficient of variation.

    This is what the analysis engine's fluctuation guard uses. It is
    :func:`mad_ratio` times 1.4826; the two are not interchangeable in a
    threshold comparison.
    """
    ratio = mad_ratio(values)
    return None if ratio is None else MAD_TO_SIGMA * ratio


# ---------------------------------------------------------------------------
# Change
# ---------------------------------------------------------------------------

def robust_z(current: float | None, baseline_values: Sequence[float],
             unit: str | None) -> float | None:
    """How many robust sigmas worse than its own history a point is.

    Signed so positive always means worse, whichever way ``unit`` points.
    Returns None when the baseline has no spread to measure against -- a zero
    MAD makes every deviation infinitely significant, which is never the
    answer we want.
    """
    if current is None:
        return None
    centre = safe_median(baseline_values)
    sigma = robust_sigma(baseline_values)
    if centre is None or sigma is None or sigma <= 0.0 or not math.isfinite(sigma):
        return None
    delta = current - centre if lower_is_better(unit) else centre - current
    return delta / sigma
