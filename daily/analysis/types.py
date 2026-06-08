"""Shared data types for the analysis engine.

All modules within ``daily/analysis`` import from here; nothing outside
``analysis`` should depend on these types except for reading results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Tunable parameters for one analysis run.

    Attributes:
        pct_threshold:       Minimum absolute improvement_pct to be labelled
                             improved or regressed (default 5 %).
        z_threshold:         MAD-based z-score gate used in dual-gate mode
                             (default 3.0, i.e. Shewhart 3σ).
        noisy_cv_threshold:  Series with CV above this value are labelled
                             *noisy* and excluded from regression verdicts
                             (default 10 %).
        top_regressions:     How many worst regressions to surface in the
                             report (default 5).
        baseline_green_only: When True only consider baseline runs with
                             overall_status = 'green'; used for bisect mode.
        min_recent_points:   Minimum number of recent points required for
                             dual-gate mode (default 5).
        min_baseline_points: Minimum number of baseline points required for
                             dual-gate mode (default 7).
    """

    pct_threshold: float = 0.05
    z_threshold: float = 3.0
    noisy_cv_threshold: float = 0.10
    top_regressions: int = 5
    baseline_green_only: bool = False
    min_recent_points: int = 5
    min_baseline_points: int = 7
    history_window: int = 10
    reference_top_k: int = 5
    fluctuation_sigma_scale: float = 1.5
    baseline_purpose: str | None = None


# ---------------------------------------------------------------------------
# Keys and row types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesKey:
    """Uniquely identifies one performance series across runs."""

    model: str
    precision: str
    in_token: int
    out_token: int
    exec_mode: str


Verdict = Literal["improved", "same", "regressed", "noisy", "insufficient", "unavailable"]
OverallStatus = Literal["green", "yellow", "red", "gray"]


@dataclass
class ComparisonRow:
    """Per-series comparison result between current run and baseline run."""

    key: SeriesKey
    unit: str | None
    current_value: float
    baseline_value: float
    improvement_pct: float | None   # positive = better, None = unavailable
    verdict: Verdict
    history_count: int = 0
    reference_source: str = "baseline"
    history_median: float | None = None
    history_mad: float | None = None
    history_sigma: float | None = None
    history_cv: float | None = None
    worsening_z: float | None = None
    within_fluctuation: bool = False


@dataclass
class FunctionalIssue:
    """One failing or erroring test from the current run."""

    nodeid: str
    outcome: str          # 'failed' | 'error' | 'timeout'
    message: str          # short normalised message


# ---------------------------------------------------------------------------
# Sub-result blocks
# ---------------------------------------------------------------------------

@dataclass
class BaselineInfo:
    """Metadata about the baseline run that was selected for comparison."""

    status: Literal["found", "not_found"]
    run_id: str | None = None
    stamp: str | None = None
    ov_version: str | None = None
    selection_reason: str | None = None   # human-readable why this run was picked


@dataclass
class FunctionalResult:
    """Aggregate of functional test outcomes for the current run."""

    total: int
    passed: int
    failed: int
    error: int
    skipped: int
    issues: list[FunctionalIssue] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        """Return failures/errors plus explicit timeout-only issues."""
        timeout_count = sum(1 for issue in self.issues if issue.outcome == "timeout")
        return self.failed + self.error + timeout_count


@dataclass
class PerformanceResult:
    """Aggregate of run-to-run performance comparisons."""

    compared: int
    improved: int
    same: int
    regressed: int
    unavailable: int


@dataclass
class ModelSummary:
    """Per-model aggregate across all series."""

    model: str
    avg_improvement_pct: float | None
    improved: int
    same: int
    regressed: int


@dataclass
class CurrentRunInfo:
    """Current run metadata shown in the HTML summary."""

    ov_version: str | None = None
    purpose: str | None = None
    machine_name: str | None = None
    gpu_driver_version: str | None = None
    gpu_info: str | None = None
    host_info: str | None = None
    memory_size: str | None = None
    memory_speed: str | None = None


@dataclass
class BisectDelta:
    """Issue-run vs last-known-good summary for bisect assistance."""

    status: Literal["available", "unavailable"]
    issue_run_id: str
    issue_stamp: str | None
    issue_ov_version: str | None
    issue_ov_build: str | None
    issue_ov_sha: str | None
    last_good_run_id: str | None
    last_good_stamp: str | None
    last_good_ov_version: str | None
    last_good_ov_build: str | None
    last_good_ov_sha: str | None
    compared_count: int
    regressed_count: int
    functional_issue_count: int
    build_changed: bool | None
    sha_changed: bool | None


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Complete analysis result for one run.

    This is the contract between the engine and all consumers: report
    renderer, persistence layer, mail template, and dashboard queries.
    """

    overall_status: OverallStatus
    baseline: BaselineInfo
    functional: FunctionalResult
    performance: PerformanceResult
    models: list[ModelSummary]
    top_regressions: list[ComparisonRow]
    rows: list[ComparisonRow]           # full comparison table (not in JSON output)
    current_run: CurrentRunInfo | None = None
    last_known_good: BaselineInfo | None = None
    bisect_delta: BisectDelta | None = None
