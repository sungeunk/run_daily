"""Shared semantics for the daily benchmark data.

Everything that reads or writes daily results -- the pytest suite, the
viewer, the analysis engine, the MCP server, the fleet report -- goes through
this package, so that "what is a series", "which runs count", "which way is
worse" and "how many should there be" have exactly one answer each.

Import from the package root rather than the submodules; the split between
``series``/``filters``/``stats``/``counts`` is an implementation detail.
"""

from __future__ import annotations

from .counts import (
    LLM_SERIES_PER_PROMPT,
    count_prompts,
    count_success_series,
    counts_are_consistent,
    expected_cases,
    expected_series_for_app,
    expected_series_for_image_gen,
    expected_series_for_llm,
    failed_series,
    success_series_scalar_sql,
    success_series_sql,
)
from .filters import (
    DEFAULT_RUN_KINDS,
    RUN_KINDS,
    RunScope,
    classify_run_kind,
    has_column,
    has_table,
    parse_triggered_by,
    run_kind_predicate,
    tables,
)
from .series import (
    INFER_EXEC_MODES,
    INFER_SUFFIX,
    LOWER_IS_BETTER_UNITS,
    SERIES_KEY_COLUMNS,
    SHORT_TOKEN_MAX,
    TOKEN_EXEC_MODES,
    SeriesKey,
    direction_label,
    direction_label_sql,
    direction_sign,
    exclude_infer_sql,
    infer_twin,
    is_infer_exec_mode,
    lower_is_better,
    normalize_unit,
    normalize_unit_sql,
    normalize_value,
    normalize_value_sql,
    series_key_sql,
    token_bucket,
    worsening_pct,
)
from .validate import Finding, check_run
from .verdict import (
    Verdict,
    VerdictThresholds,
    improvement_pct,
    verdict_from_pct,
    verdict_from_signal,
)
from .stats import (
    MAD_TO_SIGMA,
    geomean,
    geomean_sql,
    mad,
    mad_ratio,
    robust_cv,
    robust_sigma,
    robust_z,
    safe_median,
)

__all__ = [
    # series
    "INFER_EXEC_MODES", "INFER_SUFFIX", "LOWER_IS_BETTER_UNITS",
    "SERIES_KEY_COLUMNS", "SHORT_TOKEN_MAX", "TOKEN_EXEC_MODES", "SeriesKey",
    "direction_label", "direction_label_sql", "direction_sign",
    "exclude_infer_sql", "infer_twin", "is_infer_exec_mode", "lower_is_better",
    "normalize_unit", "normalize_unit_sql", "normalize_value",
    "normalize_value_sql", "series_key_sql", "token_bucket", "worsening_pct",
    # filters
    "DEFAULT_RUN_KINDS", "RUN_KINDS", "RunScope", "classify_run_kind",
    "has_column", "has_table", "parse_triggered_by", "run_kind_predicate",
    "tables",
    # stats
    "MAD_TO_SIGMA", "geomean", "geomean_sql", "mad", "mad_ratio", "robust_cv",
    "robust_sigma", "robust_z", "safe_median",
    # counts
    "LLM_SERIES_PER_PROMPT", "count_prompts", "count_success_series",
    "counts_are_consistent", "expected_cases", "expected_series_for_app",
    "expected_series_for_image_gen", "expected_series_for_llm",
    "failed_series", "success_series_scalar_sql", "success_series_sql",
    # verdict
    "Verdict", "VerdictThresholds", "improvement_pct",
    "verdict_from_pct", "verdict_from_signal",
    # validate
    "Finding", "check_run",
]
