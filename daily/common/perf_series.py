"""Naming of the LLM latency series stored in ``perf.exec_mode``.

Two families are ingested per prompt, from the same llm_bench JSON row:

``1st`` / ``2nd``
    End-to-end token latency (``first_latency`` / ``second_avg_latency``).
    This is *the* daily metric — it is what a pipeline user actually waits
    for — and the only family the regression verdict is computed from.

``1st-infer`` / ``2nd-infer``
    The infer-only slice of the same token (``first_infer_latency`` /
    ``second_infer_avg_latency``), i.e. what llm_bench accumulates in
    ``token_infer_durations``. It excludes tokenization, detokenization and
    multimodal embedding preparation, so on VLMs it can understate TTFT by
    more than an order of magnitude. It is stored so a GPU-plugin *kernel*
    regression can be seen apart from pipeline-side overhead — compare a
    ``1st`` move against its ``1st-infer`` twin: both moving points at the
    kernels, only ``1st`` moving points at the host side of the pipeline.

The infer family is deliberately kept out of the verdict: it is a diagnostic
signal, and letting it vote would both double the comparison count and let a
metric that is wrong-by-design on VLMs fail a run.
"""

from __future__ import annotations

# exec_mode values, in the order the loader emits them.
TOKEN_EXEC_MODES = ("1st", "2nd")
INFER_EXEC_MODES = ("1st-infer", "2nd-infer")

INFER_SUFFIX = "-infer"


def is_infer_exec_mode(exec_mode: str | None) -> bool:
    """True for the diagnostic infer-only series."""
    return bool(exec_mode) and exec_mode.endswith(INFER_SUFFIX)


def exclude_infer_sql(column: str = "exec_mode") -> str:
    """SQL predicate keeping only the metric families that carry a verdict.

    Written as a suffix match rather than an ``IN`` list so a future infer
    series (say ``pipeline-infer``) is excluded without a second edit here.
    """
    return f"{column} NOT LIKE '%{INFER_SUFFIX}'"
