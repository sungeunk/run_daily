"""How many series a run should produce, and how many it did.

These two halves are one contract, and they used to live four files apart:
the pytest cases each computed ``expected_series`` their own way, and the
viewer, the fleet digest and the mail subject each counted rows in ``perf``
their own way. Nothing tied them together, so when a7852ea started storing
the ``*-infer`` family the "did" side grew and the "should" side did not.
Four call sites reported success above total at once, and because
``series_failed`` is ``max(0, total - skipped - success)`` the overflow
clamped to zero and hid every genuinely missing series instead of showing it.

They are in one module now so that adding a family forces you past both
halves.

The invariant, on a run that measured everything it meant to:

    expected_cases == skipped_cases + count_success_series(...)

``expected_cases`` counts benchmark *cases* -- one LLM test covers N prompts
times 1st/2nd -- not pytest functions, so it is summed from each test's
``expected_series`` rather than counted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .series import TOKEN_EXEC_MODES, exclude_infer_sql

#: exec_mode values one LLM prompt yields that count toward the total.
#: The infer twins are diagnostics and are deliberately not in here.
LLM_SERIES_PER_PROMPT = len(TOKEN_EXEC_MODES)


# ---------------------------------------------------------------------------
# Producers -- what a test declares before it runs
# ---------------------------------------------------------------------------

def count_prompts(prompt_path: str | Path) -> int:
    """Non-blank lines in a JSONL prompt file, or 0 if it cannot be read.

    A missing prompt file means the case will not run; reporting 0 keeps it
    out of the expectation instead of inventing series that never arrive.
    """
    try:
        with open(prompt_path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def expected_series_for_llm(prompt_path: str | Path) -> int:
    """One series per prompt for each token family (1st and 2nd)."""
    return count_prompts(prompt_path) * LLM_SERIES_PER_PROMPT


def expected_series_for_image_gen(prompt_path: str | Path,
                                  prompt_index: int | None = None) -> int:
    """One pipeline timing per prompt run; a pinned index runs exactly one."""
    if prompt_index is not None:
        return 1
    return count_prompts(prompt_path)


def expected_series_for_app() -> int:
    """benchmark_app reports a single throughput number per case."""
    return 1


def expected_cases(summary: Mapping, outcomes: Iterable[str] | None = None) -> int:
    """Sum ``expected_series`` over a summary's tests, optionally by outcome."""
    wanted = set(outcomes) if outcomes is not None else None
    total = 0
    for test in summary.get("tests", []) or []:
        if wanted is not None and test.get("outcome") not in wanted:
            continue
        total += int((test.get("metrics") or {}).get("expected_series") or 0)
    return total


# ---------------------------------------------------------------------------
# Consumer -- what actually landed
# ---------------------------------------------------------------------------

def success_series_sql(run_id_placeholders: str, *,
                       include_infer: bool = False,
                       distinct: bool = True) -> str:
    """Query counting the series each run produced, keyed by ``run_id``.

    ``include_infer`` defaults to False because every expectation this is
    weighed against counts the token family only. Pass True only to count raw
    stored rows, never to compare against ``expected_cases``.

    ``distinct`` de-duplicates on the full series key plus ``prompt_idx``,
    which is what the fleet digest wants; the viewer's legacy "Success count"
    row counts stored rows instead.
    """
    infer_clause = "" if include_infer else f" AND {exclude_infer_sql()}"
    if not distinct:
        return (f"SELECT run_id, count(*) AS series_count FROM perf "
                f"WHERE run_id IN ({run_id_placeholders}){infer_clause} "
                f"GROUP BY run_id")
    return (
        "SELECT run_id, count(*) AS series_count FROM ("
        "  SELECT DISTINCT run_id, model, precision, in_token, out_token,"
        "                  exec_mode, prompt_idx"
        f"  FROM perf WHERE run_id IN ({run_id_placeholders}){infer_clause}"
        ") GROUP BY run_id"
    )


def success_series_scalar_sql(run_id_expr: str, *,
                              alias: str = "p",
                              include_infer: bool = False) -> str:
    """Correlated scalar subquery form, for a per-row count inside a bigger
    query where the batch form doesn't fit."""
    infer_clause = ("" if include_infer
                    else f" AND {exclude_infer_sql(f'{alias}.exec_mode')}")
    return (f"(SELECT count(*) FROM perf {alias} "
            f"WHERE {alias}.run_id = {run_id_expr}{infer_clause})")


def count_success_series(con, run_ids: Sequence[str], *,
                         include_infer: bool = False,
                         distinct: bool = True) -> dict[str, int]:
    """Series produced per run, as ``{run_id: count}``.

    Runs that produced nothing are absent from the mapping rather than
    present with 0, matching what the callers already expected.
    """
    ids = [str(run_id) for run_id in run_ids]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    sql = success_series_sql(placeholders, include_infer=include_infer,
                             distinct=distinct)
    return {str(run_id): int(count)
            for run_id, count in con.execute(sql, ids).fetchall()}


def failed_series(expected: int | None, skipped: int | None,
                  success: int | None) -> int:
    """Series that were expected, not skipped, and never arrived.

    Clamped at zero because a negative count is meaningless -- but the clamp
    is exactly what hid the infer overflow, so :mod:`daily.data.validate`
    reports the overflow separately instead of letting this swallow it.
    """
    return max(0, int(expected or 0) - int(skipped or 0) - int(success or 0))


def counts_are_consistent(expected: int | None, skipped: int | None,
                          success: int | None) -> bool:
    """False when success exceeds what the run could possibly have produced."""
    return int(success or 0) + int(skipped or 0) <= int(expected or 0)
