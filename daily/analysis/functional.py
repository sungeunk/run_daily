"""Functional test outcome aggregation.

Reads ``totals`` and ``tests`` from a ``summary.json`` dict and produces
a :class:`~analysis.types.FunctionalResult`.
"""

from __future__ import annotations

from .types import FunctionalIssue, FunctionalResult

_ISSUE_OUTCOMES = {"failed", "error", "timeout"}
_MAX_MESSAGE_LEN = 200


def _shorten(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) > _MAX_MESSAGE_LEN:
        return text[:_MAX_MESSAGE_LEN] + "…"
    return text


def aggregate_functional(summary: dict) -> FunctionalResult:
    """Extract functional pass/fail totals and individual issue list.

    Args:
        summary: Parsed content of ``summary.json`` (a plain ``dict``).

    Returns:
        A :class:`FunctionalResult` with counts and up to N issue records.
    """
    totals = summary.get("totals", {})

    total   = int(totals.get("total",   0))
    passed  = int(totals.get("passed",  0))
    failed  = int(totals.get("failed",  0))
    error   = int(totals.get("error",   0))
    skipped = int(totals.get("skipped", 0))

    issues: list[FunctionalIssue] = []
    for test in summary.get("tests", []):
        outcome = test.get("outcome", "")
        if outcome not in _ISSUE_OUTCOMES:
            continue
        nodeid = test.get("nodeid", "")
        # Prefer explicit longrepr; fall back to call.longrepr or empty.
        message = (
            test.get("longrepr")
            or test.get("call", {}).get("longrepr", "")
        )
        issues.append(FunctionalIssue(
            nodeid=nodeid,
            outcome=outcome,
            message=_shorten(str(message)),
        ))

    return FunctionalResult(
        total=total,
        passed=passed,
        failed=failed,
        error=error,
        skipped=skipped,
        issues=issues,
    )
