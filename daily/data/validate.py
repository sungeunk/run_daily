"""Invariants a stored run has to satisfy, and what to do when it doesn't.

Every counting bug this layer exists to prevent had the same shape: a number
went wrong, nothing downstream noticed, and a clamp or a default turned the
inconsistency into a plausible-looking report. ``Success 165 / Total 85``
sat on the dashboard for days because ``max(0, total - skipped - success)``
turned the overflow into ``Failed: 0``.

So the checks here are not about rejecting data. They record what does not
add up, and the report, the viewer and the MCP digest show it. Ingest never
raises on a violation: a nightly run that already cost an hour of machine
time must land even if its bookkeeping is odd, and a run that cannot be
stored cannot be investigated either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .counts import counts_are_consistent
from .series import is_infer_exec_mode, infer_twin

#: Ordered worst-first, which is also how a report should show them.
SEVERITIES = ("error", "warning")


@dataclass(frozen=True, slots=True)
class Finding:
    """One invariant that did not hold for one run."""

    code: str
    severity: str
    detail: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")


def _series_keys(perf_rows: Iterable) -> list[tuple]:
    return [(row.model, row.precision, row.in_token, row.out_token,
             row.exec_mode, getattr(row, "prompt_idx", 0))
            for row in perf_rows]


def check_run(rec) -> list[Finding]:
    """Every invariant that can be judged from one run in isolation.

    Cross-run rules (a build mapping to two SHAs, a machine that stopped
    reporting) need a cohort and belong to a query, not to ingest.
    """
    findings: list[Finding] = []
    perf_rows = list(getattr(rec, "perf", None) or [])
    expected = getattr(rec, "expected_cases", None)
    skipped = getattr(rec, "skipped_cases", None) or 0

    token_rows = [r for r in perf_rows if not is_infer_exec_mode(r.exec_mode)]
    success = len(token_rows)

    # The one that hid for days. Reported as an error because the clamp
    # downstream makes it invisible in the numbers themselves.
    if expected and not counts_are_consistent(expected, skipped, success):
        findings.append(Finding(
            "counts_overflow", "error",
            f"produced {success} token series plus {skipped} skipped, but the "
            f"run only expected {expected}; series_failed will clamp to zero "
            f"and hide anything genuinely missing",
        ))

    # A run that declares nothing cannot be judged complete or incomplete,
    # so its failures are indistinguishable from an empty suite.
    if perf_rows and not expected:
        findings.append(Finding(
            "expectation_missing", "warning",
            f"stored {len(perf_rows)} perf rows but declared no expected "
            f"cases, so completeness cannot be checked",
        ))

    keys = _series_keys(perf_rows)
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        example = ", ".join(
            f"{k[0]}/{k[4]}/in={k[2]}" for k in sorted(duplicates)[:3])
        findings.append(Finding(
            "duplicate_series", "error",
            f"{len(duplicates)} series key(s) stored more than once "
            f"({example}); the writer keeps the last and the rest are lost",
        ))

    # An infer row with no token twin means the family filter has nothing to
    # pair it with, and every count that excludes infer will disagree with
    # the raw row total for reasons nobody can see.
    token_keys = {(r.model, r.precision, r.in_token, r.out_token, r.exec_mode)
                  for r in token_rows}
    orphans = [
        r for r in perf_rows
        if is_infer_exec_mode(r.exec_mode)
        and not any(infer_twin(mode) == r.exec_mode
                    and (r.model, r.precision, r.in_token, r.out_token,
                         mode) in token_keys
                    for mode in ("1st", "2nd"))
    ]
    if orphans:
        findings.append(Finding(
            "orphan_infer_series", "warning",
            f"{len(orphans)} infer series have no token twin; they are "
            f"stored but excluded from every count, so raw row totals will "
            f"not reconcile",
        ))

    return findings


VALIDATIONS_DDL = """
    CREATE TABLE IF NOT EXISTS run_validations (
        run_id     TEXT NOT NULL,
        code       TEXT NOT NULL,
        severity   TEXT NOT NULL,
        detail     TEXT,
        checked_at TIMESTAMP DEFAULT now(),
        PRIMARY KEY (run_id, code)
    )
"""


def store(con, run_id: str, findings: list[Finding]) -> None:
    """Replace this run's findings. Called inside the ingest transaction."""
    con.execute(VALIDATIONS_DDL)
    con.execute("DELETE FROM run_validations WHERE run_id = ?", [run_id])
    if not findings:
        return
    con.executemany(
        "INSERT INTO run_validations (run_id, code, severity, detail) "
        "VALUES (?, ?, ?, ?)",
        [(run_id, f.code, f.severity, f.detail) for f in findings],
    )
