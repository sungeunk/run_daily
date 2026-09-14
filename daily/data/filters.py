"""Which runs a query is allowed to see.

"Daily runs only, nothing manually excluded, nothing partial" is one idea,
but it had five spellings: ``_run_kind_predicate``, ``_run_kind_clause``,
``_perf_flat_kind_clause``, ``_exclusion_predicate`` and
``_perf_flat_exclusion_clause`` -- plus a sixth hand-written copy inside
``analysis/remote.py``. Callers picked whichever they happened to find, and
the ones that picked none silently reported over excluded and partial runs.

The five existed because the same idea is physically different depending on
what you select from:

``runs``
    Raw table. Exclusion needs a ``NOT EXISTS`` against ``run_exclusions``
    and there is no ``is_partial`` at all.
``runs_with_flags`` / ``perf_flat``
    Views that already derive ``excluded`` and ``is_partial`` as columns, so
    the filter is a plain boolean.

That is a real difference, so it is modelled explicitly here as the relation
rather than left for each caller to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import duckdb

#: Canonical run kinds. Anything else the sidebar offers is free text.
RUN_KINDS = ("daily", "pr", "test", "manual")
DEFAULT_RUN_KINDS = ("daily",)

#: Relations that expose ``excluded`` / ``is_partial`` as derived columns.
FLAG_RELATIONS = frozenset({"runs_with_flags", "perf_flat", "perf_stats"})


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------
#
# The viewer opens the DB read-only and cannot migrate it, so a deployment
# where ingest has not yet run with the current schema must degrade to the
# columns it does have instead of failing to render.

@lru_cache(maxsize=32)
def _cached_tables(db_path_str: str, mtime_ns: int) -> frozenset[str]:
    with duckdb.connect(db_path_str, read_only=True) as con:
        return frozenset(
            row[0] for row in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        )


@lru_cache(maxsize=64)
def _cached_columns(db_path_str: str, relation: str,
                    mtime_ns: int) -> frozenset[str]:
    with duckdb.connect(db_path_str, read_only=True) as con:
        try:
            return frozenset(row[0] for row in
                             con.execute(f"DESCRIBE {relation}").fetchall())
        except duckdb.Error:
            return frozenset()


def tables(db_path: Path) -> frozenset[str]:
    return _cached_tables(str(db_path), db_path.stat().st_mtime_ns)


def has_table(db_path: Path, name: str) -> bool:
    return name in tables(db_path)


def has_column(db_path: Path, relation: str, column: str) -> bool:
    return column in _cached_columns(str(db_path), relation,
                                     db_path.stat().st_mtime_ns)


# ---------------------------------------------------------------------------
# Run kind
# ---------------------------------------------------------------------------

def run_kind_predicate(run_kinds: Sequence[str] | None,
                       alias: str | None = "r") -> tuple[str, list]:
    """Boolean predicate (no leading ``AND``) for the Run kinds selector.

    The selector accepts free-typed text alongside the canonical kinds.
    Anything that isn't one of ``RUN_KINDS`` is treated as a keyword and
    matched case-insensitively against ``purpose``/``description`` -- the same
    free text the canonical kinds were classified from -- so a user can narrow
    to a PR number or username the fixed categories don't capture.

    ``alias=None`` emits unqualified names, for a single-table select straight
    from ``perf_flat`` where there is nothing to qualify against.
    """
    if not run_kinds:
        return "", []
    kinds = [k for k in run_kinds if k in RUN_KINDS]
    keywords = [k.strip() for k in run_kinds if k not in RUN_KINDS and k.strip()]
    col = f"{alias}." if alias else ""

    parts: list[str] = []
    params: list = []
    if kinds:
        placeholders = ",".join(["?"] * len(kinds))
        parts.append(f"COALESCE({col}run_kind, 'daily') IN ({placeholders})")
        params.extend(kinds)
    for keyword in keywords:
        parts.append(
            f"(LOWER(COALESCE({col}purpose, '')) LIKE ? "
            f"OR LOWER(COALESCE({col}description, '')) LIKE ?)"
        )
        like = f"%{keyword.lower()}%"
        params.extend([like, like])

    if not parts:
        return "", []
    predicate = parts[0] if len(parts) == 1 else f"({' OR '.join(parts)})"
    return predicate, params


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunScope:
    """The set of runs a cohort query may report over.

    Build one, call :meth:`where`, and splice the result into the query. The
    defaults are the cohort semantics -- daily runs, nothing excluded, nothing
    partial -- so a caller that wants something looser has to say so, which is
    the opposite of how the old helpers behaved.
    """

    db_path: Path
    relation: str = "runs_with_flags"
    alias: str | None = "r"
    run_kinds: tuple[str, ...] | None = DEFAULT_RUN_KINDS
    include_excluded: bool = False
    include_partial: bool = False

    def _column(self, name: str) -> str:
        return f"{self.alias}.{name}" if self.alias else name

    def _has_flags(self) -> bool:
        return self.relation in FLAG_RELATIONS

    def predicates(self) -> tuple[list[str], list]:
        """The individual boolean predicates and their bound parameters."""
        parts: list[str] = []
        params: list = []

        kind_sql, kind_params = ("", [])
        if self.run_kinds and (
                not self._has_flags()
                or has_column(self.db_path, self.relation, "run_kind")):
            kind_sql, kind_params = run_kind_predicate(self.run_kinds, self.alias)
        if kind_sql:
            parts.append(kind_sql)
            params.extend(kind_params)

        if not self.include_excluded:
            if self._has_flags():
                if has_column(self.db_path, self.relation, "excluded"):
                    parts.append(f"NOT {self._column('excluded')}")
            elif has_table(self.db_path, "run_exclusions"):
                # The raw table has no derived flag, so the exclusion list has
                # to be consulted directly.
                parts.append(
                    "NOT EXISTS (SELECT 1 FROM run_exclusions e "
                    f"WHERE e.run_id = {self._column('run_id')})"
                )

        if not self.include_partial and self._has_flags():
            if has_column(self.db_path, self.relation, "is_partial"):
                parts.append(f"NOT {self._column('is_partial')}")

        return parts, params

    def where(self, *, leading_and: bool = True) -> tuple[str, list]:
        """SQL fragment and parameters.

        With ``leading_and`` (the default) the fragment is ready to append to
        an existing ``WHERE``; without it, it is a bare conjunction.
        """
        parts, params = self.predicates()
        if not parts:
            return "", []
        joined = " AND ".join(parts)
        return (f" AND {joined}" if leading_and else joined), params


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
#
# Lives here rather than in the ingest loader because the identity a run
# declares and the identity a query filters on have to be the same idea.
# queries.py already reached across into viewer.ingest.loader_new for
# parse_triggered_by; with the loader as the owner, data would have had to
# import its own consumer.

# Free-form purpose text is the only hint about who launched a run, so map it
# to a fixed vocabulary the viewer can filter on. Order matters: an explicit
# PR marker wins over everything, and a run that calls itself daily stays
# daily even if its description also mentions validation.
# Word boundaries keep short tokens like 'ci' and 'pr' from matching inside
# words such as 'precision'.
_RUN_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pr", r"\bpr[-_# ]*\d+|\bpull[-_ ]?request\b|\bpre-?commit\b"),
    ("daily", r"\bdaily|\bnightly\b|\bweekly\b"),
    ("test", r"\btest|\btrial\b|\bdebug\b|\bexperiment|\bjenkins\b|\bci\b|\bvalidation\b"),
)


def classify_run_kind(purpose: str | None, description: str | None = None) -> str:
    """Map purpose/description text onto 'daily' | 'pr' | 'test' | 'manual'."""
    text = f"{purpose or ''} {description or ''}".strip().lower()
    if not text:
        return "manual"
    for kind, pattern in _RUN_KIND_PATTERNS:
        if re.search(pattern, text):
            return kind
    return "manual"


_GENERIC_TRIGGERED_BY = {
    "daily", "nightly", "weekly", "timer", "pipeline",
    "llm", "cb", "test", "validation", "manual", "scheduler",
    "unknown", "jenkins", "build", "report"
}

_LEGACY_TIMER_PURPOSES = {
    "daily2 timer",
    "daily_cb timer",
    "daily_pipeline timer",
}


def parse_triggered_by(purpose: str | None, description: str | None = None) -> str | None:
    """Recover the execution identity from free-form purpose text.

    Older summary files often omit the dedicated ``triggered_by`` metadata, but
    the run description still ends with the launcher's identity, e.g.
    ``daily pipeline sungeunk`` or ``daily_CB jenkins-user``.
    """
    normalized_purpose = (purpose or "").strip().lower()
    if normalized_purpose in _LEGACY_TIMER_PURPOSES:
        return "timer"

    text = " ".join(part for part in (purpose, description) if part).strip()
    if not text:
        return None
    tail = text.rsplit(maxsplit=1)[-1].strip(" \t\r\n.,;:()[]{}")
    if not tail:
        return None
    # Ignore descriptive keywords and keep a meaningful creator name when the
    # metadata is missing.
    candidate = tail.strip("_-")
    if not candidate or candidate.lower() in _GENERIC_TRIGGERED_BY:
        return None
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate):
        return candidate
    return None
