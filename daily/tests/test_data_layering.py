"""Structural guards: the shared semantics stay shared.

daily/data exists because the same rules had been written out in several
places and drifted. Extracting them fixes today; these tests are what stops
tomorrow's call site from spelling them out again, which is exactly how the
infer-family bug reached four functions at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.dev_only

DAILY_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DAILY_DIR / "data"


def _python_files(*, exclude_dirs=("data", "tests", "__pycache__")):
    for path in sorted(DAILY_DIR.rglob("*.py")):
        rel = path.relative_to(DAILY_DIR)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        yield path, path.read_text(encoding="utf-8")


def _sql_and_python_files():
    yield from _python_files()
    for path in sorted(DAILY_DIR.rglob("*.sql")):
        if "data" not in path.relative_to(DAILY_DIR).parts:
            yield path, path.read_text(encoding="utf-8")


def test_direction_rule_is_not_respelled_outside_data():
    """`unit in {'ms','s','%'}` had six copies. One is enough."""
    pattern = re.compile(r"""['"]ms['"]\s*,\s*['"]s['"]\s*,\s*['"]%['"]""")
    offenders = [str(path.relative_to(DAILY_DIR))
                 for path, text in _sql_and_python_files() if pattern.search(text)]
    assert not offenders, (
        "the lower-is-better unit set belongs to data.series; use "
        f"lower_is_better()/direction_sign()/direction_label_sql() in: {offenders}"
    )


#: Comparing against an infer name is the thing to catch; naming one while
#: explaining it is not. Prose mentions have no operator next to the literal.
_MATCHING_OPERATOR = re.compile(
    r"(==|!=|\bIN\s*\(|\bLIKE\b|endswith\(|startswith\()", re.IGNORECASE)


def test_infer_family_is_not_matched_by_hand():
    """Every exclusion must go through exclude_infer_sql/is_infer_exec_mode,
    so a new infer-style family is excluded everywhere at once."""
    literal = re.compile(r"""['"](?:1st-infer|2nd-infer|-infer)['"]""")
    offenders = []
    for path, text in _sql_and_python_files():
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("#", "--")):
                continue
            if not literal.search(line) or not _MATCHING_OPERATOR.search(line):
                continue
            if "exclude_infer_sql" in line or "INFER_SUFFIX" in line:
                continue
            offenders.append(f"{path.relative_to(DAILY_DIR)}: {stripped[:70]}")
    assert not offenders, (
        "use data.series helpers instead of literal infer names: " + str(offenders))


def test_perf_rows_are_not_counted_by_hand():
    """Counting rows in `perf` is the contract data.counts owns; four call
    sites doing it themselves is what let success run past total."""
    pattern = re.compile(r"count\(\*\)\s*(?:AS\s+\w+\s*)?FROM\s+perf\b",
                         re.IGNORECASE)
    offenders = [str(path.relative_to(DAILY_DIR))
                 for path, text in _sql_and_python_files() if pattern.search(text)]
    assert not offenders, (
        f"use data.counts.count_success_series() in: {offenders}")


def test_seconds_to_milliseconds_lives_in_one_place():
    """Outside data, nobody converts units by hand.

    data/schema.sql still spells the rule out in the perf_with_buckets view
    because a view cannot import Python; collapsing that onto data.series is
    the one remaining copy, and it is inside the layer that owns the rule.
    """
    pattern = re.compile(r"value\s*\*\s*1000|\*\s*1000\.0\s*(?:#|$)")
    offenders = {str(path.relative_to(DAILY_DIR))
                 for path, text in _sql_and_python_files() if pattern.search(text)}
    assert not offenders, (
        "use data.series.normalize_value()/normalize_value_sql() in: "
        f"{sorted(offenders)}")


def test_only_the_data_layer_and_the_monitor_open_duckdb():
    """A direct connection bypasses every filter and normalisation rule.

    monitor_parquet is exempt: it reads Parquet sample files, not the
    benchmark tables. The MCP server and the ingest writer are exempt because
    they own a connection's lifetime rather than a query's meaning — they are
    the next thing to move behind data.read/data.write.
    """
    # The MCP server owns a connection's lifetime rather than a query's
    # meaning, so it stays. Nothing else outside the layer connects.
    allowed = {"common/monitor_parquet.py", "mcp_server/server.py"}
    offenders = [
        str(path.relative_to(DAILY_DIR))
        for path, text in _python_files()
        if "duckdb.connect" in text
        and str(path.relative_to(DAILY_DIR)) not in allowed
    ]
    assert not offenders, (
        f"go through daily.data instead of connecting directly: {offenders}")


def test_data_layer_does_not_import_its_consumers():
    """daily/data must stay a leaf, or the layering is decorative."""
    forbidden = ("from viewer", "import viewer", "from analysis",
                 "import analysis", "from report", "from mcp_server")
    offenders = []
    for path in sorted(DATA_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if any(stripped.startswith(bad) for bad in forbidden):
                offenders.append(f"{path.name}: {stripped}")
    assert not offenders, f"daily/data must not depend on its callers: {offenders}"
