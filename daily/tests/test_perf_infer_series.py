"""The infer-only latency series: ingested, queryable, never voted.

``1st``/``2nd`` are the daily metric; ``1st-infer``/``2nd-infer`` exist so a
GPU-plugin kernel regression can be told apart from pipeline overhead. These
tests pin both halves of that contract — the rows reach the DB, and they stay
out of the verdict.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

pytestmark = pytest.mark.dev_only

DAILY_DIR = Path(__file__).resolve().parent.parent
if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

from analysis.engine import _fetch_comparison_rows  # noqa: E402
from analysis.types import AnalysisConfig  # noqa: E402
from common.perf_series import is_infer_exec_mode  # noqa: E402
from viewer import queries as q  # noqa: E402
from viewer.ingest import writer  # noqa: E402
from viewer.ingest.loader_new import _llm_rows  # noqa: E402
from viewer.ingest.record import RunRecord  # noqa: E402


def _metrics(**data) -> dict:
    return {
        "model": "llama-3.1-8b",
        "precision": "INT4",
        "data": [{"prompt_idx": 0, "in_token": 32, "out_token": 256, **data}],
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_infer_series_is_extracted_alongside_token_latency():
    rows = list(_llm_rows(_metrics(perf=[87.73, 58.65],
                                   infer_perf=[87.685, 58.649])))

    assert [(r.exec_mode, r.value) for r in rows] == [
        ("1st", 87.73),
        ("2nd", 58.65),
        ("1st-infer", 87.685),
        ("2nd-infer", 58.649),
    ]
    assert {r.unit for r in rows} == {"ms"}
    assert {r.prompt_idx for r in rows} == {0}


def test_runs_without_infer_perf_still_ingest():
    """Runs predating the parser that emits infer_perf must not regress."""
    rows = list(_llm_rows(_metrics(perf=[87.73, 58.65])))

    assert [r.exec_mode for r in rows] == ["1st", "2nd"]


def test_partial_infer_perf_yields_only_the_values_present():
    """llm_bench reports -1 for an unmeasured slice; the parser drops it, so
    infer_perf can be shorter than perf. Zip must not invent a 2nd-infer."""
    rows = list(_llm_rows(_metrics(perf=[87.73, 58.65], infer_perf=[87.685])))

    assert [r.exec_mode for r in rows] == ["1st", "2nd", "1st-infer"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _ingest(db_path: Path, metrics: dict) -> None:
    rec = RunRecord(
        run_id="run-001",
        source_format="new",
        report_file="daily.1.summary.json",
        machine="TEST-01",
        ts=datetime(2026, 1, 1, 12, 0),
        run_kind="daily",
    )
    rec.perf.extend(_llm_rows(metrics))
    con = writer.connect(db_path)
    writer.ensure_schema(con)
    writer.upsert_run(con, rec)
    con.close()


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "test.duckdb"
    con = writer.connect(path)
    writer.ensure_schema(con)
    con.close()
    return path


def test_infer_rows_survive_the_writer(db: Path):
    """Both families share a prompt, so they only coexist because exec_mode
    is part of the perf primary key."""
    _ingest(db, _metrics(perf=[87.73, 58.65], infer_perf=[87.685, 58.649]))

    con = duckdb.connect(str(db), read_only=True)
    stored = dict(con.execute(
        "SELECT exec_mode, value FROM perf ORDER BY exec_mode"
    ).fetchall())
    con.close()

    assert stored == {
        "1st": 87.73,
        "2nd": 58.65,
        "1st-infer": 87.685,
        "2nd-infer": 58.649,
    }


def test_infer_rows_are_not_reported_as_unexpected_series(db: Path):
    """extra_rows flags perf series with no display row. The infer series has
    none by design, so listing it would drown the real signal."""
    _ingest(db, _metrics(perf=[87.73, 58.65], infer_perf=[87.685, 58.649]))

    extra = q.extra_rows(db, ["run-001"])

    assert not extra.empty, "the token series has no display profile here"
    assert not any(is_infer_exec_mode(m) for m in extra["exec_mode"])


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@pytest.fixture
def perf_db():
    with duckdb.connect(":memory:") as con:
        con.execute(
            "CREATE TABLE perf (run_id VARCHAR, model VARCHAR, "
            "precision VARCHAR, in_token INTEGER, out_token INTEGER, "
            "exec_mode VARCHAR, unit VARCHAR, value DOUBLE)"
        )
        con.execute(
            "INSERT INTO perf VALUES "
            "('cur', 'llama', 'INT4', 32, 256, '1st',       'ms', 120.0), "
            "('cur', 'llama', 'INT4', 32, 256, '1st-infer', 'ms', 119.0)"
        )
        yield con


def _keys(rows) -> set[tuple[str, str]]:
    return {(r.key.model, r.key.exec_mode) for r in rows}


def test_infer_series_never_reaches_the_verdict(perf_db):
    rows = _fetch_comparison_rows(
        perf_db, "cur", AnalysisConfig(),
        reference_values={("llama", "INT4", 32, 256, "1st"): (100.0, "ms"),
                          ("llama", "INT4", 32, 256, "1st-infer"): (99.0, "ms")},
        history_map={},
    )

    assert _keys(rows) == {("llama", "1st")}


def test_infer_series_absent_locally_is_still_not_voted(perf_db):
    """A reference DB may carry the infer series before the local machine
    starts producing it; that must not create a phantom 'missing' row."""
    perf_db.execute("DELETE FROM perf WHERE exec_mode = '1st-infer'")

    rows = _fetch_comparison_rows(
        perf_db, "cur", AnalysisConfig(),
        reference_values={("llama", "INT4", 32, 256, "1st"): (100.0, "ms"),
                          ("llama", "INT4", 32, 256, "2nd-infer"): (20.0, "ms")},
        history_map={},
    )

    assert _keys(rows) == {("llama", "1st")}
