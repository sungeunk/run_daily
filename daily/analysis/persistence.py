"""Persistence layer: write AnalysisResult to summary.json and DuckDB.

Two responsibilities:
1. Append an ``analysis`` block to the existing ``summary.json`` so that
   downstream consumers (mail, dashboard, backfill) can read it without
   re-running the engine.
2. (Future / M3) Upsert into the ``analysis_results``,
   ``analysis_comparisons``, and ``functional_issues`` DB tables once
   those are added to ``schema.sql``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import AnalysisResult, BaselineInfo, ComparisonRow

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------

def write_analysis_to_summary(
    summary_json: Path,
    result: AnalysisResult,
    config: "AnalysisConfig | None" = None,
) -> None:
    """Append ``result`` as an ``analysis`` key in *summary_json*.

    The file is updated in-place.  Existing keys are not touched.
    If the ``analysis`` key already exists it is overwritten so that
    re-runs are idempotent. Config snapshot is included for reproducibility.
    """
    try:
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("persistence: cannot read %s: %s", summary_json, exc)
        return

    summary["analysis"] = _result_to_dict(result, config=config)

    try:
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("persistence: analysis block written to %s", summary_json)
    except Exception as exc:
        log.error("persistence: cannot write %s: %s", summary_json, exc)


# ---------------------------------------------------------------------------
# DuckDB (M3 — tables added in schema.sql)
# ---------------------------------------------------------------------------

def write_analysis_to_db(
    con,
    run_id: str,
    result: AnalysisResult,
    threshold_pct: float | None = None,
    config: "AnalysisConfig | None" = None,
) -> None:
    """Upsert analysis result into DB aggregate tables.

    Requires ``analysis_results``, ``analysis_comparisons``, and
    ``functional_issues`` to exist (added in M3).  Silently skips if
    those tables are absent so that M1/M2 deployments are unaffected.
    Config snapshot is stored for audit trail and reproducibility.
    """
    required = ("analysis_results", "analysis_comparisons", "functional_issues")
    if not all(_table_exists(con, name) for name in required):
        log.debug("analysis DB persistence skipped: required tables missing")
        return

    try:
        con.execute("BEGIN")
        _try_upsert_analysis_results(con, run_id, result, config=config)
        _try_upsert_analysis_comparisons(con, run_id, result, threshold_pct)
        _try_upsert_functional_issues(con, run_id, result)
        con.execute("COMMIT")
    except Exception as exc:  # noqa: BLE001
        con.execute("ROLLBACK")
        log.warning("analysis DB persistence rolled back for %s: %s", run_id, exc)


def _try_upsert_analysis_results(con, run_id: str, result: AnalysisResult, config: "AnalysisConfig | None" = None) -> None:
    p = result.performance
    b = result.baseline
    
    # Store config_json for reproducibility/audit
    config_json = None
    if config is not None:
        from dataclasses import asdict as dc_asdict
        config_json = json.dumps(dc_asdict(config))
    
    con.execute(
        """
        INSERT INTO analysis_results (
            run_id, baseline_run_id, overall_status,
            compared_count, improved_count, same_count,
            regressed_count, functional_fail_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            baseline_run_id      = excluded.baseline_run_id,
            overall_status       = excluded.overall_status,
            compared_count       = excluded.compared_count,
            improved_count       = excluded.improved_count,
            same_count           = excluded.same_count,
            regressed_count      = excluded.regressed_count,
            functional_fail_count = excluded.functional_fail_count,
            updated_at           = now()
        """,
        [
            run_id,
            b.run_id if b.status == "found" else None,
            result.overall_status,
            p.compared, p.improved, p.same, p.regressed,
            result.functional.issue_count,
        ],
    )


def _try_upsert_analysis_comparisons(
    con, run_id: str, result: AnalysisResult, threshold_pct: float | None
) -> None:
    con.execute(
        "DELETE FROM analysis_comparisons WHERE run_id = ?", [run_id]
    )
    if not result.rows:
        return
    con.executemany(
        """
        INSERT INTO analysis_comparisons (
            run_id, baseline_run_id,
            model, precision, in_token, out_token, exec_mode,
            unit, current_value, baseline_value,
            improvement_pct, verdict, threshold_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            _row_tuple(run_id, result.baseline.run_id, row, threshold_pct)
            for row in result.rows
        ],
    )


def _try_upsert_functional_issues(
    con, run_id: str, result: AnalysisResult
) -> None:
    con.execute(
        "DELETE FROM functional_issues WHERE run_id = ?", [run_id]
    )
    if not result.functional.issues:
        return
    con.executemany(
        """
        INSERT INTO functional_issues (run_id, nodeid, outcome, message)
        VALUES (?, ?, ?, ?)
        """,
        [
            (run_id, i.nodeid, i.outcome, i.message)
            for i in result.functional.issues
        ],
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _row_tuple(
    run_id: str,
    baseline_run_id: str | None,
    row: ComparisonRow,
    threshold_pct: float | None,
) -> tuple:
    k = row.key
    return (
        run_id, baseline_run_id,
        k.model, k.precision, k.in_token, k.out_token, k.exec_mode,
        row.unit, row.current_value, row.baseline_value,
        row.improvement_pct, row.verdict, threshold_pct,
    )


def _table_exists(con, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ? LIMIT 1",
        [table_name],
    ).fetchone()
    return bool(row)


def _result_to_dict(result: AnalysisResult, config: "AnalysisConfig | None" = None) -> dict[str, Any]:
    """Convert AnalysisResult to a JSON-serialisable dict.

    The ``rows`` field (full comparison table) is excluded from the JSON
    to keep file size reasonable; ``top_regressions`` is sufficient.
    Config snapshot is included for reproducibility and audit trails.
    """
    def _row_dict(row: ComparisonRow) -> dict:
        k = row.key
        return {
            "model": k.model,
            "precision": k.precision,
            "in_token": k.in_token,
            "out_token": k.out_token,
            "exec_mode": k.exec_mode,
            "unit": row.unit,
            "current_value": row.current_value,
            "baseline_value": row.baseline_value,
            "improvement_pct": row.improvement_pct,
            "verdict": row.verdict,
        }

    b = result.baseline
    payload = {
        "overall_status": result.overall_status,
        "baseline": {
            "status": b.status,
            "run_id": b.run_id,
            "stamp": b.stamp,
            "ov_version": b.ov_version,
            "selection_reason": b.selection_reason,
        },
        "functional": {
            "total":   result.functional.total,
            "passed":  result.functional.passed,
            "failed":  result.functional.failed,
            "error":   result.functional.error,
            "skipped": result.functional.skipped,
            "issue_count": result.functional.issue_count,
            "issues": [
                {"nodeid": i.nodeid, "outcome": i.outcome, "message": i.message}
                for i in result.functional.issues
            ],
        },
        "performance": {
            "compared":    result.performance.compared,
            "improved":    result.performance.improved,
            "same":        result.performance.same,
            "regressed":   result.performance.regressed,
            "unavailable": result.performance.unavailable,
        },
        "models": [
            {
                "model": m.model,
                "avg_improvement_pct": m.avg_improvement_pct,
                "improved":  m.improved,
                "same":      m.same,
                "regressed": m.regressed,
            }
            for m in result.models
        ],
        "top_regressions": [_row_dict(r) for r in result.top_regressions],
    }
    
    # Include config snapshot for reproducibility
    if config is not None:
        from dataclasses import asdict as dc_asdict
        payload["config_snapshot"] = dc_asdict(config)
    
    if result.last_known_good is not None:
        lkg = result.last_known_good
        payload["last_known_good"] = {
            "status": lkg.status,
            "run_id": lkg.run_id,
            "stamp": lkg.stamp,
            "ov_version": lkg.ov_version,
            "selection_reason": lkg.selection_reason,
        }
    if result.bisect_delta is not None:
        delta = result.bisect_delta
        payload["bisect_delta"] = {
            "status": delta.status,
            "issue_run_id": delta.issue_run_id,
            "issue_stamp": delta.issue_stamp,
            "issue_ov_version": delta.issue_ov_version,
            "issue_ov_build": delta.issue_ov_build,
            "issue_ov_sha": delta.issue_ov_sha,
            "last_good_run_id": delta.last_good_run_id,
            "last_good_stamp": delta.last_good_stamp,
            "last_good_ov_version": delta.last_good_ov_version,
            "last_good_ov_build": delta.last_good_ov_build,
            "last_good_ov_sha": delta.last_good_ov_sha,
            "compared_count": delta.compared_count,
            "regressed_count": delta.regressed_count,
            "functional_issue_count": delta.functional_issue_count,
            "build_changed": delta.build_changed,
            "sha_changed": delta.sha_changed,
        }
    return payload
