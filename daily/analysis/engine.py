"""Analysis engine: top-level orchestration.

Usage::

    from analysis.engine import analyze_run
    from analysis.types import AnalysisConfig

    result = analyze_run(summary_json, db_path)          # default config
    result = analyze_run(summary_json, db_path, config)  # custom thresholds
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from .baseline import select_baseline
from .functional import aggregate_functional
from .types import (
    AnalysisConfig,
    AnalysisResult,
    BaselineInfo,
    ComparisonRow,
    ModelSummary,
    OverallStatus,
    PerformanceResult,
    SeriesKey,
)
from .verdict import make_comparison_row, verdict_from_pct

log = logging.getLogger(__name__)


def analyze_run(
    summary_json: Path,
    db_path: Path,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Run the full analysis pipeline for one completed run.

    Steps:

    1. Parse ``summary.json`` for metadata and functional totals.
    2. Ingest / upsert the run into DuckDB (idempotent).
    3. Select the best baseline run.
    4. Fetch matched perf rows and compute per-series verdicts.
    5. Aggregate model-level and overall summaries.

    Args:
        summary_json: Path to the ``*.summary.json`` produced by run.py.
        db_path:      Path to ``bench.duckdb``.
        config:       Optional tuning; defaults to :class:`AnalysisConfig`.

    Returns:
        A fully populated :class:`AnalysisResult`.
    """
    if config is None:
        config = AnalysisConfig()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    # --- functional aggregation (no DB needed) ---
    functional = aggregate_functional(summary)

    # --- ingest + baseline selection ---
    from viewer.ingest.loader_new import load_summary
    from viewer.ingest.writer import connect, ensure_schema, upsert_run

    rec = load_summary(summary_json)

    with connect(db_path) as con:
        ensure_schema(con)
        upsert_run(con, rec)
        baseline_info = select_baseline(con, rec, config)
        rows = _fetch_comparison_rows(con, rec.run_id, baseline_info, config)

        # --- aggregate ---
        performance = _aggregate_performance(rows)
        models = _aggregate_models(rows)
        top_regressions = _top_regressions(rows, config.top_regressions)
        overall_status = _overall_status(functional, performance, baseline_info)

        result = AnalysisResult(
            overall_status=overall_status,
            baseline=baseline_info,
            functional=functional,
            performance=performance,
            models=models,
            top_regressions=top_regressions,
            rows=rows,
        )

        # Best-effort persistence for green-only baseline and downstream tabs.
        from .persistence import write_analysis_to_db

        write_analysis_to_db(
            con,
            rec.run_id,
            result,
            threshold_pct=config.pct_threshold,
        )
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_comparison_rows(
    con,
    run_id: str,
    baseline_info: BaselineInfo,
    config: AnalysisConfig,
) -> list[ComparisonRow]:
    if baseline_info.status != "found" or not baseline_info.run_id:
        return []

    db_rows = con.execute(
        """
        SELECT
            COALESCE(c.model, b.model) AS model,
            COALESCE(c.precision, b.precision) AS precision,
            COALESCE(c.in_token, b.in_token) AS in_token,
            COALESCE(c.out_token, b.out_token) AS out_token,
            COALESCE(c.exec_mode, b.exec_mode) AS exec_mode,
            c.unit AS current_unit,
            b.unit AS baseline_unit,
            c.value AS current_value,
            b.value AS baseline_value
        FROM perf c
        FULL OUTER JOIN perf b
          ON c.model     = b.model
         AND c.precision = b.precision
         AND c.in_token  = b.in_token
         AND c.out_token = b.out_token
         AND c.exec_mode = b.exec_mode
         AND c.run_id = ?
         AND b.run_id = ?
        WHERE c.run_id = ? OR b.run_id = ?
        ORDER BY model, precision, in_token, out_token, exec_mode
        """,
        [run_id, baseline_info.run_id, run_id, baseline_info.run_id],
    ).fetchall()

    result: list[ComparisonRow] = []

    def _as_finite_float(value) -> float | None:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        return num if math.isfinite(num) else None

    for model, precision, in_token, out_token, exec_mode, current_unit, baseline_unit, cur, base in db_rows:
        key = SeriesKey(
            model=model,
            precision=precision,
            in_token=int(in_token),
            out_token=int(out_token),
            exec_mode=exec_mode,
        )
        unit = current_unit or baseline_unit
        current_value = _as_finite_float(cur)
        baseline_value = _as_finite_float(base)

        # Unit mismatch: treat as unavailable to avoid comparing apples to oranges.
        if current_unit is not None and baseline_unit is not None and current_unit != baseline_unit:
            result.append(
                ComparisonRow(
                    key=key,
                    unit=unit,
                    current_value=float("nan"),
                    baseline_value=float("nan"),
                    improvement_pct=None,
                    verdict=verdict_from_pct(None, config),
                )
            )
            continue

        if current_value is None or baseline_value is None:
            result.append(
                ComparisonRow(
                    key=key,
                    unit=unit,
                    current_value=float("nan") if current_value is None else current_value,
                    baseline_value=float("nan") if baseline_value is None else baseline_value,
                    improvement_pct=None,
                    verdict=verdict_from_pct(None, config),
                )
            )
            continue

        result.append(make_comparison_row(key, unit, current_value, baseline_value, config))
    return result


def _aggregate_performance(rows: list[ComparisonRow]) -> PerformanceResult:
    counts = {"improved": 0, "same": 0, "regressed": 0, "unavailable": 0, "noisy": 0}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    return PerformanceResult(
        compared=len(rows),
        improved=counts["improved"],
        same=counts["same"],
        regressed=counts["regressed"],
        unavailable=counts["unavailable"] + counts["noisy"],
    )


def _aggregate_models(rows: list[ComparisonRow]) -> list[ModelSummary]:
    stats: dict[str, dict] = {}
    for row in rows:
        m = row.key.model
        if m not in stats:
            stats[m] = {"sum": 0.0, "n": 0, "improved": 0, "same": 0, "regressed": 0}
        s = stats[m]
        if row.improvement_pct is not None:
            s["sum"] += row.improvement_pct
            s["n"] += 1
        if row.verdict in ("improved", "same", "regressed"):
            s[row.verdict] += 1

    def _avg(s: dict) -> float | None:
        return s["sum"] / s["n"] if s["n"] else None

    # Sort by absolute average change descending so biggest movers appear first.
    return [
        ModelSummary(
            model=m,
            avg_improvement_pct=_avg(s),
            improved=s["improved"],
            same=s["same"],
            regressed=s["regressed"],
        )
        for m, s in sorted(
            stats.items(),
            key=lambda kv: abs(kv[1]["sum"] / kv[1]["n"]) if kv[1]["n"] else 0,
            reverse=True,
        )
    ]


def _top_regressions(
    rows: list[ComparisonRow],
    n: int,
) -> list[ComparisonRow]:
    regressed = [r for r in rows if r.verdict == "regressed" and r.improvement_pct is not None]
    return sorted(regressed, key=lambda r: r.improvement_pct)[:n]  # type: ignore[arg-type]


def _overall_status(
    functional: "FunctionalResult",
    performance: PerformanceResult,
    baseline: BaselineInfo,
) -> OverallStatus:
    # Functional failures take priority over performance signals.
    if functional.issue_count > 0:
        return "red"
    if baseline.status != "found":
        return "gray"
    valid_compared = performance.improved + performance.same + performance.regressed
    if valid_compared == 0:
        return "gray"
    if performance.regressed > 0:
        return "yellow"
    return "green"
