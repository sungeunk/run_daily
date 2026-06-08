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
from statistics import mean, median
from types import SimpleNamespace

from .baseline import _reference_purposes, find_last_known_good, select_baseline
from .functional import aggregate_functional
from .types import (
    AnalysisConfig,
    AnalysisResult,
    CurrentRunInfo,
    BaselineInfo,
    BisectDelta,
    ComparisonRow,
    ModelSummary,
    OverallStatus,
    PerformanceResult,
    SeriesKey,
)
from .verdict import improvement_pct, make_comparison_row, verdict_from_pct, verdict_from_signal

log = logging.getLogger(__name__)


def _build_current_run_info(rec) -> CurrentRunInfo:
    device = rec.devices[0] if getattr(rec, "devices", None) else None
    gpu_info = getattr(rec, "gpu_info", None) or (device.device if device else None) or getattr(rec, "device", None)
    gpu_driver_version = getattr(rec, "gpu_driver_version", None) or (device.driver if device else None)
    memory_size = (
        f"{rec.host_memory_size_gb:.1f} GB"
        if getattr(rec, "host_memory_size_gb", None) is not None
        else None
    )
    memory_speed = (
        f"{rec.host_memory_speed_mhz:.0f} MHz"
        if getattr(rec, "host_memory_speed_mhz", None) is not None
        else None
    )
    return CurrentRunInfo(
        ov_version=getattr(rec, "ov_version", None),
        purpose=getattr(rec, "purpose", None) or getattr(rec, "description", None),
        machine_name=getattr(rec, "machine", None),
        gpu_driver_version=gpu_driver_version,
        gpu_info=gpu_info,
        host_info=getattr(rec, "host_info", None),
        memory_size=memory_size,
        memory_speed=memory_speed,
    )


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
        rows = _fetch_comparison_rows(con, rec, baseline_info, config)

        # --- aggregate ---
        performance = _aggregate_performance(rows)
        models = _aggregate_models(rows)
        top_regressions = _top_regressions(rows, config.top_regressions)
        overall_status = _overall_status(functional, performance, baseline_info)
        last_known_good = None
        bisect_delta = None
        if overall_status in {"red", "yellow"}:
            last_known_good = find_last_known_good(con, rec, config)
            bisect_delta = _build_bisect_delta(
                con,
                current_run_id=rec.run_id,
                lkg=last_known_good,
                functional_issue_count=functional.issue_count,
                config=config,
            )

        result = AnalysisResult(
            overall_status=overall_status,
            baseline=baseline_info,
            functional=functional,
            performance=performance,
            models=models,
            top_regressions=top_regressions,
            current_run=_build_current_run_info(rec),
            last_known_good=last_known_good,
            bisect_delta=bisect_delta,
            rows=rows,
        )

        # Best-effort persistence for green-only baseline and downstream tabs.
        from .persistence import write_analysis_to_db, write_analysis_to_summary

        write_analysis_to_summary(summary_json, result, config=config)

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
    rec,
    baseline_info: BaselineInfo,
    config: AnalysisConfig,
) -> list[ComparisonRow]:
    if isinstance(rec, str):
        run_id = rec
        rec_ctx = _fetch_run_context(con, run_id)
    else:
        run_id = rec.run_id
        rec_ctx = rec
    if baseline_info.status != "found" or not baseline_info.run_id:
        return []

    history_map = _load_series_history(con, rec_ctx, config) if rec_ctx is not None else {}

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
        history_values = history_map.get((model, precision, int(in_token), int(out_token), exec_mode), [])
        history_stats = _history_stats(history_values, unit, config)

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
                    history_count=history_stats["count"],
                    reference_source="unit_mismatch",
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
                    history_count=history_stats["count"],
                )
            )
            continue

        reference_value = baseline_value
        reference_source = "baseline"
        enough_history_for_topk = history_stats["count"] >= max(
            config.min_recent_points,
            config.min_baseline_points,
        )
        if history_stats["topk_mean"] is not None and enough_history_for_topk:
            reference_value = history_stats["topk_mean"]
            reference_source = "topk_mean"

        row = make_comparison_row(key, unit, current_value, reference_value, config)
        row.reference_source = reference_source
        row.history_count = history_stats["count"]
        row.history_median = history_stats["median"]
        row.history_mad = history_stats["mad"]
        row.history_sigma = history_stats["sigma"]
        row.history_cv = history_stats["cv"]
        row.worsening_z = history_stats["worsening_z"](current_value)

        if reference_source == "topk_mean":
            row.improvement_pct = improvement_pct(current_value, reference_value, unit)
            row.verdict = verdict_from_signal(
                row.improvement_pct,
                config,
                worsening_z=row.worsening_z,
                recent_cv=row.history_cv,
                recent_n=row.history_count,
                baseline_n=row.history_count,
            )

            # When the delta is within historical fluctuation range,
            # downgrade to "same" even if pct threshold was crossed.
            if _is_within_fluctuation(
                current=current_value,
                reference=reference_value,
                sigma=row.history_sigma,
                scale=config.fluctuation_sigma_scale,
            ) and row.verdict in {"improved", "regressed"}:
                row.within_fluctuation = True
                row.verdict = "same"

        result.append(row)
    return result


def _load_series_history(con, rec, config: AnalysisConfig) -> dict[tuple, list[float]]:
    reference_purposes = _reference_purposes(config, getattr(rec, "purpose", None))
    if len(reference_purposes) == 1:
        purpose_clause = "COALESCE(r.purpose, '') = COALESCE(?, '')"
    else:
        purpose_clause = "COALESCE(r.purpose, '') IN ({})".format(
            ", ".join("?" for _ in reference_purposes)
        )
    try:
        rows = con.execute(
            """
        WITH ranked AS (
            SELECT
                p.model,
                p.precision,
                p.in_token,
                p.out_token,
                p.exec_mode,
                p.value,
                ROW_NUMBER() OVER (
                    PARTITION BY p.model, p.precision, p.in_token, p.out_token, p.exec_mode
                    ORDER BY r.ts DESC
                ) AS rn
            FROM perf p
            JOIN runs r USING (run_id)
            WHERE r.machine = ?
              AND r.run_id <> ?
              AND r.ts < ?
              AND r.short_run IS NOT DISTINCT FROM ?
              AND {purpose_clause}
        )
        SELECT model, precision, in_token, out_token, exec_mode, value
        FROM ranked
        WHERE rn <= ?
        ORDER BY model, precision, in_token, out_token, exec_mode, rn
            """.format(purpose_clause=purpose_clause),
            [
                rec.machine,
                rec.run_id,
                rec.ts,
                rec.short_run,
                *reference_purposes,
                config.history_window,
            ],
        ).fetchall()
    except Exception:
        return {}

    out: dict[tuple, list[float]] = {}
    for model, precision, in_token, out_token, exec_mode, value in rows:
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(num):
            continue
        key = (model, precision, int(in_token), int(out_token), exec_mode)
        out.setdefault(key, []).append(num)
    return out


def _history_stats(values: list[float], unit: str | None, config: AnalysisConfig) -> dict:
    if not values:
        return {
            "count": 0,
            "topk_mean": None,
            "median": None,
            "mad": None,
            "sigma": None,
            "cv": None,
            "worsening_z": lambda _current: None,
        }

    med = median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = median(abs_dev)
    sigma = 1.4826 * mad if mad is not None else None
    cv = None
    if sigma is not None and med not in (None, 0.0):
        cv = sigma / abs(med)

    top_k = max(1, min(config.reference_top_k, len(values)))
    lower_is_better = unit in {"ms", "s", "%"}
    selected = sorted(values)[:top_k] if lower_is_better else sorted(values, reverse=True)[:top_k]
    topk_mean = mean(selected)

    def _worsening_z(current: float) -> float | None:
        if sigma is None or sigma <= 0.0 or not math.isfinite(sigma):
            return None
        if lower_is_better:
            return (current - med) / sigma
        return (med - current) / sigma

    return {
        "count": len(values),
        "topk_mean": topk_mean,
        "median": med,
        "mad": mad,
        "sigma": sigma,
        "cv": cv,
        "worsening_z": _worsening_z,
    }


def _is_within_fluctuation(
    *,
    current: float,
    reference: float,
    sigma: float | None,
    scale: float,
) -> bool:
    if sigma is None or not math.isfinite(sigma) or sigma <= 0.0:
        return False
    return abs(current - reference) <= scale * sigma


def _aggregate_performance(rows: list[ComparisonRow]) -> PerformanceResult:
    counts = {
        "improved": 0,
        "same": 0,
        "regressed": 0,
        "unavailable": 0,
        "noisy": 0,
        "insufficient": 0,
    }
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    return PerformanceResult(
        compared=len(rows),
        improved=counts["improved"],
        same=counts["same"],
        regressed=counts["regressed"],
        unavailable=counts["unavailable"] + counts["noisy"] + counts["insufficient"],
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


def _build_bisect_delta(
    con,
    *,
    current_run_id: str,
    lkg: BaselineInfo,
    functional_issue_count: int,
    config: AnalysisConfig,
) -> BisectDelta:
    issue_meta = _fetch_run_meta(con, current_run_id)

    if lkg.status != "found" or not lkg.run_id:
        return BisectDelta(
            status="unavailable",
            issue_run_id=current_run_id,
            issue_stamp=issue_meta.get("stamp"),
            issue_ov_version=issue_meta.get("ov_version"),
            issue_ov_build=issue_meta.get("ov_build"),
            issue_ov_sha=issue_meta.get("ov_sha"),
            last_good_run_id=None,
            last_good_stamp=None,
            last_good_ov_version=None,
            last_good_ov_build=None,
            last_good_ov_sha=None,
            compared_count=0,
            regressed_count=0,
            functional_issue_count=functional_issue_count,
            build_changed=None,
            sha_changed=None,
        )

    rec = _fetch_run_context(con, current_run_id)
    if rec is None:
        return BisectDelta(
            status="unavailable",
            issue_run_id=current_run_id,
            issue_stamp=issue_meta.get("stamp"),
            issue_ov_version=issue_meta.get("ov_version"),
            issue_ov_build=issue_meta.get("ov_build"),
            issue_ov_sha=issue_meta.get("ov_sha"),
            last_good_run_id=lkg.run_id,
            last_good_stamp=lkg.stamp,
            last_good_ov_version=lkg.ov_version,
            last_good_ov_build=None,
            last_good_ov_sha=None,
            compared_count=0,
            regressed_count=0,
            functional_issue_count=functional_issue_count,
            build_changed=None,
            sha_changed=None,
        )

    lkg_rows = _fetch_comparison_rows(con, rec, lkg, config)
    comparable_rows = [
        row for row in lkg_rows
        if row.improvement_pct is not None and row.verdict != "unavailable"
    ]
    lkg_perf = _aggregate_performance(comparable_rows)
    lkg_meta = _fetch_run_meta(con, lkg.run_id)

    build_changed = _changed(issue_meta.get("ov_build"), lkg_meta.get("ov_build"))
    sha_changed = _changed(issue_meta.get("ov_sha"), lkg_meta.get("ov_sha"))

    status = "available" if lkg_perf.compared > 0 else "unavailable"

    return BisectDelta(
        status=status,
        issue_run_id=current_run_id,
        issue_stamp=issue_meta.get("stamp"),
        issue_ov_version=issue_meta.get("ov_version"),
        issue_ov_build=issue_meta.get("ov_build"),
        issue_ov_sha=issue_meta.get("ov_sha"),
        last_good_run_id=lkg.run_id,
        last_good_stamp=lkg_meta.get("stamp") or lkg.stamp,
        last_good_ov_version=lkg_meta.get("ov_version") or lkg.ov_version,
        last_good_ov_build=lkg_meta.get("ov_build"),
        last_good_ov_sha=lkg_meta.get("ov_sha"),
        compared_count=lkg_perf.compared,
        regressed_count=lkg_perf.regressed,
        functional_issue_count=functional_issue_count,
        build_changed=build_changed,
        sha_changed=sha_changed,
    )


def _fetch_run_meta(con, run_id: str) -> dict[str, str | None]:
    row = con.execute(
        """
        SELECT
            strftime(ts, '%Y%m%d_%H%M') AS stamp,
            ov_version,
            ov_build,
            ov_sha
        FROM runs
        WHERE run_id = ?
        LIMIT 1
        """,
        [run_id],
    ).fetchone()
    if not row:
        return {
            "stamp": None,
            "ov_version": None,
            "ov_build": None,
            "ov_sha": None,
        }
    stamp, ov_version, ov_build, ov_sha = row
    return {
        "stamp": stamp,
        "ov_version": ov_version,
        "ov_build": ov_build,
        "ov_sha": ov_sha,
    }


def _fetch_run_context(con, run_id: str):
    try:
        row = con.execute(
            """
            SELECT run_id, machine, ts, short_run, purpose
            FROM runs
            WHERE run_id = ?
            LIMIT 1
            """,
            [run_id],
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return SimpleNamespace(
        run_id=row[0],
        machine=row[1],
        ts=row[2],
        short_run=row[3],
        purpose=row[4],
    )


def _changed(current: str | None, previous: str | None) -> bool | None:
    if not current or not previous:
        return None
    return current != previous


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
