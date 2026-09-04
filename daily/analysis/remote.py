"""Reference and release runs, read from the central ``daily_results`` server.

Scheduled ("timer") runs land in the shared DuckDB behind the daily_results
MCP server, not in the local per-machine database, so both the reference run
and the history behind sigma/CV have to come from there. A failed lookup is
reported as ``unavailable`` and never silently downgraded to a local guess:
comparing today's numbers against the wrong reference is worse than showing
no comparison at all.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from viewer.ingest.record import RunRecord

from .types import AnalysisConfig, BaselineInfo, ReleaseInfo

log = logging.getLogger(__name__)

SeriesKeyTuple = tuple[str, str, int, int, str]
SeriesValues = dict[SeriesKeyTuple, tuple[float, str | None]]
SeriesHistory = dict[SeriesKeyTuple, list[float]]


class ReferenceResult:
    """Reference run metadata plus the series data derived from it."""

    def __init__(
        self,
        info: BaselineInfo,
        values: SeriesValues | None = None,
        history: SeriesHistory | None = None,
    ) -> None:
        self.info = info
        self.values: SeriesValues = values or {}
        self.history: SeriesHistory = history or {}


def fetch_reference(config: AnalysisConfig, rec: "RunRecord") -> ReferenceResult:
    """Return the newest scheduled run older than *rec*, plus its history.

    One round trip pulls the last ``config.history_window`` scheduled runs;
    the newest of them is the reference and all of them feed the sigma/CV
    statistics.
    """
    machine = getattr(rec, "machine", None)
    url = config.mcp_url
    if not machine:
        return ReferenceResult(
            BaselineInfo(status="not_found", source_url=url,
                         detail="current run has no machine name")
        )

    from common.mcp_client import McpError, run_sql

    try:
        runs = run_sql(url, _reference_runs_sql(config, rec), timeout=config.mcp_timeout_sec)
    except McpError as exc:
        return ReferenceResult(
            BaselineInfo(status="unavailable", machine=machine, source_url=url, detail=str(exc))
        )

    if not runs:
        return ReferenceResult(
            BaselineInfo(
                status="not_found",
                machine=machine,
                source_url=url,
                detail=(f"no earlier run on {machine} with purpose like "
                        f"{config.reference_purpose_like!r}"),
            )
        )

    newest = runs[0]
    run_ids = [str(row.get("run_id") or "") for row in runs if row.get("run_id")]
    try:
        perf_rows = run_sql(url, _perf_sql(run_ids), timeout=config.mcp_timeout_sec)
    except McpError as exc:
        return ReferenceResult(
            BaselineInfo(status="unavailable", machine=machine, source_url=url, detail=str(exc))
        )

    values, history = _split_perf(perf_rows, newest_run_id=run_ids[0], order=run_ids)
    info = BaselineInfo(
        status="found",
        run_id=run_ids[0],
        stamp=newest.get("stamp") or None,
        ov_version=newest.get("ov_version") or None,
        machine=newest.get("machine") or machine,
        source_url=url,
        selection_reason=(f"newest of {len(run_ids)} run(s) on {machine} with purpose like "
                          f"{config.reference_purpose_like!r} (daily_results)"),
    )
    return ReferenceResult(info, values, history)


def fetch_release(config: AnalysisConfig, machine: str | None) -> tuple[ReleaseInfo, SeriesValues]:
    """Return the newest release run for *machine* and its per-series values."""
    if not config.release_enabled:
        return ReleaseInfo(status="disabled"), {}

    url = config.mcp_url
    if not machine:
        return ReleaseInfo(status="not_found", source_url=url,
                           detail="current run has no machine name"), {}

    from common.mcp_client import McpError, run_sql

    try:
        runs = run_sql(url, _latest_release_sql(machine, config.release_purpose_like),
                       timeout=config.mcp_timeout_sec)
    except McpError as exc:
        return ReleaseInfo(status="unavailable", machine=machine,
                           source_url=url, detail=str(exc)), {}

    if not runs:
        return (
            ReleaseInfo(
                status="not_found",
                machine=machine,
                source_url=url,
                detail=f"no run on {machine} with purpose like {config.release_purpose_like!r}",
            ),
            {},
        )

    run = runs[0]
    run_id = str(run.get("run_id") or "")
    try:
        perf_rows = run_sql(url, _perf_sql([run_id]), timeout=config.mcp_timeout_sec)
    except McpError as exc:
        return ReleaseInfo(status="unavailable", machine=machine,
                           source_url=url, detail=str(exc)), {}

    values, _ = _split_perf(perf_rows, newest_run_id=run_id, order=[run_id])
    return (
        ReleaseInfo(
            status="found",
            run_id=run_id,
            stamp=run.get("stamp") or None,
            ov_version=run.get("ov_version") or None,
            machine=run.get("machine") or machine,
            source_url=url,
            matched_count=len(values),
        ),
        values,
    )


# ---------------------------------------------------------------------------
# SQL builders — the MCP tool takes a bare statement, so literals are inlined
# ---------------------------------------------------------------------------

def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _reference_runs_sql(config: AnalysisConfig, rec: "RunRecord") -> str:
    short_run = "TRUE" if getattr(rec, "short_run", False) else "FALSE"
    ts = getattr(rec, "ts", None)
    ts_clause = f"AND ts < TIMESTAMP {_quote(ts.strftime('%Y-%m-%d %H:%M:%S'))} " if ts else ""
    return (
        "SELECT run_id, strftime(ts, '%Y%m%d_%H%M') AS stamp, "
        "COALESCE(ov_version, '') AS ov_version, machine "
        "FROM runs "
        f"WHERE machine = {_quote(rec.machine)} "
        f"AND lower(COALESCE(purpose, '')) LIKE lower({_quote(config.reference_purpose_like)}) "
        f"AND COALESCE(short_run, FALSE) = {short_run} "
        f"{ts_clause}"
        f"ORDER BY ts DESC LIMIT {int(config.history_window)}"
    )


def _latest_release_sql(machine: str, purpose_like: str) -> str:
    return (
        "SELECT run_id, strftime(ts, '%Y%m%d_%H%M') AS stamp, "
        "COALESCE(ov_version, '') AS ov_version, machine "
        "FROM runs "
        f"WHERE machine = {_quote(machine)} "
        f"AND lower(COALESCE(purpose, '')) LIKE lower({_quote(purpose_like)}) "
        "ORDER BY ts DESC LIMIT 1"
    )


def _perf_sql(run_ids: list[str]) -> str:
    # Grouping guards against a series ever being stored once per prompt_idx.
    ids = ", ".join(_quote(run_id) for run_id in run_ids)
    return (
        "SELECT run_id, model, precision, in_token, out_token, exec_mode, "
        "min(unit) AS unit, avg(value) AS value "
        "FROM perf "
        f"WHERE run_id IN ({ids}) "
        "GROUP BY run_id, model, precision, in_token, out_token, exec_mode"
    )


# ---------------------------------------------------------------------------
# Row shaping
# ---------------------------------------------------------------------------

def _split_perf(
    rows: list[dict],
    *,
    newest_run_id: str,
    order: list[str],
) -> tuple[SeriesValues, SeriesHistory]:
    """Split flat perf rows into the newest run's values and the full history."""
    rank = {run_id: i for i, run_id in enumerate(order)}
    by_run: dict[str, SeriesValues] = {}
    for row in rows:
        parsed = _parse_row(row)
        if parsed is None:
            continue
        key, value, unit = parsed
        by_run.setdefault(str(row.get("run_id") or ""), {})[key] = (value, unit)

    values = by_run.get(newest_run_id, {})
    history: SeriesHistory = {}
    for run_id in sorted(by_run, key=lambda r: rank.get(r, len(order))):
        for key, (value, _unit) in by_run[run_id].items():
            history.setdefault(key, []).append(value)
    return values, history


def _parse_row(row: dict) -> tuple[SeriesKeyTuple, float, str | None] | None:
    try:
        key = (
            row["model"],
            row["precision"],
            int(row["in_token"]),
            int(row["out_token"]),
            row["exec_mode"],
        )
        value = float(row["value"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return key, value, row.get("unit")
