"""Release-build reference pulled from the central ``daily_results`` server.

Release runs are not produced locally, so they cannot come from the local
DuckDB file. They are read over MCP from the shared server instead, which also
keeps every machine looking at the same published release numbers.
"""

from __future__ import annotations

import logging
import math

from .types import AnalysisConfig, ReleaseInfo

log = logging.getLogger(__name__)

SeriesValues = dict[tuple, tuple[float, str | None]]


def fetch_release(config: AnalysisConfig, machine: str | None) -> tuple[ReleaseInfo, SeriesValues]:
    """Return the newest release run for *machine* and its per-series values.

    Never raises: a server that is down or has no release run yet must not
    fail the daily report, so every failure degrades to an empty result.
    """
    if not config.release_enabled:
        return ReleaseInfo(status="disabled"), {}
    if not machine:
        return ReleaseInfo(status="not_found", detail="current run has no machine name"), {}

    from common.mcp_client import McpError, run_sql

    url = config.release_mcp_url
    try:
        runs = run_sql(url, _latest_release_sql(machine, config.release_purpose_like),
                       timeout=config.release_timeout_sec)
    except McpError as exc:
        log.warning("release lookup failed: %s", exc)
        return ReleaseInfo(status="unavailable", source_url=url, detail=str(exc)), {}

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
        perf_rows = run_sql(url, _release_perf_sql(run_id), timeout=config.release_timeout_sec)
    except McpError as exc:
        log.warning("release perf lookup failed: %s", exc)
        return ReleaseInfo(status="unavailable", source_url=url, detail=str(exc)), {}

    values = _to_series_values(perf_rows)
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
# Internal helpers
# ---------------------------------------------------------------------------

def _quote(value: str) -> str:
    """SQL string literal; the MCP tool takes no bind parameters."""
    return "'" + str(value).replace("'", "''") + "'"


def _latest_release_sql(machine: str, purpose_like: str) -> str:
    return (
        "SELECT run_id, "
        "strftime(ts, '%Y%m%d_%H%M') AS stamp, "
        "COALESCE(ov_version, '') AS ov_version, machine "
        "FROM runs "
        f"WHERE machine = {_quote(machine)} "
        f"AND lower(COALESCE(purpose, '')) LIKE lower({_quote(purpose_like)}) "
        "ORDER BY ts DESC LIMIT 1"
    )


def _release_perf_sql(run_id: str) -> str:
    # Several prompt_idx rows can share one series key; average them so the
    # release column matches the aggregated value shown for the current run.
    return (
        "SELECT model, precision, in_token, out_token, exec_mode, "
        "min(unit) AS unit, avg(value) AS value "
        "FROM perf "
        f"WHERE run_id = {_quote(run_id)} "
        "GROUP BY model, precision, in_token, out_token, exec_mode"
    )


def _to_series_values(rows: list[dict]) -> SeriesValues:
    out: SeriesValues = {}
    for row in rows:
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
            continue
        if math.isfinite(value):
            out[key] = (value, row.get("unit"))
    return out
