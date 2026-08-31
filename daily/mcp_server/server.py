"""Standalone MCP server exposing the daily LLM benchmark DuckDB.

Replaces the previous `gnai toolkits serve` deployment: the tools are plain
Python functions here instead of per-call subprocesses driven by YAML specs,
and no GNAI authentication layer is involved.

The DuckDB connection is always opened read-only, and `daily_results_run_sql`
additionally rejects anything but a single SELECT/WITH statement.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

DAILY_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path("/var/www/html/daily2/daily_llm_benchmark.duckdb")
MAX_ROWS = 500

if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

import duckdb  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from viewer import queries  # noqa: E402

log = logging.getLogger(__name__)

mcp = MCPServer(
    name="daily_results",
    instructions=(
        "Query local OpenVINO GPU daily/nightly LLM benchmark results "
        "(DuckDB-backed). Read-only."
    ),
)

_db_path = DEFAULT_DB_PATH


def _sanitize(value: Any) -> Any:
    """Make values strict-JSON safe (NaN/Inf, datetimes, numpy scalars)."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _dump(obj: Any) -> str:
    return json.dumps(_sanitize(obj), default=str, indent=2)


def _connect() -> duckdb.DuckDBPyConnection:
    """Read-only connection with filesystem access disabled.

    `read_only=True` alone still lets DuckDB table functions such as
    `read_text`/`read_csv_auto` read arbitrary host files, which matters
    because this server is exposed unauthenticated.
    """
    return duckdb.connect(
        str(_db_path), read_only=True,
        config={"enable_external_access": False},
    )


def _rows(sql: str) -> list[dict]:
    """Run a query and return at most MAX_ROWS rows."""
    with _connect() as con:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchmany(MAX_ROWS)]


@mcp.tool()
def daily_results_describe_schema() -> str:
    """List tables and columns available in the daily benchmark DuckDB (use
    this before writing a daily_results_run_sql query)."""
    return _dump(_rows(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ))


@mcp.tool()
def daily_results_list_machines() -> str:
    """List machines that have daily benchmark results, with the latest run
    timestamp, OpenVINO version, and pass/fail counts per machine."""
    return _dump(_rows(
        """
        SELECT r.machine,
               count(*) AS total_runs,
               max(r.ts) AS latest_run_ts,
               arg_max(r.ov_version, r.ts) AS latest_ov_version,
               arg_max(r.passed_tests, r.ts) AS latest_passed_tests,
               arg_max(r.failed_tests, r.ts) AS latest_failed_tests,
               arg_max(r.total_tests, r.ts) AS latest_total_tests
        FROM runs r
        GROUP BY r.machine
        ORDER BY r.machine
        """
    ))


@mcp.tool()
def daily_results_list_models(machine: str | None = None) -> str:
    """List benchmark model names present in the daily results, optionally
    scoped to one machine (e.g. LNL-03). Omit for all machines."""
    return _dump(queries.list_models(_db_path, machine=machine))


@mcp.tool()
def daily_results_recent_runs(machine: str, limit: int = 10) -> str:
    """List the most recent daily benchmark runs for a machine (timestamp,
    OpenVINO version/build/sha, purpose, pass/fail counts).

    Args:
        machine: Machine name, e.g. LNL-03, PTLH-01.
        limit: Max number of runs to return, newest first.
    """
    df = queries.recent_runs(_db_path, machine, limit=limit)
    return _dump(df.to_dict(orient="records"))


@mcp.tool()
def daily_results_perf_history(
    machine: str,
    model: str,
    precision: str,
    in_token: int,
    out_token: int,
    exec_mode: str,
    days: int = 60,
) -> str:
    """Time series of one benchmark point (machine + model + precision +
    in/out token counts + exec_mode) over time, with rolling 30-day
    median/MAD baseline, z-score and pct-diff per point.

    Use daily_results_list_models and daily_results_run_sql
    (SELECT DISTINCT precision, in_token, out_token, exec_mode FROM perf ...)
    to discover valid values first.

    Args:
        machine: Machine name, e.g. LNL-03.
        model: Model name as it appears in the perf table.
        precision: Precision string as stored in perf.precision, e.g. INT4.
        in_token: Input token count (perf.in_token).
        out_token: Output token count (perf.out_token).
        exec_mode: Execution mode as stored in perf.exec_mode, e.g. 1st, 2nd.
        days: How many days of history to include.
    """
    df = queries.series_history(
        _db_path, machine, model, precision, in_token, out_token, exec_mode,
        days=days,
    )
    return _dump(df.to_dict(orient="records"))


@mcp.tool()
def daily_results_trend_regressions(
    machine: str,
    recent_days: int = 7,
    baseline_days: int = 21,
) -> str:
    """Regression scan for a machine: compares the median of each benchmark
    series over the last recent_days vs. the baseline_days before that, using
    MAD-based robust stats. Returns one row per series, sorted worst-first by
    worsening_pct (positive = worse regardless of unit); status is
    'insufficient_data' when a window doesn't have enough points.

    Args:
        machine: Machine name, e.g. LNL-03.
        recent_days: Size of the "recent" window in days.
        baseline_days: Size of the baseline window preceding the recent one.
    """
    df = queries.trend_regressions(
        _db_path, machine, recent_days=recent_days, baseline_days=baseline_days,
    )
    return _dump(df.to_dict(orient="records"))


_BLOCKED_KEYWORDS = (
    "ATTACH", "DETACH", "COPY", "PRAGMA", "INSTALL", "LOAD",
    "CALL", "EXPORT", "IMPORT", "SET", "CREATE", "INSERT",
    "UPDATE", "DELETE", "DROP", "ALTER",
)


def _validate_sql(sql: str) -> str:
    stripped = sql.strip().rstrip(";").strip()
    # Blank out string literals so keywords inside them neither trip the
    # denylist nor hide a statement separator.
    probe = re.sub(r"'(?:''|[^'])*'", "''", stripped)
    if ";" in probe:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"^(SELECT|WITH)\b", probe, re.IGNORECASE):
        raise ValueError("Only SELECT/WITH (read-only) statements are allowed.")
    for kw in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", probe, re.IGNORECASE):
            raise ValueError(f"Statement contains a disallowed keyword: {kw}")
    return stripped


@mcp.tool()
def daily_results_run_sql(sql: str) -> str:
    """Run an arbitrary read-only SQL query (SELECT/WITH only) against the
    daily benchmark DuckDB, for questions the other daily_results_* tools
    don't cover directly. Call daily_results_describe_schema first to learn
    table/column names.

    Key tables: runs (one row per benchmark run: machine, ts, ov_version,
    pass/fail counts), perf (raw per-series numbers: run_id, model, precision,
    in_token, out_token, exec_mode, value, unit), analysis_comparisons
    (per-series verdicts vs. baseline). The connection is read-only with
    filesystem access disabled, only a single SELECT/WITH statement is
    accepted, and at most 500 rows are returned.

    Args:
        sql: A single read-only SQL SELECT/WITH statement (DuckDB dialect).
    """
    return _dump(_rows(_validate_sql(sql)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="Path to the daily benchmark DuckDB file.")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"),
                        default="streamable-http",
                        help="stdio for local clients, streamable-http for "
                             "network clients.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address for streamable-http.")
    parser.add_argument("--port", type=int, default=8090,
                        help="Bind port for streamable-http.")
    args = parser.parse_args()

    global _db_path
    _db_path = args.db
    if not _db_path.exists():
        parser.error(f"DuckDB file not found: {_db_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        log.info("Serving daily_results MCP on http://%s:%d/mcp (db=%s)",
                 args.host, args.port, _db_path)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
