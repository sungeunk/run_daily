#!/usr/bin/env python3
"""Generate a distribution-aware HTML analysis report through daily_results MCP.

Selects a run from the central daily_results server (latest by default, or an
explicit --run-id/--stamp), then renders HTML using the history-based
fluctuation-guard analysis engine. It never reads a local DuckDB or benchmark
report artefact.

The output file is always a *new* file (never overwrites an existing one).

Usage::

    # default: select the latest central run
    python scripts/generate_analysis_report.py

    # pick a past run by run ID (or by timestamp)
    python scripts/generate_analysis_report.py --run-id <run-id>
    python scripts/generate_analysis_report.py --stamp 20260530_0315

    # select a machine and write to a custom output directory
    python scripts/generate_analysis_report.py \\
        --machine ARLH-01 \
        --out-dir /tmp/reports

    # tune analysis parameters
    python scripts/generate_analysis_report.py \\
        --history-window 15 --fluctuation-scale 2.0

    # override baseline purpose used for baseline selection
    python scripts/generate_analysis_report.py --baseline-purpose "daily2 timer"

    # also overwrite daily.<stamp>.html in out-dir for stable filename access
    python scripts/generate_analysis_report.py --write-daily-html

Quick re-run alias (runs from any directory)::

    PYTHONPATH=<repo>/daily python scripts/generate_analysis_report.py
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DAILY_DIR = Path(__file__).resolve().parent.parent / 'daily'

STAMP_RE = re.compile(r'daily\.(\d{8}_\d{4})\.summary\.json$')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stamp_of(path: Path) -> str:
    m = STAMP_RE.search(path.name)
    return m.group(1) if m else '00000000_0000'


def _unique_out_path(out_dir: Path, current_stamp: str, now_tag: str) -> Path:
    base = out_dir / f'analysis.current_{current_stamp}.generated_{now_tag}.html'
    if not base.exists():
        return base
    idx = 1
    while True:
        p = out_dir / f'analysis.current_{current_stamp}.generated_{now_tag}.{idx}.html'
        if not p.exists():
            return p
        idx += 1


def _daily_out_path(out_dir: Path, machine: str, current_stamp: str) -> Path:
    return out_dir / f'daily.{machine}.{current_stamp}.html'


def _resolve_stamp_from_name(name: str) -> str:
    m = STAMP_RE.search(name)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{8}_\d{4})", name)
    if m2:
        return m2.group(1)
    return datetime.now().strftime('%Y%m%d_%H%M')


def _stamp_from_run(run: dict) -> str:
    timestamp = run.get('ts')
    if timestamp:
        try:
            return datetime.fromisoformat(str(timestamp)).strftime('%Y%m%d_%H%M')
        except ValueError:
            pass
    return _resolve_stamp_from_name(str(run.get('run_id') or ''))


def _stamp_of_any(path: Path) -> str:
    name = path.name
    m = re.search(r"daily\.(\d{8}_\d{4})", name)
    if m:
        return m.group(1)
    return "00000000_0000"


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _pick_run_from_mcp(config, *, machine: str | None, run_id: str | None,
                       stamp: str | None, client=None) -> dict:
    from analysis.remote import _run_sql

    if run_id and stamp:
        raise ValueError('Use only one of --run-id or --stamp')

    filters = ["machine NOT LIKE 'tmp%'"]
    if machine:
        filters.append(f"machine = {_sql_quote(machine)}")
    if run_id:
        filters.append(f"run_id = {_sql_quote(run_id)}")
    elif stamp:
        filters.append(f"strftime(ts, '%Y%m%d_%H%M') = {_sql_quote(stamp)}")

    sql = (
        "SELECT run_id, ts, machine, ov_version, purpose, device, description, "
        "host_info, host_memory_size_gb, host_memory_speed_mhz, total_tests, "
        "passed_tests, failed_tests, error_tests, skipped_tests "
        f"FROM runs WHERE {' AND '.join(filters)} ORDER BY ts DESC, run_id DESC LIMIT 1"
    )
    rows = client.run_sql(sql) if client is not None else _run_sql(config, sql)
    if not rows:
        raise ValueError('No matching run found through daily_results MCP')
    return rows[0]


def _build_functional_from_mcp(config, run_id: str, run: dict, *, client=None):
    from analysis.remote import _run_sql
    from analysis.types import FunctionalIssue, FunctionalResult
    issue_rows = _run_sql(
        config,
        "SELECT nodeid, outcome, COALESCE(message, '') AS message, model, precision "
        f"FROM functional_issues WHERE run_id = {_sql_quote(run_id)} ORDER BY nodeid",
        client,
    )
    issues = [
        FunctionalIssue(
            nodeid=str(issue['nodeid']), outcome=str(issue['outcome']),
            message=str(issue['message']), model=issue.get('model'), precision=issue.get('precision'),
        )
        for issue in issue_rows
    ]
    return FunctionalResult(
        total=int(run.get('total_tests') or 0),
        passed=int(run.get('passed_tests') or 0),
        failed=int(run.get('failed_tests') or 0),
        error=int(run.get('error_tests') or 0),
        skipped=int(run.get('skipped_tests') or 0),
        issues=issues,
    )


def _build_current_run_info(config, run_id: str, run: dict, *, client=None):
    from analysis.remote import _run_sql
    from analysis.types import CurrentRunInfo
    device_rows = _run_sql(
        config,
        "SELECT device_index, device, driver, eu, clock_freq_mhz, global_mem_size_gb "
        f"FROM system_devices WHERE run_id = {_sql_quote(run_id)} ORDER BY device_index",
        client,
    )

    primary_driver = None
    gpu_parts: list[str] = []
    mem_sizes: list[str] = []
    for device_row in device_rows:
        device = device_row.get('device')
        driver = device_row.get('driver')
        eu = device_row.get('eu')
        clock_freq_mhz = device_row.get('clock_freq_mhz')
        global_mem_size_gb = device_row.get('global_mem_size_gb')
        if primary_driver is None and driver:
            primary_driver = str(driver)
        parts = []
        if device:
            parts.append(str(device))
        if eu is not None:
            parts.append(f"{int(eu)} EU")
        if clock_freq_mhz is not None:
            parts.append(f"{float(clock_freq_mhz):.0f} MHz")
        if global_mem_size_gb is not None:
            parts.append(f"{float(global_mem_size_gb):.1f} GB VRAM")
        if parts:
            gpu_parts.append(" / ".join(parts))
        if global_mem_size_gb is not None:
            mem_sizes.append(f"{float(global_mem_size_gb):.1f} GB")

    gpu_info = "; ".join(gpu_parts) if gpu_parts else (str(run.get('device')) if run.get('device') else None)
    memory_size = ", ".join(mem_sizes) if mem_sizes else None

    purpose = run.get('purpose')
    description = run.get('description')
    host_memory_size_gb = run.get('host_memory_size_gb')
    host_memory_speed_mhz = run.get('host_memory_speed_mhz')
    host_info = run.get('host_info') or (description if description and description != purpose else None)
    memory_size = (f"{float(host_memory_size_gb):.1f} GB" if host_memory_size_gb is not None else None) or memory_size
    memory_speed = f"{float(host_memory_speed_mhz):.0f} MHz" if host_memory_speed_mhz is not None else None

    return CurrentRunInfo(
        ov_version=str(run['ov_version']) if run.get('ov_version') else None,
        purpose=str(purpose) if purpose else None,
        machine_name=str(run['machine']) if run.get('machine') else None,
        gpu_driver_version=primary_driver,
        gpu_info=gpu_info,
        host_info=host_info,
        memory_size=memory_size,
        memory_speed=memory_speed,
    )


def _analyze_run_from_mcp(config, run: dict, *, client=None):
    from analysis.engine import (
        _aggregate_models,
        _aggregate_performance,
        _overall_status,
        _top_regressions,
        build_comparison_rows,
    )
    from analysis.remote import _run_sql, fetch_reference, fetch_release, fetch_series_values
    from analysis.types import AnalysisResult

    rec = SimpleNamespace(
        run_id=str(run['run_id']),
        machine=str(run['machine']),
        ts=datetime.fromisoformat(str(run['ts'])),
        is_partial=False,
        purpose=run.get('purpose'),
    )

    functional = _build_functional_from_mcp(config, rec.run_id, run, client=client)
    current_run = _build_current_run_info(config, rec.run_id, run, client=client)
    reference = fetch_reference(config, rec, client=client)
    release_info, release_values = fetch_release(config, rec.machine, client=client)
    current_values = fetch_series_values(config, rec.run_id, client=client)
    rows = build_comparison_rows(
        current_values,
        config=config,
        reference_values=reference.values,
        history_map=reference.history,
        release_values=release_values,
    )
    performance = _aggregate_performance(rows)
    models = _aggregate_models(rows)
    top_regressions = _top_regressions(rows, config.top_regressions)
    overall_status = _overall_status(functional, performance, reference.info)

    return AnalysisResult(
        overall_status=overall_status,
        baseline=reference.info,
        functional=functional,
        performance=performance,
        models=models,
        top_regressions=top_regressions,
        rows=rows,
        current_run=current_run,
        release=release_info,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '--mcp-url', type=str, default='http://dg2raptorlake.ikor.intel.com:8090/mcp',
        help='Streamable-HTTP endpoint for the daily_results MCP server.',
    )
    ap.add_argument(
        '--machine', type=str, default=None,
        help='Limit run selection to this machine.',
    )
    ap.add_argument(
        '--run-id', type=str, default=None,
        help='Run ID to analyze. Default: latest run by timestamp.',
    )
    ap.add_argument(
        '--stamp', type=str, default=None,
        help='Timestamp selector (e.g. 20260601_0315).',
    )
    ap.add_argument(
        '--out-dir', type=Path, default=None,
        help='Output directory for the HTML report. Default: ./report.',
    )
    ap.add_argument(
        '--history-window', type=int, default=10,
        help='Number of past runs to use for reference distribution.',
    )
    ap.add_argument(
        '--fluctuation-scale', type=float, default=1.5,
        help='Sigma multiplier: delta <= scale*sigma is treated as same.',
    )
    ap.add_argument(
        '--pct-threshold', type=float, default=0.05,
        help='Minimum fractional change to consider improved/regressed (e.g. 0.05 = 5%%).',
    )
    ap.add_argument(
        '--baseline-purpose', type=str, default=None,
        help='Baseline purpose override used by baseline selection logic.',
    )
    ap.add_argument(
        '--write-daily-html', action='store_true',
        help='Also overwrite daily.<machine>.<stamp>.html in out-dir.',
    )
    args = ap.parse_args(argv)

    # Ensure daily package is importable when invoked from outside the repo.
    if str(DAILY_DIR) not in sys.path:
        sys.path.insert(0, str(DAILY_DIR))

    from analysis.report import render_analysis_html
    from analysis.types import AnalysisConfig

    cfg = AnalysisConfig(
        history_window=args.history_window,
        fluctuation_sigma_scale=args.fluctuation_scale,
        pct_threshold=args.pct_threshold,
        mcp_url=args.mcp_url,
        reference_purpose_like=args.baseline_purpose or "%timer%",
    )

    print('[report] mode     : mcp-only')
    print(f'[report] mcp      : {args.mcp_url}')
    try:
        from common.mcp_client import McpHttpClient

        with McpHttpClient(cfg.mcp_url, timeout=cfg.mcp_timeout_sec) as client:
            run = _pick_run_from_mcp(
                cfg, machine=args.machine, run_id=args.run_id, stamp=args.stamp, client=client
            )
            result = _analyze_run_from_mcp(cfg, run, client=client)
    except (RuntimeError, ValueError) as exc:
        print(f'[report] MCP query failed: {exc}', file=sys.stderr)
        return 1

    out_dir = args.out_dir or Path.cwd() / 'report'
    out_dir.mkdir(parents=True, exist_ok=True)
    now_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_stamp = _stamp_from_run(run)
    print(f"[report] current  : run_id={run['run_id']} machine={run['machine']}")

    html = render_analysis_html(result)
    out_path = _unique_out_path(out_dir, current_stamp, now_tag)
    out_path.write_text(html, encoding='utf-8')

    daily_out_path = None
    if args.write_daily_html:
        daily_out_path = _daily_out_path(out_dir, str(run['machine']), current_stamp)
        daily_out_path.write_text(html, encoding='utf-8')

    p = result.performance
    f = result.functional
    b = result.baseline
    fluctuation_same = sum(1 for r in result.rows if r.within_fluctuation)

    print()
    print(f'  overall   : {result.overall_status.upper()}')
    print(f'  baseline  : {b.stamp}  ({b.selection_reason})')
    print(f'  perf      : compared={p.compared}  improved={p.improved}  '
          f'same={p.same}  regressed={p.regressed}')
    print(f'  fluct.    : {fluctuation_same} series treated as same by fluctuation guard')
    print(f'  functional: failed={f.failed}  error={f.error}  issues={f.issue_count}')
    print()
    print(f'[report] output   : {out_path}')
    if daily_out_path is not None:
        print(f'[report] output2  : {daily_out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
