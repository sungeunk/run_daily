#!/usr/bin/env python3
"""Generate a distribution-aware HTML analysis report from daily run results.

Reads from an existing DuckDB (runs/perf) and selects a run
(latest by default, or explicit --run-id/--stamp), then renders HTML using
the history-based fluctuation-guard analysis engine.

Before selecting a run, it ingests the latest artefact from --root into DB,
so the newest run is always available in DB first.

The output file is always a *new* file (never overwrites an existing one).

Usage::

    # default: use existing bench.duckdb and latest run
    python scripts/generate_analysis_report.py

    # pick a past run from DB by run_id (or by stamp)
    python scripts/generate_analysis_report.py --run-id daily.20260530_0315.report
    python scripts/generate_analysis_report.py --stamp 20260530_0315

    # explicit DB path
    python scripts/generate_analysis_report.py --db daily/viewer/bench.duckdb

    # ingest from custom root and write to custom output directory
    python scripts/generate_analysis_report.py \\
        --root /var/www/html/daily2/ARLH-01 \\
        --out-dir /tmp/reports

    # tune analysis parameters
    python scripts/generate_analysis_report.py \\
        --history-window 15 --reference-top-k 7 --fluctuation-scale 2.0

Quick re-run alias (runs from any directory)::

    PYTHONPATH=<repo>/daily python scripts/generate_analysis_report.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_ROOT = Path('/var/www/html/daily2/dg2alderlake')
DEFAULT_DB = Path(__file__).resolve().parent.parent / 'daily' / 'viewer' / 'bench.duckdb'
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


def _resolve_stamp_from_name(name: str) -> str:
    m = STAMP_RE.search(name)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{8}_\d{4})", name)
    if m2:
        return m2.group(1)
    return datetime.now().strftime('%Y%m%d_%H%M')


def _stamp_of_any(path: Path) -> str:
    name = path.name
    m = re.search(r"daily\.(\d{8}_\d{4})", name)
    if m:
        return m.group(1)
    return "00000000_0000"


def _ingest_from_root(*, root: Path, db_path: Path) -> tuple[int, int, int, str | None]:
    """Ingest artefacts from root into DB.

    Returns tuple: (candidates, added, skipped, latest_run_id)
    """
    from viewer.ingest.cli import discover, ingest_files
    from viewer.ingest.loader_new import load_summary
    from viewer.ingest.loader_old import load_report

    if not root.exists() or not root.is_dir():
        return (0, 0, 0, None)

    files = discover(root, fmt="auto")
    if not files:
        return (0, 0, 0, None)

    latest_path, latest_fmt = max(
        files,
        key=lambda pf: (_stamp_of_any(pf[0]), int(pf[0].stat().st_mtime)),
    )
    if latest_fmt == "new":
        latest_run_id = load_summary(latest_path).run_id
    else:
        latest_run_id = load_report(latest_path).run_id

    added, skipped, failures = ingest_files(files, db_path, force=False)
    if failures:
        raise RuntimeError(f"ingest failed: {files[0][0]} -> {failures[0][1]}")
    return (len(files), added, skipped, latest_run_id)


def _pick_run_from_db(con, *, run_id: str | None, stamp: str | None):
    if run_id and stamp:
        raise ValueError('Use only one of --run-id or --stamp')

    if run_id:
        row = con.execute(
            """
            SELECT run_id, source_path, report_file, ts, machine
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        return row

    if stamp:
        # Accept either exact stamp or any run_id/report_file containing the stamp.
        row = con.execute(
            """
            SELECT run_id, source_path, report_file, ts, machine
            FROM runs
            WHERE run_id LIKE '%' || ? || '%'
               OR report_file LIKE '%' || ? || '%'
               OR source_path LIKE '%' || ? || '%'
            ORDER BY ts DESC
            LIMIT 1
            """,
            [stamp, stamp, stamp],
        ).fetchone()
        return row

    row = con.execute(
        """
        SELECT run_id, source_path, report_file, ts, machine
        FROM runs
                WHERE COALESCE(source_path, '') NOT LIKE '/tmp/%'
                    AND machine NOT LIKE 'tmp%'
        ORDER BY ts DESC
        LIMIT 1
        """
    ).fetchone()
    return row


def _build_functional_from_db(con, run_id: str):
    from analysis.types import FunctionalIssue, FunctionalResult

    issue_rows = con.execute(
        """
        SELECT nodeid, outcome, COALESCE(message, '')
        FROM functional_issues
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()

    issues = [FunctionalIssue(nodeid=n, outcome=o, message=m) for n, o, m in issue_rows]
    failed = sum(1 for i in issues if i.outcome == 'failed')
    errored = sum(1 for i in issues if i.outcome == 'error')
    skipped = sum(1 for i in issues if i.outcome == 'skipped')

    row = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM perf WHERE run_id = ?) AS perf_count
        """,
        [run_id],
    ).fetchone()
    perf_count = int(row[0] or 0)

    total = max(perf_count, failed + errored + skipped)
    passed = max(total - failed - errored - skipped, 0)
    return FunctionalResult(
        total=total,
        passed=passed,
        failed=failed,
        error=errored,
        skipped=skipped,
        issues=issues,
    )


def _build_current_run_info(con, run_id: str):
    from analysis.types import CurrentRunInfo

    run_row = con.execute(
        """
        SELECT ov_version, purpose, machine, device, description,
               host_info, host_memory_size_gb, host_memory_speed_mhz
        FROM runs
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run_row is None:
        return CurrentRunInfo()

    device_rows = con.execute(
        """
        SELECT device_index, device, driver, eu, clock_freq_mhz, global_mem_size_gb
        FROM system_devices
        WHERE run_id = ?
        ORDER BY device_index
        """,
        [run_id],
    ).fetchall()

    (
        ov_version,
        purpose,
        machine,
        run_device,
        description,
        host_info_db,
        host_memory_size_gb,
        host_memory_speed_mhz,
    ) = run_row

    primary_driver = None
    gpu_parts: list[str] = []
    mem_sizes: list[str] = []
    for _device_index, device, driver, eu, clock_freq_mhz, global_mem_size_gb in device_rows:
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

    gpu_info = "; ".join(gpu_parts) if gpu_parts else (str(run_device) if run_device else None)
    memory_size = ", ".join(mem_sizes) if mem_sizes else None

    host_info = str(host_info_db) if host_info_db else (description if description and description != purpose else None)
    memory_size = (f"{float(host_memory_size_gb):.1f} GB" if host_memory_size_gb is not None else None) or memory_size
    memory_speed = f"{float(host_memory_speed_mhz):.0f} MHz" if host_memory_speed_mhz is not None else None

    return CurrentRunInfo(
        ov_version=str(ov_version) if ov_version else None,
        purpose=str(purpose) if purpose else None,
        machine_name=str(machine) if machine else None,
        gpu_driver_version=primary_driver,
        gpu_info=gpu_info,
        host_info=host_info,
        memory_size=memory_size,
        memory_speed=memory_speed,
    )


def _analyze_run_id_from_db(con, run_id: str, cfg):
    from analysis.baseline import find_last_known_good, select_baseline
    from analysis.engine import (
        _aggregate_models,
        _aggregate_performance,
        _build_bisect_delta,
        _fetch_comparison_rows,
        _overall_status,
        _top_regressions,
    )
    from analysis.types import AnalysisResult

    rec_row = con.execute(
        """
        SELECT run_id, machine, ts, short_run, purpose
        FROM runs
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if rec_row is None:
        raise ValueError(f'run_id not found in DB: {run_id}')

    rec = SimpleNamespace(
        run_id=rec_row[0],
        machine=rec_row[1],
        ts=rec_row[2],
        short_run=bool(rec_row[3]) if rec_row[3] is not None else False,
        purpose=rec_row[4],
    )

    functional = _build_functional_from_db(con, run_id)
    current_run = _build_current_run_info(con, run_id)
    baseline_info = select_baseline(con, rec, cfg)
    rows = _fetch_comparison_rows(con, rec, baseline_info, cfg)
    performance = _aggregate_performance(rows)
    models = _aggregate_models(rows)
    top_regressions = _top_regressions(rows, cfg.top_regressions)
    overall_status = _overall_status(functional, performance, baseline_info)

    last_known_good = None
    bisect_delta = None
    if overall_status in {"red", "yellow"}:
        last_known_good = find_last_known_good(con, rec)
        bisect_delta = _build_bisect_delta(
            con,
            current_run_id=run_id,
            lkg=last_known_good,
            functional_issue_count=functional.issue_count,
            config=cfg,
        )

    return AnalysisResult(
        overall_status=overall_status,
        baseline=baseline_info,
        functional=functional,
        performance=performance,
        models=models,
        top_regressions=top_regressions,
        rows=rows,
        current_run=current_run,
        last_known_good=last_known_good,
        bisect_delta=bisect_delta,
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
        '--db', type=Path, default=DEFAULT_DB,
        help='DuckDB path (runs/perf source) for DB mode.',
    )
    ap.add_argument(
        '--run-id', type=str, default=None,
        help='Run ID to analyze when using --db mode. Default: latest run by ts.',
    )
    ap.add_argument(
        '--stamp', type=str, default=None,
        help='Stamp selector (e.g. 20260601_0315) when using --db mode.',
    )
    ap.add_argument(
        '--root', type=Path, default=DEFAULT_ROOT,
        help='Input root to ingest latest artefact into DB before analysis.',
    )
    ap.add_argument(
        '--out-dir', type=Path, default=None,
        help='Output directory for the HTML report. Default: <root>/report.',
    )
    ap.add_argument(
        '--history-window', type=int, default=10,
        help='Number of past runs to use for reference distribution.',
    )
    ap.add_argument(
        '--reference-top-k', type=int, default=5,
        help='Top-K best runs to average as the reference value.',
    )
    ap.add_argument(
        '--fluctuation-scale', type=float, default=1.5,
        help='Sigma multiplier: delta <= scale*sigma is treated as same.',
    )
    ap.add_argument(
        '--pct-threshold', type=float, default=0.05,
        help='Minimum fractional change to consider improved/regressed (e.g. 0.05 = 5%%).',
    )
    args = ap.parse_args(argv)

    # Ensure daily package is importable when invoked from outside the repo.
    if str(DAILY_DIR) not in sys.path:
        sys.path.insert(0, str(DAILY_DIR))

    from analysis.report import render_analysis_html
    from analysis.types import AnalysisConfig
    from viewer.ingest.writer import connect

    cfg = AnalysisConfig(
        history_window=args.history_window,
        reference_top_k=args.reference_top_k,
        fluctuation_sigma_scale=args.fluctuation_scale,
        pct_threshold=args.pct_threshold,
    )

    db_path = args.db

    print(f'[report] mode     : db-only')
    print(f'[report] db       : {db_path}')

    # 1) Ingest latest artefact into DB first.
    try:
        candidate_count, added, skipped, latest_run_id = _ingest_from_root(root=args.root, db_path=db_path)
        if candidate_count == 0:
            if not db_path.exists():
                print(
                    f'[report] ingest   : no artefacts found under {args.root} and DB does not exist: {db_path}',
                    file=sys.stderr,
                )
                return 1
            print(f'[report] ingest   : no artefacts found under {args.root}; using existing DB rows')
        else:
            print(
                f'[report] ingest   : candidates={candidate_count} added={added} skipped={skipped}'
            )
    except Exception as e:  # noqa: BLE001
        print(f'[report] ingest failed: {e}', file=sys.stderr)
        return 1

    # 2) Select run from DB and analyze using DB rows only.
    selected_run_id = args.run_id
    if selected_run_id is None and args.stamp is None and latest_run_id is not None:
        selected_run_id = latest_run_id

    with connect(db_path, read_only=True) as con:
        picked = _pick_run_from_db(con, run_id=selected_run_id, stamp=args.stamp)
        if picked is None:
            print('[report] No runs found in DB.', file=sys.stderr)
            return 1
        run_id, source_path, report_file, _ts, machine = picked

        out_dir: Path
        if args.out_dir:
            out_dir = args.out_dir
        elif source_path:
            out_dir = Path(source_path).resolve().parent / 'report'
        elif report_file:
            report_file_path = Path(report_file)
            if report_file_path.is_absolute():
                out_dir = report_file_path.parent
            else:
                out_dir = Path.cwd() / 'report'
        else:
            out_dir = Path.cwd() / 'report'
        out_dir.mkdir(parents=True, exist_ok=True)

        now_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_stamp = _resolve_stamp_from_name(str(run_id))

        print(f'[report] current  : run_id={run_id} machine={machine}')
        result = _analyze_run_id_from_db(con, run_id, cfg)

    html = render_analysis_html(result)
    out_path = _unique_out_path(out_dir, current_stamp, now_tag)
    out_path.write_text(html, encoding='utf-8')

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
    return 0


if __name__ == '__main__':
    sys.exit(main())
