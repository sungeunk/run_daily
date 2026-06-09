#!/usr/bin/env python3
"""Entry point: invoke pytest, then build reports from its JSON output.

The goal is to keep pytest in charge of test execution (selection, isolation,
reporting) while this wrapper handles the bookkeeping the old
``run_llm_daily.py`` did — naming output files, building the final text
report, and exposing a single command for the daily cron.

Typical use::

    # run everything
    python daily/run.py

    # smoke-run a subset
    python daily/run.py --short-run -k llama

    # any flag after ``--`` is passed straight to pytest
    python daily/run.py -- --collect-only -q
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import ctypes
from ctypes import wintypes
from pathlib import Path


DAILY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DAILY_DIR.parent
VIEWER_DB = DAILY_DIR / 'viewer' / 'bench.duckdb'


def _default_model_dir() -> str:
    """Platform-specific default model root.

    The daily suite runs on both Windows and Ubuntu test rigs; each has its
    own canonical model layout. Fall back to the Windows path for anything
    else so the old default is preserved.
    """
    import platform
    if platform.system() == 'Linux':
        return '/var/www/html/models/daily'
    return 'c:/dev/models/daily'


def _default_device() -> str:
    """Per-machine target device.

    Different test rigs have their target accelerator wired to different
    OpenVINO device IDs (``GPU``, ``GPU.0``, ``GPU.1`` …). Rather than hard
    code, each machine sets ``DAILY_DEVICE`` once in its shell rc file.
    """
    return os.environ.get('DAILY_DEVICE', 'GPU')


def _now_stamp() -> str:
    return dt.datetime.now().strftime('%Y%m%d_%H%M')


def _windows_total_memory_gb() -> float | None:
    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', wintypes.DWORD),
            ('dwMemoryLoad', wintypes.DWORD),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]

    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return round(stat.ullTotalPhys / (1024 ** 3), 1)


def _windows_query_gpu(device: str) -> tuple[str | None, str | None]:
    ps_cmd = (
        'Get-CimInstance Win32_VideoController | '
        'Select-Object Name,DriverVersion | ConvertTo-Json -Compress'
    )
    try:
        proc = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None

    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None

    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, None

    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return None, None

    def _is_igpu(name: str) -> bool:
        n = name.lower()
        return 'uhd' in n or 'iris' in n

    selected = None
    if device == 'GPU.1':
        selected = next(
            (row for row in rows if isinstance(row, dict) and not _is_igpu(str(row.get('Name') or ''))),
            None,
        )
    elif device == 'GPU.0':
        selected = next(
            (row for row in rows if isinstance(row, dict) and _is_igpu(str(row.get('Name') or ''))),
            None,
        )

    if selected is None:
        selected = next((row for row in rows if isinstance(row, dict)), None)
    if not selected:
        return None, None

    return selected.get('Name'), selected.get('DriverVersion')


def _windows_memory_speed_mhz() -> float | None:
    ps_cmd = 'Get-CimInstance Win32_PhysicalMemory | Select-Object -ExpandProperty Speed'
    try:
        proc = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    speeds: list[float] = []
    for line in proc.stdout.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            speeds.append(float(token))
        except ValueError:
            continue
    if not speeds:
        return None
    return max(speeds)


def _collect_runtime_meta(device: str) -> dict:
    """Return best-effort runtime metadata for the current machine."""
    import platform

    is_windows = platform.system().lower() == 'windows'

    try:
        import psutil
        memory_size_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        memory_size_gb = _windows_total_memory_gb() if is_windows else None

    host_parts = [platform.system(), platform.release(), platform.machine(), platform.processor()]
    host_info = " / ".join(part for part in host_parts if part)

    gpu_info = None
    gpu_driver_version = None
    try:
        from openvino import Core

        core = Core()
        try:
            gpu_info = core.get_property(device, 'FULL_DEVICE_NAME')
        except Exception:
            gpu_info = None
    except Exception:
        pass

    if is_windows:
        win_gpu_info, win_driver = _windows_query_gpu(device)
        if not gpu_info:
            gpu_info = win_gpu_info
        gpu_driver_version = win_driver or gpu_driver_version
        memory_speed_mhz = _windows_memory_speed_mhz()
    else:
        memory_speed_mhz = None

    if not gpu_info:
        gpu_info = device

    return {
        'host_info': host_info or None,
        'host_memory_size_gb': memory_size_gb,
        'host_memory_speed_mhz': memory_speed_mhz,
        'gpu_info': gpu_info,
        'gpu_driver_version': gpu_driver_version,
    }


def _collect_meta(stamp: str, args: argparse.Namespace) -> dict:
    """Run-level metadata that downstream consumers (viewer, xlsx) need.

    Called after pytest finishes so the environment that actually ran the
    tests (sourced setupvars.sh, DAILY_DEVICE, etc.) is reflected.
    """
    import platform
    try:
        from openvino import get_version as _ov_version
        ov_version = _ov_version()
    except Exception:
        ov_version = 'none'

    # workweek from stamp (YYYYMMDD_HHMM)
    try:
        d = dt.datetime.strptime(stamp, '%Y%m%d_%H%M')
        iso = d.isocalendar()
        workweek = f'{iso.year}.WW{iso.week}.{iso.weekday}'
    except ValueError:
        workweek = 'N/A'

    # Try to split "2026.2.0-21664-ad5d8e0f99b" into build/sha; keep raw if
    # the format ever changes so we never lose the full version string.
    build, sha = '', ''
    m = re.search(r'-(\d+)-([0-9a-fA-F]{7,40})', ov_version)
    if m:
        build, sha = m.group(1), m.group(2).lower()

    return {
        'stamp':          stamp,
        'machine':        platform.node(),
        'device':         args.device,
        'description':    args.description,
        'purpose':        args.description,
        'workweek':       workweek,
        'ov_version':     ov_version,
        'ov_build':       build,
        'ov_sha':         sha,
        'short_run':      bool(args.short_run),
        **_collect_runtime_meta(args.device),
    }


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description='Run the daily test suite and build a report.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--device', default=_default_device(),
                   help='Target OpenVINO device (override: $DAILY_DEVICE)')
    p.add_argument('--model-dir', default=_default_model_dir(),
                   help='Root directory for models')
    p.add_argument('--model-date', default='WW45_llm-optimum_2025.4.0-20381-RC1')
    p.add_argument('--cache-dir', default=str(REPO_ROOT / 'llm-cache'))
    p.add_argument('--output-dir', default=str(REPO_ROOT / 'daily_output'))
    p.add_argument('--daily-timeout', type=int, default=1800)
    p.add_argument('--short-run', action='store_true',
                   help='Use reduced token counts / iterations')
    p.add_argument('-k', dest='keyword', default=None,
                   help='pytest -k expression for selecting tests')
    p.add_argument('--tests', default=None,
                   help='Test path(s) to run (defaults to daily/tests)')

    # --- post-run delivery ---
    p.add_argument('--backup', action='store_true',
                   help='scp artefacts to $MAIL_RELAY_SERVER after the run')
    p.add_argument('--mail', default='',
                   help='Comma-separated recipients. Enables mail delivery.')
    p.add_argument('--description', default='LLM',
                   help='Free-text tag used in the mail subject')
    p.add_argument('--pip-freeze', action='store_true',
                   help='Also write pip-freeze output alongside the report')

    # --- shared xlsx append ---
    p.add_argument('--xlsx-update', type=Path, default=None,
                   help='Path to OneDrive-synced master xlsx; appends a '
                        'new column with this run\'s numbers.')
    p.add_argument('--xlsx-sheet', default=None,
                   help='Sheet name in --xlsx-update (default: first sheet)')
    p.add_argument('--xlsx-key-cols', default='1,2,3,4,5',
                   help='1-based columns holding model,precision,in,out,exec')
    p.add_argument('--xlsx-header-rows', type=int, default=3,
                   help='Header row count above the first data row')
    return p.parse_known_args()


def _run_analysis(text_report: Path, summary_json: Path) -> Path | None:
    """Best-effort: ingest output_dir artefacts, then run analysis and update reports."""
    output_dir = summary_json.parent
    try:
        from viewer.ingest.cli import discover, ingest_files
        from analysis.engine import analyze_run
        from analysis.report import prepend_to_report, write_analysis_html
        from analysis.persistence import write_analysis_to_summary

        files = discover(output_dir, fmt='auto')
        if files:
            added, skipped, failures = ingest_files(files, VIEWER_DB)
            if failures:
                raise RuntimeError(
                    f'ingest failed for {len(failures)} file(s); first error: {failures[0][1]}'
                )
            print(
                f'[run.py] ingest: candidates={len(files)} added={added} '
                f'skipped={skipped} db={VIEWER_DB}'
            )
        else:
            print(f'[run.py] ingest skipped: no artefacts found under {output_dir}')

        result = analyze_run(summary_json, VIEWER_DB)
        write_analysis_to_summary(summary_json, result)

        prepend_to_report(text_report, result)
        html_report = write_analysis_html(text_report, result)
        print(f'[run.py] analysis summary prepended to {text_report}')
        print(f'[run.py] analysis html report: {html_report}')
        return html_report
    except Exception as exc:  # noqa: BLE001 — analysis must not fail the run
        print(f'[run.py] analysis skipped: {exc}', file=sys.stderr)
        return None


def main() -> int:
    args, passthrough = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = _now_stamp()
    pytest_json = output_dir / f'daily.{stamp}.pytest.json'
    summary_json = output_dir / f'daily.{stamp}.summary.json'
    text_report = output_dir / f'daily.{stamp}.report'
    pip_freeze_file = output_dir / f'daily.{stamp}.requirements.txt'

    tests_target = args.tests or str(DAILY_DIR / 'tests')

    pytest_cmd = [
        sys.executable, '-m', 'pytest', tests_target,
        '-v',
        f'--device={args.device}',
        f'--model-dir={args.model_dir}',
        f'--model-date={args.model_date}',
        f'--cache-dir={args.cache_dir}',
        f'--output-dir={args.output_dir}',
        f'--daily-timeout={args.daily_timeout}',
        '--json-report',
        f'--json-report-file={pytest_json}',
        '--json-report-omit=collectors',
        '-m', 'not dev_only',
    ]
    if args.short_run:
        pytest_cmd.append('--short-run')
    if args.keyword:
        pytest_cmd.extend(['-k', args.keyword])
    pytest_cmd.extend(passthrough)

    print(f'[run.py] pytest: {" ".join(pytest_cmd)}', flush=True)
    # pytest exit code: 0 = all pass, 1 = failures, 5 = no tests collected.
    # Individual test failures do not fail this script — callers (Jenkins,
    # cron) treat a non-zero exit as "the run itself broke" and should not
    # page on routine test regressions. Only infra issues (no pytest output
    # at all) propagate below.
    rc = subprocess.call(pytest_cmd, cwd=str(DAILY_DIR))

    if not pytest_json.exists():
        # No JSON means pytest never produced a result (config error, crash,
        # etc.). That IS an infra failure worth surfacing.
        print(f'[run.py] no pytest json at {pytest_json} (rc={rc})',
              file=sys.stderr)
        return rc or 2

    # Import lazily so `python run.py --help` works without tabulate installed.
    sys.path.insert(0, str(DAILY_DIR))
    from report.builder import build_reports
    from common.delivery import (mail_title_suffix, scp_backup, send_mail,
                                 write_pip_freeze)

    extra_meta = _collect_meta(stamp, args)
    summary = build_reports(pytest_json,
                            text_out=text_report,
                            summary_out=summary_json,
                            extra_meta=extra_meta)

    totals = summary['totals']
    print(f'[run.py] passed={totals["passed"]} failed={totals["failed"]} '
          f'total={totals["total"]}')
    print(f'[run.py] text report:    {text_report}')
    print(f'[run.py] summary json:   {summary_json}')
    print(f'[run.py] pytest json:    {pytest_json}')

    html_report = _run_analysis(text_report, summary_json)

    # --- post-run delivery ---
    # Find the session raw log — the RawLogSink names it with the OV version
    # stamp, so glob to avoid duplicating that logic.
    raw_logs = sorted(output_dir.glob(f'daily.{stamp}.*.raw'))

    if args.pip_freeze or args.backup or args.mail:
        write_pip_freeze(pip_freeze_file)
        print(f'[run.py] pip freeze:     {pip_freeze_file}')

    if args.backup:
        to_upload = [text_report, summary_json, pytest_json, pip_freeze_file]
        to_upload.extend(raw_logs)
        scp_backup(to_upload)

    if args.mail:
        suffix = mail_title_suffix(summary)
        mail_report = html_report or text_report
        send_mail(mail_report, args.mail, args.description,
                  suffix_title=suffix, now_stamp=stamp,
                  summary_json=summary_json)

    if args.xlsx_update:
        from viewer.perf_rows import flatten, as_lookup
        from viewer.xlsx_update import XlsxTarget, update_master_xlsx

        lookup = as_lookup(flatten(summary))
        meta = summary.get('meta', {})
        headers = (meta.get('ov_version', ''),
                   meta.get('workweek', ''),
                   meta.get('stamp', stamp))
        target = XlsxTarget(
            path=args.xlsx_update,
            sheet=args.xlsx_sheet,
            key_cols=tuple(int(x) for x in args.xlsx_key_cols.split(',')),
            header_rows=args.xlsx_header_rows,
        )
        matched, total = update_master_xlsx(target, lookup, headers)
        print(f'[run.py] xlsx updated: {matched}/{total} rows '
              f'→ {args.xlsx_update}')

    # Run completed end-to-end. Test pass/fail is reflected in the JSON
    # summary and the mail/backup artefacts; don't double-report via exit
    # code.
    return 0


if __name__ == '__main__':
    sys.exit(main())
