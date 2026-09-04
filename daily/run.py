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
import shutil
import subprocess
import sys
import ctypes
from ctypes import wintypes
from pathlib import Path


DAILY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DAILY_DIR.parent
sys.path.insert(0, str(DAILY_DIR))

from common.paths import machine_dir, run_dir  # noqa: E402

# Pre-dated location of the viewer DB, relocated under the output root on
# first use so a machine's history travels with its artefacts.
LEGACY_VIEWER_DB = DAILY_DIR / 'viewer' / 'bench.duckdb'


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


def _now_stamp(root: Path | None = None) -> str:
    """Run stamp, unique among the artefacts already under ``root``.

    Stamps only carry minutes, so back-to-back runs started within the same
    minute would overwrite each other's report files. Keep the format (it is
    parsed by the viewer, the month bucketing and the baseline queries) and
    move to the next free minute instead.
    """
    now = dt.datetime.now()
    if root is None:
        return now.strftime('%Y%m%d_%H%M')
    for _ in range(60):
        stamp = now.strftime('%Y%m%d_%H%M')
        if not (run_dir(root, stamp) / f'daily.{stamp}.pytest.json').exists():
            return stamp
        now += dt.timedelta(minutes=1)
    return now.strftime('%Y%m%d_%H%M')


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


def _adapter_name_matches(a: str | None, b: str | None) -> bool:
    """Loose match between adapter names coming from CIM, DXGI and the registry."""
    if not a or not b:
        return False

    def _norm(text: str) -> str:
        # Drop "(R)"/"(TM)"/"(16GB)" style groups so CIM, DXGI and registry names align.
        return ''.join(ch for ch in re.sub(r'\([^)]*\)', ' ', text).lower() if ch.isalnum())

    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def _windows_gpu_memory_mb(
    adapter_name: str | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(dedicated_video_mb, dedicated_system_mb, shared_mb)`` via DXGI.

    The adapter matching ``adapter_name`` is preferred so this agrees with the
    device under test; otherwise the first hardware adapter is used. Shared
    memory is the system RAM the GPU may borrow; Intel Graphics Software can
    change it, so it is read live instead of assumed.
    """
    import ctypes
    from ctypes import wintypes

    class _LUID(ctypes.Structure):
        _fields_ = [('LowPart', wintypes.DWORD), ('HighPart', ctypes.c_long)]

    class _DXGI_ADAPTER_DESC1(ctypes.Structure):
        _fields_ = [
            ('Description', ctypes.c_wchar * 128),
            ('VendorId', ctypes.c_uint),
            ('DeviceId', ctypes.c_uint),
            ('SubSysId', ctypes.c_uint),
            ('Revision', ctypes.c_uint),
            ('DedicatedVideoMemory', ctypes.c_size_t),
            ('DedicatedSystemMemory', ctypes.c_size_t),
            ('SharedSystemMemory', ctypes.c_size_t),
            ('AdapterLuid', _LUID),
            ('Flags', ctypes.c_uint),
        ]

    _DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
    # IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
    iid = (ctypes.c_byte * 16)(*(
        (0x770AAE78).to_bytes(4, 'little')
        + (0xF26F).to_bytes(2, 'little')
        + (0x4DBA).to_bytes(2, 'little')
        + bytes([0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87])
    ))

    mb = 1024 ** 2
    fallback: tuple[float | None, float | None, float | None] | None = None
    try:
        dxgi = ctypes.WinDLL('dxgi')
        factory = ctypes.c_void_p()
        if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return None, None, None

        vtbl = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        enum_adapters = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )(vtbl[12])  # IDXGIFactory1::EnumAdapters1

        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters(factory, index, ctypes.byref(adapter)) != 0:
                return fallback or (None, None, None)
            avtbl = ctypes.cast(
                adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            ).contents
            get_desc = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1)
            )(avtbl[10])  # IDXGIAdapter1::GetDesc1
            desc = _DXGI_ADAPTER_DESC1()
            if get_desc(adapter, ctypes.byref(desc)) == 0 and not (
                desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE
            ):
                found = (round(desc.DedicatedVideoMemory / mb, 1),
                         round(desc.DedicatedSystemMemory / mb, 1),
                         round(desc.SharedSystemMemory / mb, 1))
                if _adapter_name_matches(adapter_name, desc.Description):
                    return found
                if fallback is None:
                    fallback = found
            index += 1
    except Exception:
        return fallback or (None, None, None)


def _windows_gpu_shared_memory_override(
    adapter_name: str | None = None,
) -> tuple[int, bool] | None:
    """Intel ``IncreaseFixedSegment`` setting for the adapter under test.

    Returns ``(value, present)``; an absent value means the driver default is in
    use, so it is reported as ``(0, False)``. ``None`` means the adapter's
    registry key could not be read at all.
    """
    import winreg

    key_path = (r'SYSTEM\CurrentControlSet\Control\Class'
                r'\{4d36e968-e325-11ce-bfc1-08002be10318}')

    def _read(adapter) -> tuple[int, bool]:
        try:
            value, _ = winreg.QueryValueEx(adapter, 'IncreaseFixedSegment')
            return int(value), True
        except OSError:
            return 0, False

    fallback: tuple[int, bool] | None = None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if not (len(sub) == 4 and sub.isdigit()):
                    continue
                try:
                    with winreg.OpenKey(root, sub) as adapter:
                        desc, _ = winreg.QueryValueEx(adapter, 'DriverDesc')
                        if _adapter_name_matches(adapter_name, str(desc)):
                            return _read(adapter)
                        if fallback is None and 'intel' in str(desc).lower():
                            fallback = _read(adapter)
                except OSError:
                    continue
    except OSError:
        return fallback
    return fallback


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

    gpu_dedicated_memory_mb = None
    gpu_dedicated_system_memory_mb = None
    gpu_shared_memory_mb = None
    gpu_shared_memory_override = None
    gpu_shared_memory_override_present = None
    if is_windows:
        win_gpu_info, win_driver = _windows_query_gpu(device)
        if not gpu_info:
            gpu_info = win_gpu_info
        gpu_driver_version = win_driver or gpu_driver_version
        memory_speed_mhz = _windows_memory_speed_mhz()
        (gpu_dedicated_memory_mb,
         gpu_dedicated_system_memory_mb,
         gpu_shared_memory_mb) = _windows_gpu_memory_mb(win_gpu_info)
        override = _windows_gpu_shared_memory_override(win_gpu_info)
        if override is not None:
            gpu_shared_memory_override, gpu_shared_memory_override_present = override
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
        'gpu_dedicated_memory_mb': gpu_dedicated_memory_mb,
        'gpu_dedicated_system_memory_mb': gpu_dedicated_system_memory_mb,
        'gpu_shared_memory_mb': gpu_shared_memory_mb,
        'gpu_shared_memory_override': gpu_shared_memory_override,
        'gpu_shared_memory_override_present': gpu_shared_memory_override_present,
    }


def _package_version(module_name: str) -> str | None:
    """Best-effort version string for an installed OpenVINO companion package."""
    try:
        import importlib

        mod = importlib.import_module(module_name)
    except Exception:
        return None

    for attr in ('get_version', '__version__'):
        value = getattr(mod, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value:
            return str(value)
    return None


def _genai_commit() -> str | None:
    """HEAD sha of the openvino.genai checkout the benchmark script runs from."""
    genai_dir = REPO_ROOT / 'openvino.genai'
    if not (genai_dir / '.git').exists():
        return None
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(genai_dir),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip().lower()
    return sha or None


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

    # openvino.genai / openvino_tokenizers versions: the DB schema, RunRecord
    # and writer all carry these columns, but only the legacy pickle loader
    # ever populated them. Fill them here so the new summary.json path stops
    # writing NULLs and a genai-only regression stays attributable.
    genai_version = _package_version('openvino_genai')
    tok_version = _package_version('openvino_tokenizers')

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
        'genai_version':  genai_version,
        'genai_commit':   _genai_commit(),
        'tok_commit':     tok_version,
        'short_run':      bool(args.short_run),
        # Jenkins keeps the console only until log rotation, so the URL is
        # recorded per run rather than re-derived later.
        'build_url':      os.environ.get('BUILD_URL', '').strip(),
        **_collect_runtime_meta(args.device),
    }


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    from analysis.types import AnalysisConfig

    release_defaults = AnalysisConfig()
    p = argparse.ArgumentParser(
        description='Run the daily test suite and build a report.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--device', default=_default_device(),
                   help='Target OpenVINO device (override: $DAILY_DEVICE)')
    p.add_argument('--model-dir', default=_default_model_dir(),
                   help='Root directory for models')
    p.add_argument('--model-date', default='WW35_llm-optimum_2026.4.0-22930-RC1')
    p.add_argument('--cache-dir', default=str(REPO_ROOT / 'llm-cache'))
    p.add_argument('--output-dir', default=str(REPO_ROOT / 'daily_output'))
    p.add_argument('--daily-timeout', type=int, default=1800)
    p.add_argument('--short-run', action='store_true',
                   help='Use reduced token counts / iterations')
    p.add_argument('--monitor-keep-days', type=float, default=7.0,
                   help='Keep monitor JSONL this long after Parquet conversion (0 = forever)')
    p.add_argument('--verbose', action='store_true',
                   help='Also stream the raw subprocess log to the terminal')
    p.add_argument('-k', dest='keyword', default=None,
                   help='pytest -k expression. Use "|" to separate multiple keywords (e.g. "qwen3|phi-4") — converted to "or" for pytest')
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

    # --- release comparison (daily_results MCP server) ---
    p.add_argument('--no-release', dest='release', action='store_false',
                   help='Skip the release-build column in the analysis report')
    p.add_argument('--release-mcp-url',
                   default=os.environ.get('DAILY_RELEASE_MCP_URL',
                                          release_defaults.release_mcp_url),
                   help='daily_results MCP endpoint used to fetch the release run')
    p.add_argument('--release-purpose-like',
                   default=os.environ.get('DAILY_RELEASE_PURPOSE_LIKE',
                                          release_defaults.release_purpose_like),
                   help="SQL LIKE pattern matched against runs.purpose to find release runs")
    return p.parse_known_args()


def _image_assets(staged_images: dict[str, Path], root: Path,
                  baseline_stamp: str | None, *, publish: bool) -> dict[str, dict]:
    """Pair each staged image with its published URL and the baseline run's
    image for the same slot."""
    from common.delivery import (backup_server_url, image_slot_name,
                                 staged_images_for)

    baseline_slots = staged_images_for(root, baseline_stamp) if baseline_stamp else {}
    assets: dict[str, dict] = {}
    for source, staged in staged_images.items():
        baseline = baseline_slots.get(image_slot_name(staged.name))
        assets[source] = {
            'url': backup_server_url(None, staged.name) if publish else None,
            'baseline_path': baseline,
            'baseline_url': (backup_server_url(None, baseline.name)
                             if baseline and publish else None),
        }
    return assets


def _viewer_db(root: Path) -> Path:
    """Per-machine viewer DB, moving the legacy in-package one on first use."""
    db = machine_dir(root) / 'bench.duckdb'
    if not db.exists() and LEGACY_VIEWER_DB.exists():
        db.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(LEGACY_VIEWER_DB), str(db))
        print(f'[run.py] viewer db moved: {LEGACY_VIEWER_DB} -> {db}')
    return db


def _baseline_meta(root: Path, stamp: str | None, db_path: Path) -> dict:
    """``meta``-shaped view of the baseline run.

    Prefers the baseline's own summary.json anywhere in the output tree;
    falls back to the viewer DB so a rotated-away artefact still yields an
    environment comparison.
    """
    if not stamp:
        return {}
    for path in sorted(Path(root).rglob(f'daily.{stamp}.summary.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            break
        return payload.get('meta') or {}
    return _baseline_meta_from_db(db_path, stamp)


def _baseline_meta_from_db(db_path: Path, stamp: str) -> dict:
    columns = ('machine', 'purpose', 'ov_version', 'host_info',
               'host_memory_size_gb', 'host_memory_speed_mhz',
               'gpu_info', 'gpu_driver_version',
               'gpu_dedicated_memory_mb', 'gpu_shared_memory_mb')
    try:
        import duckdb
        with duckdb.connect(str(db_path), read_only=True) as con:
            row = con.execute(
                f"SELECT {', '.join(columns)} FROM runs "
                "WHERE strftime(ts, '%Y%m%d_%H%M') = ? ORDER BY ts DESC LIMIT 1",
                [stamp],
            ).fetchone()
    except Exception:  # noqa: BLE001 — the report must not fail on a missing DB
        return {}
    return {k: v for k, v in zip(columns, row or ()) if v is not None}


def _analysis_config(args: argparse.Namespace):
    """Build the analysis tuning object from the CLI options."""
    try:
        from analysis.types import AnalysisConfig
    except ImportError:
        return None

    return AnalysisConfig(
        release_enabled=args.release,
        release_mcp_url=args.release_mcp_url,
        release_purpose_like=args.release_purpose_like,
    )


def _run_analysis(html_report: Path, summary_json: Path, root: Path,
                  db_path: Path,
                  staged_images: dict[str, Path] | None = None,
                  *, publish: bool = False,
                  analysis_config=None) -> Path | None:
    """Best-effort: ingest the output tree, then run analysis and write the HTML report."""
    try:
        from viewer.ingest.cli import discover, ingest_files
        from analysis.engine import analyze_run
        from analysis.report import write_analysis_html
        from analysis.persistence import write_analysis_to_summary

        files = discover(root, fmt='auto')
        if files:
            added, skipped, failures = ingest_files(files, db_path)
            if failures:
                raise RuntimeError(
                    f'ingest failed for {len(failures)} file(s); first error: {failures[0][1]}'
                )
            print(
                f'[run.py] ingest: candidates={len(files)} added={added} '
                f'skipped={skipped} db={db_path}'
            )
        else:
            print(f'[run.py] ingest skipped: no artefacts found under {root}')

        result = analyze_run(summary_json, db_path, analysis_config)
        write_analysis_to_summary(summary_json, result)

        summary_data = json.loads(summary_json.read_text(encoding='utf-8'))
        image_assets = _image_assets(staged_images or {}, root,
                                     result.baseline.stamp, publish=publish)
        write_analysis_html(html_report, result, summary_data, image_assets,
                            _baseline_meta(root, result.baseline.stamp, db_path))
        print(f'[run.py] analysis html report: {html_report}')
        return html_report
    except Exception as exc:  # noqa: BLE001 — analysis must not fail the run
        print(f'[run.py] analysis skipped: {exc}', file=sys.stderr)
        return None


def _convert_monitor_parquet(output_dir: Path, root: Path, stamp: str, meta: dict,
                             summary_json: Path, keep_days: float) -> Path | None:
    """Fold this run's monitor JSONL into one Parquet file for publishing.

    The JSONL originals stay for ``keep_days`` so a bad conversion is
    recoverable and ``metrics.machine.file`` keeps resolving.
    """
    from common.monitor_parquet import convert_run, prune_jsonl
    from viewer.ingest._common import parse_stamp_from_name, run_id_of

    machine = meta.get('machine') or 'unknown'
    ts = parse_stamp_from_name(summary_json.name)
    if ts is None:
        print(f'[run.py] monitor parquet skipped: no stamp in {summary_json.name}',
              file=sys.stderr)
        return None

    parquet = convert_run(
        output_dir, stamp,
        run_id=run_id_of(machine, ts, summary_json.name),
        machine=machine,
    )
    if parquet is not None:
        prune_jsonl(root, keep_days)
    return parquet


def main() -> int:
    args, passthrough = _parse_args()

    root = Path(args.output_dir).resolve()
    stamp = _now_stamp(root)
    # Artefacts are bucketed exactly like the relay backup, so a run's files
    # sit at the same relative path locally and on the server.
    output_dir = run_dir(root, stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = _viewer_db(root)

    pytest_json = output_dir / f'daily.{stamp}.pytest.json'
    summary_json = output_dir / f'daily.{stamp}.summary.json'
    html_report_path = output_dir / f'daily.{stamp}.html'
    pip_freeze_file = output_dir / f'daily.{stamp}.requirements.txt'

    tests_target = args.tests or str(DAILY_DIR / 'tests')

    pytest_cmd = [
        sys.executable, '-m', 'pytest', tests_target,
        '-v',
        f'--device={args.device}',
        f'--model-dir={args.model_dir}',
        f'--model-date={args.model_date}',
        f'--cache-dir={args.cache_dir}',
        f'--output-dir={output_dir}',
        f'--daily-timeout={args.daily_timeout}',
        f'--run-stamp={stamp}',
        '--json-report',
        f'--json-report-file={pytest_json}',
        '--json-report-omit=collectors',
        '-m', 'not dev_only',
    ]
    if args.short_run:
        pytest_cmd.append('--short-run')
    if args.verbose:
        pytest_cmd.extend(['--tee-raw-log', '-s'])
    if args.keyword:
        keyword_expr = ' or '.join(k.strip() for k in args.keyword.split('|') if k.strip())
        pytest_cmd.extend(['-k', keyword_expr])
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

    # Import lazily so `python run.py --help` works without the report deps installed.
    sys.path.insert(0, str(DAILY_DIR))
    from report.builder import build_reports
    from common.delivery import (backup_server_url, mail_title_suffix,
                                 cleanup_genai_scratch, prepend_links_html,
                                 render_links_block, scp_backup, send_mail,
                                 stage_report_images, write_pip_freeze)

    extra_meta = _collect_meta(stamp, args)
    summary = build_reports(pytest_json,
                            summary_out=summary_json,
                            extra_meta=extra_meta)

    totals = summary['totals']
    print(f'[run.py] passed={totals["passed"]} failed={totals["failed"]} '
          f'total={totals["total"]}')
    print(f'[run.py] summary json:   {summary_json}')
    print(f'[run.py] pytest json:    {pytest_json}')

    # Generated images are staged every run so the next run can show them as a
    # baseline; with --backup they are also published and linked from the report.
    staged_images = stage_report_images(summary, output_dir, stamp)

    html_report = _run_analysis(html_report_path, summary_json, root, db_path,
                                staged_images, publish=args.backup,
                                analysis_config=_analysis_config(args))

    # --- post-run delivery ---
    # Find the session raw log. New naming is "daily.<stamp>.raw";
    # keep a legacy fallback for older OV-suffixed logs.
    raw_logs = sorted(output_dir.glob(f'daily.{stamp}.raw'))
    if not raw_logs:
        raw_logs = sorted(output_dir.glob(f'daily.{stamp}.*.raw'))
    monitor_parquet = _convert_monitor_parquet(
        output_dir, root, stamp, extra_meta, summary_json, args.monitor_keep_days)

    if args.pip_freeze or args.backup or args.mail:
        write_pip_freeze(pip_freeze_file)
        print(f'[run.py] pip freeze:     {pip_freeze_file}')

    # Artefacts published on the relay, in link-block order.
    to_upload = [summary_json, pytest_json, pip_freeze_file]
    if html_report:
        to_upload.append(html_report)
    if monitor_parquet:
        to_upload.append(monitor_parquet)
    to_upload.extend(raw_logs)

    # Prepend the published-artefact links *before* scp/mail so both the
    # uploaded copy and the mailed body carry them. URLs are derived from the
    # filenames, so this doesn't depend on the upload succeeding — but only
    # add them when the files will actually be published.
    if args.backup:
        links_block = render_links_block(to_upload)
        if links_block:
            if html_report:
                prepend_links_html(html_report, links_block)
            print(f'[run.py] links:          {backup_server_url(None, "")}')

        scp_backup(to_upload + list(staged_images.values()))

    if args.mail:
        if html_report:
            suffix = mail_title_suffix(summary)
            send_mail(html_report, args.mail, args.description,
                      suffix_title=suffix, now_stamp=stamp,
                      summary_json=summary_json)
        else:
            print('[run.py] mail skipped: no html report was produced',
                  file=sys.stderr)

    if raw_logs:
        print(f'[run.py] raw log:        {raw_logs[-1]}')
    else:
        print(f'[run.py] raw log:        not found for daily.{stamp}.raw')

    # Only now are the originals redundant: the report embeds thumbnails and
    # the staged copies have been published.
    removed = cleanup_genai_scratch(output_dir)
    if removed:
        print(f'[run.py] cleanup:        removed {removed} genai scratch file(s)')

    # Run completed end-to-end. Test pass/fail is reflected in the JSON
    # summary and the mail/backup artefacts; don't double-report via exit
    # code.
    return 0


if __name__ == '__main__':
    sys.exit(main())
