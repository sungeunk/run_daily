"""Fold a run's per-test monitor JSONL samples into a single Parquet file.

JSONL stays the format the monitor streams to: ``MachineMonitor.stop()`` kills
the sampler with ``terminate()``, and a benchmark timeout kills it mid-sample,
so the on-disk format has to survive a process that never gets to finalise.
Parquet cannot be appended to safely under those conditions, so the conversion
happens once at the end of a run, off the measurement path.

Column types are pinned rather than inferred: fields such as ``lhm_gpu_temp_c``
are all-NULL on Linux, where ``read_json_auto`` would type them as JSON and the
resulting file would refuse to union with a Windows one.
"""

from __future__ import annotations

import time
from pathlib import Path

# Mirrors the record built by monitor_machine.sample_once(). Keys absent from a
# sample become NULL and keys not listed here are dropped, so a monitor that
# gains a field needs it added here too.
SAMPLE_COLUMNS: dict[str, str] = {
    'timestamp_utc': 'TIMESTAMP',
    't_monotonic': 'DOUBLE',
    'cpu_clock_mhz': 'DOUBLE',
    'cpu_temp_c': 'DOUBLE',
    'cpu_temp_age_ms': 'DOUBLE',
    'cpu_temp_core_max_c': 'DOUBLE',
    'cpu_temp_core_avg_c': 'DOUBLE',
    'cpu_temp_core_values_c': 'DOUBLE[]',
    'cpu_temp_source': 'VARCHAR',
    'cpu_temp_query_duration_ms': 'DOUBLE',
    'cpu_usage_percent': 'DOUBLE',
    'cpu_usage_age_ms': 'DOUBLE',
    'gpu_device_index': 'BIGINT',
    'gpu_name': 'VARCHAR',
    'gpu_source': 'VARCHAR',
    'gpu_clock_mhz': 'DOUBLE',
    'gpu_clock_request_mhz': 'DOUBLE',
    'gpu_clock_max_mhz': 'DOUBLE',
    'gpu_throttle_reasons': 'VARCHAR',
    'gpu_power_watts': 'DOUBLE',
    'gpu_utilization_percent': 'DOUBLE',
    'gpu_utilization_source': 'VARCHAR',
    'gpu_memory_used_mb': 'DOUBLE',
    'gpu_memory_total_mb': 'DOUBLE',
    'gpu_query_duration_ms': 'DOUBLE',
    'lhm_gpu_power_watts': 'DOUBLE',
    'lhm_gpu_temp_c': 'DOUBLE',
    'lhm_gpu_memory_clock_mhz': 'DOUBLE',
    'lhm_gpu_fan_rpm': 'DOUBLE',
    'lhm_gpu_sample_valid': 'BOOLEAN',
    'lhm_gpu_query_duration_ms': 'DOUBLE',
    'host_memory_speed_mts': 'DOUBLE',
    'host_memory_total_gb': 'DOUBLE',
    'host_memory_usage_percent': 'DOUBLE',
    'host_memory_available_mb': 'BIGINT',
    'host_commit_used_mb': 'BIGINT',
    'host_commit_limit_mb': 'BIGINT',
    'process_pid': 'BIGINT',
    'process_name': 'VARCHAR',
    'process_cmdline': 'VARCHAR',
    'process_parent_pid': 'BIGINT',
    'process_session_id': 'BIGINT',
    'process_create_time_utc': 'TIMESTAMP',
    'process_alive': 'BOOLEAN',
    'process_priority_class': 'VARCHAR',
    'process_priority_class_value': 'BIGINT',
    'process_cpu_affinity_mask': 'VARCHAR',
    'process_cpu_affinity_count': 'BIGINT',
    'process_power_throttling': 'VARCHAR',
    'process_page_faults_per_sec': 'DOUBLE',
    'top_cpu_processes': 'STRUCT(pid BIGINT, name VARCHAR, cpu_percent DOUBLE)[]',
    'foreground_pid': 'BIGINT',
    'process_is_foreground': 'BOOLEAN',
    'timer_resolution_current_ms': 'DOUBLE',
    'timer_resolution_minimum_ms': 'DOUBLE',
    'timer_resolution_maximum_ms': 'DOUBLE',
    'sample_duration_ms': 'DOUBLE',
}

# Identity columns prepended to every sample so parquet files from different
# machines and runs can be read as one dataset.
KEY_COLUMNS: tuple[str, ...] = ('run_id', 'machine', 'stamp', 'monitor_label',
                                'source_file')


def jsonl_files(output_dir: Path, stamp: str) -> list[Path]:
    return sorted(Path(output_dir).glob(f'daily.{stamp}.monitor.*.jsonl'))


def parquet_path(output_dir: Path, stamp: str) -> Path:
    return Path(output_dir) / f'daily.{stamp}.monitor.parquet'


def _count_lines(files: list[Path]) -> int:
    total = 0
    for f in files:
        with f.open('r', encoding='utf-8', errors='ignore') as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _columns_sql() -> str:
    entries = ', '.join(f"'{name}': '{typ}'" for name, typ in SAMPLE_COLUMNS.items())
    return '{' + entries + '}'


def convert_run(output_dir: Path, stamp: str, *, run_id: str, machine: str,
                log=print) -> Path | None:
    """Write ``daily.<stamp>.monitor.parquet``; return it, or None on failure.

    The JSONL inputs are left in place — pruning them is a separate retention
    concern, and keeping them means a failed conversion is never destructive.
    """
    output_dir = Path(output_dir)
    files = jsonl_files(output_dir, stamp)
    if not files:
        return None

    try:
        import duckdb
    except ImportError:
        log('[monitor] parquet skipped: duckdb not installed')
        return None

    out = parquet_path(output_dir, stamp)
    pattern = (output_dir / f'daily.{stamp}.monitor.*.jsonl').as_posix()
    started = time.perf_counter()

    try:
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT
                    ? AS run_id,
                    ? AS machine,
                    ? AS stamp,
                    regexp_extract(filename, 'monitor\\.(.+)\\.jsonl$', 1)
                        AS monitor_label,
                    regexp_extract(filename, '([^/\\\\]+)$', 1) AS source_file,
                    * EXCLUDE (filename)
                FROM read_json(?, format='newline_delimited',
                               columns={_columns_sql()}, filename=true)
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """, [run_id, machine, stamp, pattern])
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{out.as_posix()}')"
        ).fetchone()[0]
        con.close()
    except Exception as exc:  # noqa: BLE001 — telemetry must not fail the run
        log(f'[monitor] parquet failed: {exc}')
        out.unlink(missing_ok=True)
        return None

    expected = _count_lines(files)
    if rows != expected:
        log(f'[monitor] parquet row mismatch: {rows} != {expected}, keeping jsonl')
        out.unlink(missing_ok=True)
        return None

    elapsed = time.perf_counter() - started
    size_kb = out.stat().st_size / 1024
    log(f'[monitor] parquet: {out} ({len(files)} files, {rows} samples, '
        f'{size_kb:.0f} KB, {elapsed:.2f}s)')
    return out


def prune_jsonl(output_dir: Path, keep_days: float, log=print) -> int:
    """Delete monitor JSONL older than ``keep_days``; return how many went."""
    if keep_days <= 0:
        return 0

    cutoff = time.time() - keep_days * 86400
    removed = 0
    for f in Path(output_dir).rglob('daily.*.monitor.*.jsonl'):
        try:
            if f.stat().st_mtime >= cutoff:
                continue
            # Only drop samples that made it into a parquet file.
            stamp = f.name.split('.')[1]
            if not parquet_path(f.parent, stamp).exists():
                continue
            f.unlink()
            removed += 1
        except OSError:
            continue

    if removed:
        log(f'[monitor] pruned {removed} jsonl file(s) older than {keep_days:g}d')
    return removed


def _run_id_lookup(db_path: Path | None) -> dict[tuple[str, str], str]:
    """``(machine, stamp) -> run_id`` for runs already ingested."""
    if db_path is None or not Path(db_path).exists():
        return {}
    try:
        import duckdb

        with duckdb.connect(str(db_path), read_only=True) as con:
            rows = con.execute(
                "SELECT machine, strftime(ts, '%Y%m%d_%H%M'), run_id FROM runs"
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        print(f'[monitor] run_id lookup unavailable: {exc}')
        return {}
    return {(m, s): rid for m, s, rid in rows}


def backfill_archive(archive: Path, machine: str, run_id: str,
                     log=print) -> Path | None:
    """Convert one legacy ``*.monitor.tar.gz`` into a sibling Parquet file."""
    import shutil
    import tarfile
    import tempfile

    stamp = archive.name.split('.')[1]
    out = parquet_path(archive.parent, stamp)
    if out.exists():
        return out

    with tempfile.TemporaryDirectory(prefix='monitor-backfill-') as tmp:
        tmp_dir = Path(tmp)
        try:
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(tmp_dir, filter='data')
        except (OSError, tarfile.TarError, TypeError) as exc:
            log(f'[monitor] extract failed for {archive.name}: {exc}')
            return None

        produced = convert_run(tmp_dir, stamp, run_id=run_id, machine=machine,
                               log=log)
        if produced is None:
            return None
        shutil.move(str(produced), str(out))

    return out


def _backfill_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert legacy monitor.tar.gz archives into Parquet.')
    parser.add_argument('dirs', nargs='+', type=Path,
                        help='Machine directories under the backup root')
    parser.add_argument('--db', type=Path, default=None,
                        help='DuckDB used to resolve run_id (recommended)')
    parser.add_argument('--delete-source', action='store_true',
                        help='Remove the tar.gz once its Parquet is verified')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    run_ids = _run_id_lookup(args.db)
    converted = saved = 0

    for directory in args.dirs:
        machine = directory.resolve().name
        # Archives live either directly under the machine directory (legacy
        # flat layout) or in its <YYYY.MM> buckets.
        for archive in sorted(directory.rglob('daily.*.monitor.tar.gz')):
            stamp = archive.name.split('.')[1]
            run_id = run_ids.get((machine, stamp))
            if run_id is None:
                print(f'[monitor] {machine}/{stamp}: not in db, skipped')
                continue
            if args.dry_run:
                print(f'[monitor] would convert {archive.name} (run_id={run_id})')
                continue

            out = backfill_archive(archive, machine, run_id)
            if out is None:
                continue
            converted += 1
            saved += archive.stat().st_size - out.stat().st_size
            if args.delete_source:
                archive.unlink()

    print(f'[monitor] backfill: {converted} archive(s), '
          f'{saved / 1024 ** 2:.1f} MB saved')
    return 0


if __name__ == '__main__':
    raise SystemExit(_backfill_main())

