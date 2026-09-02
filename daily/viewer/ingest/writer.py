"""DuckDB writer for RunRecord + display profile loader.

The writer is the only code that knows about the DB schema. Loaders hand
it :class:`RunRecord` instances; it upserts and commits per run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from .record import DeviceRecord, RunRecord

log = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


def connect(db_path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def ensure_schema(con: duckdb.DuckDBPyConnection,
                  schema_path: Path | None = None) -> None:
    schema_path = Path(schema_path or DEFAULT_SCHEMA_PATH)
    # Migrations run first: schema.sql recreates views that reference columns
    # added below, which would fail on a pre-existing DB.
    _apply_schema_migrations(con)
    con.execute(schema_path.read_text(encoding="utf-8"))


def _apply_schema_migrations(con: duckdb.DuckDBPyConnection) -> None:
    """Apply idempotent column migrations for long-lived existing DBs."""
    migrations = [
        "ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT now()",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS threshold_pct DOUBLE",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS host_info TEXT",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS host_memory_size_gb DOUBLE",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS host_memory_speed_mhz DOUBLE",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS run_kind TEXT",
        # Backfill only rows never classified: an ADD COLUMN default would
        # label historical PR/CI runs as 'daily' and quietly pull them into
        # every trend comparison.
        """
        UPDATE runs SET run_kind = CASE
            WHEN regexp_matches(lower(COALESCE(purpose, '') || ' ' || COALESCE(description, '')),
                 '\\bpr[-_# ]*\\d+|\\bpull[-_ ]?request\\b|\\bpre-?commit\\b') THEN 'pr'
            WHEN regexp_matches(lower(COALESCE(purpose, '') || ' ' || COALESCE(description, '')),
                 '\\bdaily|\\bnightly\\b|\\bweekly\\b') THEN 'daily'
            WHEN regexp_matches(lower(COALESCE(purpose, '') || ' ' || COALESCE(description, '')),
                 '\\btest|\\btrial\\b|\\bdebug\\b|\\bexperiment|\\bjenkins\\b|\\bci\\b|\\bvalidation\\b') THEN 'test'
            ELSE 'manual'
        END
        WHERE run_kind IS NULL
        """,
        "ALTER TABLE perf ADD COLUMN IF NOT EXISTS prompt_idx INTEGER DEFAULT 0",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS history_count INTEGER",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS history_median DOUBLE",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS history_mad DOUBLE",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS history_sigma DOUBLE",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS history_cv DOUBLE",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS worsening_z DOUBLE",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS reference_source TEXT",
        "ALTER TABLE analysis_comparisons ADD COLUMN IF NOT EXISTS within_fluctuation BOOLEAN",
        "ALTER TABLE functional_issues ADD COLUMN IF NOT EXISTS model TEXT",
        "ALTER TABLE functional_issues ADD COLUMN IF NOT EXISTS precision TEXT",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS total_tests INTEGER",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS passed_tests INTEGER",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS failed_tests INTEGER",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS error_tests INTEGER",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS skipped_tests INTEGER",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS skipped_cases INTEGER",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS expected_cases INTEGER",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS model_cache TEXT",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS duration_sec DOUBLE",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS build_url TEXT",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS gpu_info TEXT",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS gpu_driver_version TEXT",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS gpu_dedicated_memory_mb DOUBLE",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS gpu_shared_memory_mb DOUBLE",
    ]
    for sql in migrations:
        try:
            con.execute(sql)
        except Exception:
            # A failed statement leaves DuckDB's transaction in an aborted
            # state; without the rollback every later statement (including
            # schema.sql) fails with "Current transaction is aborted".
            try:
                con.rollback()
            except Exception:
                pass
            # Keep schema setup best-effort for mixed-version deployments.
            log.debug("schema migration skipped: %s", sql)


def already_ingested(con: duckdb.DuckDBPyConnection, file_hash: str) -> bool:
    if not file_hash:
        return False
    row = con.execute(
        "SELECT 1 FROM runs WHERE file_hash = ? LIMIT 1", [file_hash]
    ).fetchone()
    return row is not None


def profile_exists(con: duckdb.DuckDBPyConnection, profile: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM display_rows WHERE profile = ? LIMIT 1", [profile]
    ).fetchone()
    return row is not None


def upsert_run(con: duckdb.DuckDBPyConnection, rec: RunRecord) -> None:
    """Upsert a single RunRecord (runs + system_devices + perf) transactionally."""
    devices_to_write = rec.devices
    if not devices_to_write and (rec.gpu_info or rec.gpu_driver_version):
        devices_to_write = [
            DeviceRecord(
                device_index=0,
                device=rec.gpu_info,
                driver=rec.gpu_driver_version,
            )
        ]

    con.begin()
    try:
        con.execute(
            """
            INSERT INTO runs (
                run_id, source_format, report_file, machine, device,
                purpose, description, run_kind, ts, ww,
                ov_version, ov_build, ov_sha,
                host_info, host_memory_size_gb, host_memory_speed_mhz,
                gpu_info, gpu_driver_version,
                gpu_dedicated_memory_mb, gpu_shared_memory_mb,
                genai_version, genai_commit, tok_commit, model_cache,
                short_run, source_path, rawlog_path, file_hash,
                total_tests, passed_tests, failed_tests, error_tests,
                skipped_tests, skipped_cases, expected_cases,
                duration_sec, build_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                source_format = excluded.source_format,
                report_file   = excluded.report_file,
                machine       = excluded.machine,
                device        = excluded.device,
                purpose       = excluded.purpose,
                description   = excluded.description,
                run_kind      = excluded.run_kind,
                ts            = excluded.ts,
                ww            = excluded.ww,
                ov_version    = excluded.ov_version,
                ov_build      = excluded.ov_build,
                ov_sha        = excluded.ov_sha,
                host_info     = excluded.host_info,
                host_memory_size_gb = excluded.host_memory_size_gb,
                host_memory_speed_mhz = excluded.host_memory_speed_mhz,
                gpu_info      = excluded.gpu_info,
                gpu_driver_version = excluded.gpu_driver_version,
                gpu_dedicated_memory_mb = excluded.gpu_dedicated_memory_mb,
                gpu_shared_memory_mb = excluded.gpu_shared_memory_mb,
                genai_version = excluded.genai_version,
                genai_commit  = excluded.genai_commit,
                tok_commit    = excluded.tok_commit,
                model_cache   = excluded.model_cache,
                short_run     = excluded.short_run,
                source_path   = excluded.source_path,
                rawlog_path   = excluded.rawlog_path,
                file_hash     = excluded.file_hash,
                total_tests   = excluded.total_tests,
                passed_tests  = excluded.passed_tests,
                failed_tests  = excluded.failed_tests,
                error_tests   = excluded.error_tests,
                skipped_tests = excluded.skipped_tests,
                skipped_cases = excluded.skipped_cases,
                expected_cases = excluded.expected_cases,
                duration_sec  = excluded.duration_sec,
                build_url     = excluded.build_url
            """,
            [
                rec.run_id, rec.source_format, rec.report_file, rec.machine, rec.device,
                rec.purpose, rec.description, rec.run_kind, rec.ts, rec.ww,
                rec.ov_version, rec.ov_build, rec.ov_sha,
                rec.host_info, rec.host_memory_size_gb, rec.host_memory_speed_mhz,
                rec.gpu_info, rec.gpu_driver_version,
                rec.gpu_dedicated_memory_mb, rec.gpu_shared_memory_mb,
                rec.genai_version, rec.genai_commit, rec.tok_commit,
                rec.model_cache,
                rec.short_run, rec.source_path, rec.rawlog_path, rec.file_hash,
                rec.total_tests, rec.passed_tests, rec.failed_tests,
                rec.error_tests, rec.skipped_tests, rec.skipped_cases,
                rec.expected_cases, rec.duration_sec, rec.build_url,
            ],
        )

        # Replace child rows wholesale: simpler than diffing and the run is
        # the natural unit.
        con.execute("DELETE FROM system_devices WHERE run_id = ?", [rec.run_id])
        if devices_to_write:
            con.executemany(
                """
                INSERT INTO system_devices (
                    run_id, device_index, device, driver, eu,
                    clock_freq_mhz, global_mem_size_gb
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (rec.run_id, d.device_index, d.device, d.driver, d.eu,
                     d.clock_freq_mhz, d.global_mem_size_gb)
                    for d in devices_to_write
                ],
            )

        con.execute("DELETE FROM perf WHERE run_id = ?", [rec.run_id])
        if rec.perf:
            # Deduplicate on PK — a single run can accidentally contain the
            # same (model, precision, in, out, exec) twice if the pytest
            # retry logic re-ran a test. Keep the last one.
            dedup: dict[tuple, tuple] = {}
            for p in rec.perf:
                key = (p.model, p.precision, p.in_token, p.out_token, p.exec_mode)
                dedup[key] = (rec.run_id, *key, p.value, p.unit, p.prompt_idx)
            con.executemany(
                """
                INSERT INTO perf (
                    run_id, model, precision, in_token, out_token,
                    exec_mode, value, unit, prompt_idx
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                list(dedup.values()),
            )

        con.execute("DELETE FROM machine_monitor_stats WHERE run_id = ?", [rec.run_id])
        if rec.monitor:
            monitor_dedup: dict[str, tuple] = {}
            for m in rec.monitor:
                monitor_dedup[m.nodeid] = (
                    rec.run_id, m.nodeid, m.model, m.precision, m.samples,
                    m.duration_sec,
                    m.gpu_clock_mhz_mean, m.gpu_clock_mhz_min, m.gpu_clock_mhz_max,
                    m.gpu_clock_max_mhz, m.gpu_utilization_mean,
                    m.gpu_power_watts_mean, m.gpu_power_watts_max,
                    m.gpu_temp_c_mean, m.gpu_temp_c_max,
                    m.cpu_clock_mhz_mean, m.cpu_usage_percent_mean, m.cpu_temp_c_max,
                    m.host_memory_usage_mean, m.page_faults_per_sec_mean,
                    m.throttled_sample_ratio, m.throttle_reasons,
                    m.sample_duration_ms_max, m.monitor_file,
                )
            con.executemany(
                """
                INSERT INTO machine_monitor_stats (
                    run_id, nodeid, model, precision, samples, duration_sec,
                    gpu_clock_mhz_mean, gpu_clock_mhz_min, gpu_clock_mhz_max,
                    gpu_clock_max_mhz, gpu_utilization_mean,
                    gpu_power_watts_mean, gpu_power_watts_max,
                    gpu_temp_c_mean, gpu_temp_c_max,
                    cpu_clock_mhz_mean, cpu_usage_percent_mean, cpu_temp_c_max,
                    host_memory_usage_mean, page_faults_per_sec_mean,
                    throttled_sample_ratio, throttle_reasons,
                    sample_duration_ms_max, monitor_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                list(monitor_dedup.values()),
            )

        con.execute("DELETE FROM functional_issues WHERE run_id = ?", [rec.run_id])
        if rec.issues:
            issue_dedup: dict[tuple, tuple] = {}
            for i in rec.issues:
                issue_dedup[(i.nodeid, i.outcome)] = (
                    rec.run_id, i.nodeid, i.outcome, i.message,
                    i.model, i.precision,
                )
            con.executemany(
                """
                INSERT INTO functional_issues (
                    run_id, nodeid, outcome, message, model, precision
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                list(issue_dedup.values()),
            )
        con.commit()
    except Exception:
        con.rollback()
        raise


def load_display_profile(con: duckdb.DuckDBPyConnection, yaml_path: Path) -> int:
    """Load one profile YAML into display_rows. Returns number of rows written."""
    import yaml

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    profile = data["profile"]
    rows = data.get("rows", []) or []

    con.begin()
    try:
        con.execute("DELETE FROM display_rows WHERE profile = ?", [profile])
        payload = []
        for seq, r in enumerate(rows):
            payload.append((
                profile, seq,
                str(r["model"]), str(r["precision"]),
                str(r["in_spec"]), str(r["out_spec"]),
                str(r["exec_mode"]),
                r.get("label"),
            ))
        if payload:
            con.executemany(
                """
                INSERT INTO display_rows (
                    profile, seq, model, precision, in_spec, out_spec, exec_mode, label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        con.commit()
    except Exception:
        con.rollback()
        raise

    return len(rows)


def profile_name_from_yaml(yaml_path: Path) -> str:
    import yaml

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    return str(data["profile"])
