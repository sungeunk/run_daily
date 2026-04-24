"""DuckDB writer for RunRecord + display profile loader.

The writer is the only code that knows about the DB schema. Loaders hand
it :class:`RunRecord` instances; it upserts and commits per run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from .record import RunRecord

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
    con.execute(schema_path.read_text(encoding="utf-8"))


def already_ingested(con: duckdb.DuckDBPyConnection, file_hash: str) -> bool:
    if not file_hash:
        return False
    row = con.execute(
        "SELECT 1 FROM runs WHERE file_hash = ? LIMIT 1", [file_hash]
    ).fetchone()
    return row is not None


def upsert_run(con: duckdb.DuckDBPyConnection, rec: RunRecord) -> None:
    """Upsert a single RunRecord (runs + system_devices + perf) transactionally."""
    con.begin()
    try:
        con.execute(
            """
            INSERT INTO runs (
                run_id, source_format, report_file, machine, device,
                purpose, description, ts, ww,
                ov_version, ov_build, ov_sha,
                genai_version, genai_commit, tok_commit,
                short_run, source_path, rawlog_path, file_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                source_format = excluded.source_format,
                report_file   = excluded.report_file,
                machine       = excluded.machine,
                device        = excluded.device,
                purpose       = excluded.purpose,
                description   = excluded.description,
                ts            = excluded.ts,
                ww            = excluded.ww,
                ov_version    = excluded.ov_version,
                ov_build      = excluded.ov_build,
                ov_sha        = excluded.ov_sha,
                genai_version = excluded.genai_version,
                genai_commit  = excluded.genai_commit,
                tok_commit    = excluded.tok_commit,
                short_run     = excluded.short_run,
                source_path   = excluded.source_path,
                rawlog_path   = excluded.rawlog_path,
                file_hash     = excluded.file_hash
            """,
            [
                rec.run_id, rec.source_format, rec.report_file, rec.machine, rec.device,
                rec.purpose, rec.description, rec.ts, rec.ww,
                rec.ov_version, rec.ov_build, rec.ov_sha,
                rec.genai_version, rec.genai_commit, rec.tok_commit,
                rec.short_run, rec.source_path, rec.rawlog_path, rec.file_hash,
            ],
        )

        # Replace child rows wholesale: simpler than diffing and the run is
        # the natural unit.
        con.execute("DELETE FROM system_devices WHERE run_id = ?", [rec.run_id])
        if rec.devices:
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
                    for d in rec.devices
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
                dedup[key] = (rec.run_id, *key, p.value, p.unit)
            con.executemany(
                """
                INSERT INTO perf (
                    run_id, model, precision, in_token, out_token,
                    exec_mode, value, unit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                list(dedup.values()),
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
