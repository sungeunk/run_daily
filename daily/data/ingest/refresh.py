"""Incremental ingestion and atomic publication of the central database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import duckdb
from filelock import Timeout

from data.locking import database_lock
from . import writer
from .loader_new import _raw_log_candidate, load_summary

Progress = Callable[[int, int, str], None]
SourceState = tuple[str, str, str]
DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "default.yaml"


class RefreshResult(TypedDict):
    changed: bool
    added: int
    skipped: int
    timings: dict[str, float]


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    content: bytes
    fingerprint: str


def processing_version() -> str:
    """Invalidate existing records when ingestion semantics or schema change."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), root / "schema.sql"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_snapshot(path: Path, version: str) -> SourceSnapshot:
    """Read stable source bytes and include non-content loader dependencies."""
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during ingestion; retry refresh: {path}")
    dependency = (
        str(path.resolve()), hashlib.sha256(content).hexdigest(),
        after.st_mtime_ns, str(_raw_log_candidate(path) or ""), version,
    )
    fingerprint = hashlib.sha256(json.dumps(dependency).encode()).hexdigest()
    return SourceSnapshot(content, fingerprint)


def _has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone())


def _ensure_tracking(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingest_sources (
            source_path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
            run_id TEXT NOT NULL, file_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingest_state (
            name TEXT PRIMARY KEY, value TEXT NOT NULL
        );
    """)


def known_sources(con: duckdb.DuckDBPyConnection) -> dict[str, SourceState]:
    """Use only fingerprints whose current run still represents that source."""
    if not _has_table(con, "ingest_sources") or not _has_table(con, "runs"):
        return {}
    return {
        source: (fingerprint, run_id, file_hash)
        for source, fingerprint, run_id, file_hash in con.execute("""
            SELECT sources.source_path, sources.fingerprint, sources.run_id, sources.file_hash
            FROM ingest_sources AS sources JOIN runs USING (run_id)
            WHERE sources.file_hash = runs.file_hash AND sources.source_path = runs.source_path
        """).fetchall()
    }


def ingest_into(con: duckdb.DuckDBPyConnection, files: list[tuple[Path, str]], *,
                force: bool = False, progress: Progress | None = None,
                timings: dict[str, float] | None = None,
                version: str | None = None,
                active_sources: set[str] | None = None
                ) -> tuple[int, int, list[tuple[Path, str]]]:
    """Skip unchanged sources before parsing; replace only affected runs."""
    _ensure_tracking(con)
    known = known_sources(con)
    paths_by_run: dict[str, set[str]] = {}
    for source_name, state in known.items():
        paths_by_run.setdefault(state[1], set()).add(source_name)
    candidates = {str(source.resolve()) for source, _ in files}
    active_sources = candidates if active_sources is None else active_sources
    run_owners = {
        state[1]: source_name for source_name, state in known.items()
        if source_name in active_sources and source_name not in candidates
    }
    version = version or processing_version()
    timings = timings if timings is not None else {}
    added = skipped = 0
    failures: list[tuple[Path, str]] = []
    for index, (source, fmt) in enumerate(files, start=1):
        path = source.resolve()
        try:
            if fmt != "new":
                raise ValueError(f"unknown format {fmt!r}")
            started = time.monotonic()
            snapshot = source_snapshot(path, version)
            timings["read_hash_sec"] = timings.get("read_hash_sec", 0.0) + time.monotonic() - started
            existing = known.get(str(path))
            if not force and existing and existing[0] == snapshot.fingerprint:
                owner = run_owners.get(existing[1])
                if owner is not None and owner != str(path):
                    raise ValueError(f"Duplicate run_id {existing[1]}: {owner} and {path}")
                run_owners[existing[1]] = str(path)
                skipped += 1
            else:
                started = time.monotonic()
                record = load_summary(path, content=snapshot.content)
                timings["parse_sec"] = timings.get("parse_sec", 0.0) + time.monotonic() - started
                owner = run_owners.get(record.run_id)
                if owner is not None and owner != str(path):
                    raise ValueError(f"Duplicate run_id {record.run_id}: {owner} and {path}")
                run_owners[record.run_id] = str(path)
                started = time.monotonic()
                writer.upsert_run(con, record)
                con.execute("""
                    INSERT INTO ingest_sources VALUES (?, ?, ?, ?)
                    ON CONFLICT (source_path) DO UPDATE SET
                        fingerprint = excluded.fingerprint, run_id = excluded.run_id,
                        file_hash = excluded.file_hash
                """, [str(path), snapshot.fingerprint, record.run_id, record.file_hash])
                for previous_path in paths_by_run.pop(record.run_id, set()):
                    known.pop(previous_path, None)
                known[str(path)] = (snapshot.fingerprint, record.run_id, record.file_hash or "")
                paths_by_run[record.run_id] = {str(path)}
                timings["write_sec"] = timings.get("write_sec", 0.0) + time.monotonic() - started
                added += 1
        except Exception as exc:
            failures.append((path, str(exc)))
        if progress is not None:
            progress(index, len(files), f"added={added} skipped={skipped} failed={len(failures)}")
    return added, skipped, failures


def _configuration(version: str, profile: Path | None) -> str:
    digest = hashlib.sha256(version.encode())
    if profile is not None:
        digest.update(str(profile.resolve()).encode())
        digest.update(profile.read_bytes() if profile.exists() else b"missing")
    return digest.hexdigest()


def _copy_database(source: Path, target: Path) -> None:
    if sys.platform.startswith("linux"):
        subprocess.run(["cp", "--reflink=auto", "--sparse=always", "--",
                        str(source), str(target)], check=True)
    else:
        shutil.copyfile(source, target)


def refresh_database(root: Path, db_path: Path, *, force: bool = False,
                     profile: Path | None = DEFAULT_PROFILE,
                     lock_path: Path | None = None, timeout: float = 60.0,
                     progress: Progress | None = None) -> RefreshResult:
    """Plan under the shared lock; publish only a complete changed snapshot."""
    started = time.monotonic()
    root, db_path = root.resolve(), db_path.resolve()
    if not root.is_dir():
        raise ValueError(f"Artifact root does not exist: {root}")
    timings: dict[str, float] = {}
    if progress is not None:
        progress(0, 0, "waiting for database lock")
    with database_lock(db_path, lock_path=lock_path, timeout=timeout):
        timings["lock_wait_sec"] = time.monotonic() - started
        planning = time.monotonic()
        version = processing_version()
        configuration = _configuration(version, profile)
        known: dict[str, SourceState] = {}
        previous_configuration = None
        if db_path.exists():
            with writer.connect(db_path, read_only=True) as con:
                known = known_sources(con)
                if _has_table(con, "ingest_state"):
                    row = con.execute("SELECT value FROM ingest_state WHERE name = 'configuration'").fetchone()
                    previous_configuration = row[0] if row else None
                if progress is not None:
                    progress(0, 0, "scanning source fingerprints")
        files = sorted(root.rglob("daily.*.summary.json"))
        changed: list[tuple[Path, str]] = []
        for index, path in enumerate(files, start=1):
            snapshot = source_snapshot(path, version)
            previous = known.get(str(path.resolve()))
            if force or not previous or previous[0] != snapshot.fingerprint:
                changed.append((path, "new"))
            if progress is not None and (index % 100 == 0 or index == len(files)):
                progress(index, len(files), f"scan changed={len(changed)}")
        timings["scan_sec"] = time.monotonic() - planning
        if not force and not changed and previous_configuration == configuration:
            timings["total_sec"] = time.monotonic() - started
            return {"changed": False, "added": 0, "skipped": len(files), "timings": timings}

        db_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".refresh-", suffix=".duckdb", dir=db_path.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            if progress is not None:
                progress(0, len(changed), "copying database snapshot")
            copying = time.monotonic()
            if db_path.exists():
                _copy_database(db_path, temporary)
            else:
                temporary.unlink()
            timings["copy_sec"] = time.monotonic() - copying
            con = writer.connect(temporary)
            try:
                if progress is not None:
                    progress(0, len(changed), "preparing schema and profile")
                preparing = time.monotonic()
                writer.ensure_schema(con)
                _ensure_tracking(con)
                if profile is not None and profile.exists():
                    if force or previous_configuration != configuration or not writer.profile_exists(
                            con, writer.profile_name_from_yaml(profile)):
                        writer.load_display_profile(con, profile)
                timings["schema_profile_sec"] = time.monotonic() - preparing
                added, skipped, failures = ingest_into(
                    con, changed, force=True, progress=progress, timings=timings, version=version,
                    active_sources={str(path.resolve()) for path in files})
                if failures:
                    details = "; ".join(f"{path}: {error}" for path, error in failures[:5])
                    raise RuntimeError(f"Ingest failed for {len(failures)} file(s); live DB unchanged: {details}")
                con.execute("""
                    INSERT INTO ingest_state VALUES ('configuration', ?)
                    ON CONFLICT (name) DO UPDATE SET value = excluded.value
                """, [configuration])
            finally:
                checkpointing = time.monotonic()
                con.close()
                timings["checkpoint_sec"] = time.monotonic() - checkpointing
            publishing = time.monotonic()
            if progress is not None:
                progress(len(changed), len(changed), "publishing database snapshot")
            os.replace(temporary, db_path)
            timings["publish_sec"] = time.monotonic() - publishing
        finally:
            temporary.unlink(missing_ok=True)
            Path(f"{temporary}.wal").unlink(missing_ok=True)
        timings["total_sec"] = time.monotonic() - started
        return {"changed": True, "added": added,
                "skipped": len(files) - len(changed) + skipped, "timings": timings}


def main(argv: list[str] | None = None) -> int:
    """Refresh a central DB without opening its live file for ingestion writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--lock-timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    def progress(done: int, total: int, detail: str) -> None:
        if done % 100 == 0 or done == total:
            print(f"[refresh] {done}/{total} {detail}", flush=True)

    try:
        result = refresh_database(
            args.root, args.db, force=args.force,
            profile=None if args.skip_profile else args.profile,
            lock_path=args.lock_file, timeout=args.lock_timeout, progress=progress)
    except Timeout:
        print("[refresh] database lock timed out", file=sys.stderr)
        return 75
    except Exception as exc:
        print(f"[refresh] {exc}", file=sys.stderr)
        return 1
    print("REFRESH_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())