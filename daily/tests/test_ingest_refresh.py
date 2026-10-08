"""Regression tests for safe, incremental database refreshes."""

from __future__ import annotations

import hashlib
import ast
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path

import pytest

from data import write
from data.ingest import writer
from data.ingest.loader_new import load_summary
from data.ingest import refresh
from data.locking import database_lock
from filelock import Timeout
from viewer.refresh import CACHE_MAX_ENTRIES, cache_version, run_refresh

pytestmark = pytest.mark.dev_only


def _summary(path: Path, *, machine: str = "TEST-01") -> bytes:
    payload = {
        "meta": {"machine": machine, "stamp": "20261008_1200"},
        "totals": {"total": 0, "passed": 0},
        "tests": [],
    }
    content = json.dumps(payload).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def test_loader_hashes_the_same_bytes_it_parses(tmp_path: Path) -> None:
    path = tmp_path / "daily.20261008_1200.summary.json"
    original = _summary(path)
    _summary(path, machine="REPLACED")

    record = load_summary(path, content=original)

    assert record.machine == "TEST-01"
    assert record.file_hash == hashlib.sha256(original).hexdigest()[:24]


def test_exclusion_edits_obey_refresh_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "test.duckdb"
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    connection = writer.connect(db)
    writer.ensure_schema(connection)
    connection.close()
    monkeypatch.setattr(write, "database_lock", lambda path: database_lock(path, timeout=0))

    with database_lock(db):
        with pytest.raises(Timeout):
            write.add_exclusion(db, "run", "TEST-01", "20261008_1200", "bad run")
        with pytest.raises(Timeout):
            write.remove_exclusion(db, "run")

    write.add_exclusion(db, "run", "TEST-01", "20261008_1200", "bad run")
    write.remove_exclusion(db, "run")


def test_loader_reads_summary_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "daily.20261008_1200.summary.json"
    original = _summary(path)
    calls: list[Path] = []
    read_bytes = Path.read_bytes

    def counted_read(source: Path) -> bytes:
        calls.append(source)
        return read_bytes(source)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    record = load_summary(path)

    assert calls == [path]
    assert record.file_hash == hashlib.sha256(original).hexdigest()[:24]


def test_refresh_noop_does_not_parse_copy_or_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    _summary(root / "daily.20261008_1200.summary.json")
    db = tmp_path / "database.duckdb"
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    first = refresh.refresh_database(root, db, profile=None)
    before = db.stat()

    def unexpected(*args, **kwargs):
        pytest.fail("unchanged refresh must not parse or copy")

    monkeypatch.setattr(refresh, "load_summary", unexpected)
    monkeypatch.setattr(refresh, "_copy_database", unexpected)
    second = refresh.refresh_database(root, db, profile=None)

    assert first["changed"] and first["added"] == 1
    assert second["changed"] is False and second["skipped"] == 1
    assert (db.stat().st_ino, db.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_refresh_changed_file_and_failure_preserve_live_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    path = root / "daily.20261008_1200.summary.json"
    original = _summary(path)
    db = tmp_path / "database.duckdb"
    refresh.refresh_database(root, db, profile=None)
    path.write_text("{")
    before = db.read_bytes()

    with pytest.raises(RuntimeError, match="live DB unchanged"):
        refresh.refresh_database(root, db, profile=None)

    assert db.read_bytes() == before
    assert not list(tmp_path.glob(".refresh-*"))
    path.write_bytes(original + b"\n")
    assert refresh.refresh_database(root, db, profile=None)["added"] == 1


def test_viewer_cache_version_tracks_db_and_code_independently(tmp_path: Path) -> None:
    db = tmp_path / "db"
    code = tmp_path / "queries.py"
    db.write_bytes(b"first")
    code.write_text("queries")
    os.utime(code, ns=(9_000_000_000_000_000_000, 9_000_000_000_000_000_000))
    before = cache_version(db, code)
    db.write_bytes(b"changed")
    assert cache_version(db, code) != before
    app = Path(__file__).resolve().parents[1] / "viewer" / "app.py"
    functions = [node for node in ast.parse(app.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("cached_")]
    assert functions
    assert all("version" in [argument.arg for argument in function.args.args] for function in functions)
    for function in functions:
        decorator = function.decorator_list[0]
        assert isinstance(decorator, ast.Call)
        assert any(keyword.arg == "max_entries" and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == "CACHE_MAX_ENTRIES" for keyword in decorator.keywords)


def test_viewer_streams_and_forwards_selected_db(tmp_path: Path) -> None:
    script = tmp_path / "refresh.sh"
    script.write_text('#!/bin/sh\nprintf "%s\\n" "$DAILY_DB_FILE" "$INGEST_LOCK_FILE"\n'
                      'printf \'REFRESH_RESULT {"changed": false, "added": 0, "skipped": 1, "timings": {}}\\n\'\n')
    script.chmod(0o700)
    db, lock = tmp_path / "chosen.duckdb", tmp_path / "chosen.lock"
    lines: list[str] = []
    result = run_refresh(script, db, lock, lines.append)
    assert lines == [str(db), str(lock)]
    assert result["changed"] is False


def test_force_processing_version_and_late_raw_trigger_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    path = root / "daily.20261008_1200.summary.json"
    _summary(path)
    db = tmp_path / "database.duckdb"
    refresh.refresh_database(root, db, profile=None)
    assert refresh.refresh_database(root, db, force=True, profile=None)["added"] == 1

    raw = root / "daily.20261008_1200.raw"
    raw.write_text("late log")
    assert refresh.refresh_database(root, db, profile=None)["added"] == 1
    with writer.connect(db, read_only=True) as con:
        assert con.execute("SELECT rawlog_path FROM runs").fetchone() == (str(raw),)
    assert refresh.refresh_database(root, db, profile=None)["changed"] is False

    monkeypatch.setattr(refresh, "processing_version", lambda: "new-parser-version")
    assert refresh.refresh_database(root, db, profile=None)["added"] == 1


def test_profile_only_change_is_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    _summary(root / "daily.20261008_1200.summary.json")
    db, profile = tmp_path / "database.duckdb", tmp_path / "profile.yaml"
    profile.write_text("profile: test\nrows: []\n")
    refresh.refresh_database(root, db, profile=profile)
    profile.write_text("profile: test\nrows:\n  - {model: llama, precision: FP16, in_spec: 32, out_spec: 128, exec_mode: 2nd}\n")

    result = refresh.refresh_database(root, db, profile=profile)

    assert result["changed"] is True and result["added"] == 0
    with writer.connect(db, read_only=True) as con:
        assert con.execute("SELECT model FROM display_rows WHERE profile = 'test'").fetchone() == ("llama",)


def test_identical_content_in_different_machine_paths_is_not_skipped(tmp_path: Path) -> None:
    from data.ingest.cli import ingest_files

    files = []
    for machine in ("FIRST", "SECOND"):
        path = tmp_path / machine / "daily.20261008_1200.summary.json"
        path.parent.mkdir()
        path.write_text('{"meta": {"stamp": "20261008_1200"}, "tests": []}')
        files.append((path, "new"))
    db = tmp_path / "database.duckdb"
    assert ingest_files(files, db) == (2, 0, [])
    assert ingest_files(files, db) == (0, 2, [])
    with writer.connect(db, read_only=True) as con:
        assert con.execute("SELECT count(*) FROM runs").fetchone() == (2,)


def test_exclusion_during_snapshot_is_applied_after_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    _summary(root / "daily.20261008_1200.summary.json")
    db = tmp_path / "database.duckdb"
    refresh.refresh_database(root, db, profile=None)
    copied, editing = Event(), Event()
    original_copy = refresh._copy_database
    original_lock = write.database_lock

    def paused_copy(source: Path, target: Path) -> None:
        original_copy(source, target)
        copied.set()
        assert editing.wait(10)

    def observed_lock(path: Path):
        editing.set()
        return original_lock(path)

    monkeypatch.setattr(refresh, "_copy_database", paused_copy)
    monkeypatch.setattr(write, "database_lock", observed_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        refreshing = pool.submit(refresh.refresh_database, root, db, force=True, profile=None)
        assert copied.wait(10)
        excluding = pool.submit(write.add_exclusion, db, "run", "TEST-01", "20261008_1200", "bad run")
        assert refreshing.result(timeout=15)["changed"]
        excluding.result(timeout=15)
    with writer.connect(db, read_only=True) as con:
        assert con.execute("SELECT reason FROM run_exclusions WHERE run_id = 'run'").fetchone() == ("bad run",)


def test_snapshot_rejects_replacement_during_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "daily.20261008_1200.summary.json"
    _summary(path)
    read_bytes = Path.read_bytes

    def replace_during_read(source: Path) -> bytes:
        content = read_bytes(source)
        source.write_bytes(content + b"\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_during_read)
    with pytest.raises(RuntimeError, match="Source changed"):
        refresh.source_snapshot(path, "version")


def test_concurrent_refreshes_publish_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    _summary(root / "daily.20261008_1200.summary.json")
    db = tmp_path / "database.duckdb"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(refresh.refresh_database, root, db, profile=None)
        second = pool.submit(refresh.refresh_database, root, db, profile=None)
        results = [first.result(timeout=15), second.result(timeout=15)]
    assert sorted(result["changed"] for result in results) == [False, True]


def test_refresh_cli_noop_force_and_lock_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    _summary(root / "daily.20261008_1200.summary.json")
    db = tmp_path / "database.duckdb"
    command = [sys.executable, "-m", "data.ingest.refresh", "--root", str(root),
               "--db", str(db), "--skip-profile", "--lock-timeout", "0"]
    for extra, expected in (([], True), ([], False), (["--force"], True)):
        process = subprocess.run(command + extra, text=True, capture_output=True, timeout=30)
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout.split("REFRESH_RESULT ", 1)[1])
        assert result["changed"] is expected
        assert result["timings"]["total_sec"] >= result["timings"]["scan_sec"]
    with database_lock(db):
        process = subprocess.run(command, text=True, capture_output=True, timeout=30)
        assert process.returncode == 75


def test_incremental_summary_change_matches_forced_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    source = root / "daily.20261008_1200.summary.json"
    original = _summary(source)
    db = tmp_path / "database.duckdb"
    refresh.refresh_database(root, db, profile=None)
    payload = json.loads(original)
    payload["meta"]["purpose"] = "updated description"
    source.write_text(json.dumps(payload))
    assert refresh.refresh_database(root, db, profile=None)["added"] == 1
    with writer.connect(db, read_only=True) as con:
        incremental = con.execute("SELECT run_id, purpose, file_hash FROM runs").fetchall()
    refresh.refresh_database(root, db, force=True, profile=None)
    with writer.connect(db, read_only=True) as con:
        assert con.execute("SELECT run_id, purpose, file_hash FROM runs").fetchall() == incremental
    assert len(incremental) == 1


def test_viewer_rejects_failed_child_even_with_result(tmp_path: Path) -> None:
    script = tmp_path / "failed.sh"
    script.write_text('#!/bin/sh\necho \'REFRESH_RESULT {"changed": true}\'\nexit 1\n')
    script.chmod(0o700)
    with pytest.raises(RuntimeError, match="exit 1"):
        run_refresh(script, tmp_path / "db", tmp_path / "lock", lambda line: None)


@pytest.mark.parametrize("force", [False, True])
def test_duplicate_run_paths_never_replace_live_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  force: bool) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    original = root / "B" / "daily.20261008_1200.summary.json"
    _summary(original)
    db = tmp_path / "database.duckdb"
    refresh.refresh_database(root, db, profile=None)
    duplicate = root / "A" / original.name
    payload = json.loads(_summary(duplicate))
    payload["meta"]["purpose"] = "conflicting copy"
    duplicate.write_text(json.dumps(payload))
    before = db.read_bytes()

    for _ in range(2):
        with pytest.raises(RuntimeError, match="Duplicate run_id") as error:
            refresh.refresh_database(root, db, profile=None, force=force)
        assert str(original) in str(error.value) and str(duplicate) in str(error.value)
        assert db.read_bytes() == before
        assert not list(tmp_path.glob(".refresh-*"))

    duplicate.unlink()
    assert refresh.refresh_database(root, db, profile=None)["changed"] is False


def test_duplicate_run_paths_fail_initial_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    for directory in ("A", "B"):
        _summary(root / directory / "daily.20261008_1200.summary.json")
    db = tmp_path / "database.duckdb"
    with pytest.raises(RuntimeError, match="Duplicate run_id"):
        refresh.refresh_database(root, db, profile=None)
    assert not db.exists()


def test_moving_a_source_without_duplicate_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    original = root / "original" / "daily.20261008_1200.summary.json"
    _summary(original)
    db = tmp_path / "database.duckdb"
    refresh.refresh_database(root, db, profile=None)
    moved = root / "moved" / original.name
    moved.parent.mkdir()
    original.rename(moved)
    assert refresh.refresh_database(root, db, profile=None)["added"] == 1
    assert refresh.refresh_database(root, db, profile=None)["changed"] is False
    with writer.connect(db, read_only=True) as con:
        assert con.execute("SELECT source_path FROM runs").fetchall() == [(str(moved),)]


@pytest.mark.parametrize("lock_setting", ["default", "environment", "argument"])
def test_shell_db_override_uses_matching_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             lock_setting: str) -> None:
    monkeypatch.delenv("INGEST_LOCK_FILE", raising=False)
    root = tmp_path / "data"
    _summary(root / "daily.20261008_1200.summary.json")
    repo = Path(__file__).resolve().parents[2]
    launcher = tmp_path / "uv"
    launcher.write_text('#!/bin/sh\nshift 4\nexec "$TEST_PYTHON" "$@"\n')
    launcher.chmod(0o700)
    selected = tmp_path / "selected" / "db.duckdb"
    lock = selected.parent / ".ingest.lock" if lock_setting == "default" else tmp_path / "explicit.lock"
    environment = {**os.environ, "UV_BIN": str(launcher), "TEST_PYTHON": sys.executable,
                   "REQUIREMENTS_FILE": str(repo / "daily" / "requirements.txt"),
                   "DAILY_DB_FILE": str(tmp_path / "unused" / "default.duckdb"),
                   "DAILY_DATA_ROOT": str(root)}
    command = ["bash", str(repo / "scripts" / "ingest_db.sh"), "--db", str(selected),
               "--skip-profile", "--lock-timeout", "0"]
    if lock_setting == "environment":
        environment["INGEST_LOCK_FILE"] = str(lock)
    elif lock_setting == "argument":
        environment["INGEST_LOCK_FILE"] = str(tmp_path / "overridden.lock")
        command += ["--lock-file", str(lock)]
    with database_lock(selected, lock_path=lock):
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=30)
        assert result.returncode == 75, result.stdout + result.stderr
        assert not selected.exists()
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert selected.is_file()
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("payload", [
    "{", "null", "[]", '{}', '{"changed": false}',
    '{"changed": 0, "added": 0, "skipped": 0, "timings": {}}',
    '{"changed": false, "added": true, "skipped": 0, "timings": {}}',
    '{"changed": false, "added": -1, "skipped": 0, "timings": {}}',
    '{"changed": false, "added": 0, "skipped": "1", "timings": {}}',
    '{"changed": false, "added": 0, "skipped": 0}',
    '{"changed": false, "added": 0, "skipped": 0, "timings": []}',
    '{"changed": false, "added": 0, "skipped": 0, "timings": {"total": "slow"}}',
    '{"changed": false, "added": 0, "skipped": 0, "timings": {"total": true}}',
    '{"changed": false, "added": 0, "skipped": 0, "timings": {"total": -1}}',
    '{"changed": false, "added": 0, "skipped": 0, "timings": {"total": NaN}}',
    '{"changed": false, "added": 0, "skipped": 0, "timings": {"total": Infinity}}',
])
def test_viewer_rejects_invalid_success_result(tmp_path: Path, payload: str) -> None:
    script = tmp_path / "invalid.sh"
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' 'REFRESH_RESULT {payload}'\n")
    script.chmod(0o700)
    with pytest.raises(RuntimeError, match="REFRESH_RESULT"):
        run_refresh(script, tmp_path / "db", tmp_path / "lock", lambda line: None)


def test_cached_query_evicts_old_versions() -> None:
    import streamlit as st

    calls: list[int] = []

    @st.cache_data(show_spinner=False, max_entries=CACHE_MAX_ENTRIES)
    def query(version: int) -> int:
        calls.append(version)
        return version

    query.clear()
    try:
        for version in range(CACHE_MAX_ENTRIES + 1):
            assert query(version) == version
        assert len(calls) == CACHE_MAX_ENTRIES + 1
        query(CACHE_MAX_ENTRIES)
        assert len(calls) == CACHE_MAX_ENTRIES + 1
        query(0)
        assert len(calls) == CACHE_MAX_ENTRIES + 2
    finally:
        query.clear()