"""Command-line ingest for daily benchmark artefacts.

Usage::

    # Auto-detect: scan a directory and ingest whatever it finds.
    python -m viewer.ingest.cli --root /var/www/html/daily --db bench.duckdb

    # Single file.
    python -m viewer.ingest.cli --input output/daily.20260421_2234.summary.json
    python -m viewer.ingest.cli --input res/daily.20250224_0104.2025.1.0-18257-f77ef0f25b4.pickle

    # Force format.
    python -m viewer.ingest.cli --root /var/www/html/daily --format old

Files already present in ``runs.file_hash`` are skipped unless --force.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

from .loader_new import load_summary
from .loader_old import load_report
from .writer import (already_ingested, connect, ensure_schema,
                     load_display_profile, upsert_run)

log = logging.getLogger("ingest")

DEFAULT_DB = Path(__file__).resolve().parent.parent / "bench.duckdb"
DEFAULT_PROFILE = (Path(__file__).resolve().parent.parent
                   / "profiles" / "default.yaml")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _iter_new_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("daily.*.summary.json"))


def _iter_old_files(root: Path) -> Iterable[Path]:
    """Old format is ``.pickle`` + sibling ``.report``; we key off the pickle."""
    for pkl in sorted(root.rglob("daily.*.pickle")):
        if pkl.with_suffix(".report").exists():
            yield pkl


def _classify(path: Path) -> str | None:
    name = path.name
    if name.endswith(".summary.json"):
        return "new"
    if name.endswith(".pickle"):
        return "old"
    if name.endswith(".report"):
        return "old" if path.with_suffix(".pickle").exists() else None
    return None


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _progress(done: int, total: int, extra: str = "") -> None:
    if total <= 0:
        return
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = done / total * 100
    print(f"\r[{bar}] {done}/{total} ({pct:5.1f}%) {extra}",
          end="", flush=True)


# ---------------------------------------------------------------------------
# Ingest drivers
# ---------------------------------------------------------------------------

def ingest_files(files: list[tuple[Path, str]], db_path: Path,
                 *, force: bool = False) -> tuple[int, int, list[tuple[Path, str]]]:
    """Ingest a list of (path, format) pairs. Returns (added, skipped, failures)."""
    con = connect(db_path)
    ensure_schema(con)

    added = skipped = 0
    failures: list[tuple[Path, str]] = []
    total = len(files)
    for idx, (path, fmt) in enumerate(files, start=1):
        try:
            if fmt == "new":
                rec = load_summary(path)
            elif fmt == "old":
                rec = load_report(path)
            else:
                raise ValueError(f"unknown format {fmt!r}")

            if not force and already_ingested(con, rec.file_hash):
                skipped += 1
            else:
                upsert_run(con, rec)
                added += 1
            _progress(idx, total, f"added={added} skipped={skipped} | {path.name}")
        except Exception as e:  # noqa: BLE001 — we want to keep going
            failures.append((path, str(e)))
            _progress(idx, total, f"FAIL {path.name}: {e}")
    print()
    con.close()
    return added, skipped, failures


def discover(root: Path, *, fmt: str) -> list[tuple[Path, str]]:
    if fmt in ("new", "auto"):
        new_files = [(p, "new") for p in _iter_new_files(root)]
    else:
        new_files = []
    if fmt in ("old", "auto"):
        old_files = [(p, "old") for p in _iter_old_files(root)]
    else:
        old_files = []
    return new_files + old_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", type=Path,
                       help="Scan a directory tree for daily artefacts.")
    group.add_argument("--input", type=Path,
                       help="Ingest a single summary.json/.pickle/.report file.")

    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--format", choices=("auto", "old", "new"), default="auto")
    ap.add_argument("--force", action="store_true",
                    help="Re-ingest files even if the hash matches.")
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE,
                    help="Display profile YAML to (re-)load into display_rows.")
    ap.add_argument("--skip-profile", action="store_true",
                    help="Don't touch display_rows.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Profile load happens once per invocation.
    if not args.skip_profile and args.profile and args.profile.exists():
        con = connect(args.db)
        ensure_schema(con)
        n = load_display_profile(con, args.profile)
        con.close()
        log.info("Loaded %d display rows from %s", n, args.profile)

    if args.input:
        fmt = args.format
        if fmt == "auto":
            fmt = _classify(args.input)
            if fmt is None:
                print(f"[ingest] cannot classify {args.input}", file=sys.stderr)
                return 2
        files = [(args.input, fmt)]
    else:
        files = discover(args.root, fmt=args.format)
        if not files:
            print(f"[ingest] no files found under {args.root}")
            return 0

    t0 = time.time()
    added, skipped, failures = ingest_files(files, args.db, force=args.force)
    elapsed = time.time() - t0

    print(f"[ingest] added={added} skipped={skipped} "
          f"failed={len(failures)} total={len(files)} "
          f"elapsed={elapsed:.1f}s db={args.db}")
    if failures:
        print("[ingest] failures:")
        for p, err in failures[:20]:
            print(f"  - {p}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
