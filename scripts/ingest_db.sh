#!/usr/bin/env bash
# Refresh the central DuckDB from daily result artifacts. The viewer executes
# this tracked script directly.

set -Eeuo pipefail

DB_DIR="${DAILY_DB_DIR:-/mnt/hdd/daily/db}"
LOCK_FILE="${INGEST_LOCK_FILE:-${DB_DIR}/.ingest.lock}"
DB_FILE="${DAILY_DB_FILE:-${DB_DIR}/daily_llm_benchmark.duckdb}"
BUSY_EXIT_CODE=75
TEMP_DB=""

mkdir -p "$DB_DIR"

cleanup() {
  if [[ -n "$TEMP_DB" ]]; then
    rm -f -- "$TEMP_DB"
  fi
}
trap cleanup EXIT

# DuckDB takes an exclusive file lock for the read-write connection, so two
# concurrent refreshes would race; re-exec under flock and fail fast instead.
if [ "${INGEST_LOCK_HELD:-0}" != "1" ]; then
  export INGEST_LOCK_HELD=1
  exec flock -n -E "$BUSY_EXIT_CODE" "$LOCK_FILE" "$0" "$@"
fi

UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-/home/sungeunk/repo/run_daily/daily/requirements.txt}"
if [ ! -x "$UV_BIN" ]; then
  echo "uv not found at $UV_BIN" >&2
  exit 127
fi
if [ ! -r "$REQUIREMENTS_FILE" ]; then
  echo "requirements file not found at $REQUIREMENTS_FILE" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/home/sungeunk/repo/run_daily/daily"

TEMP_DB="$(mktemp --tmpdir="$DB_DIR" daily_llm_benchmark.XXXXXX.duckdb)"
if [[ -f "$DB_FILE" ]]; then
  cp --reflink=auto --sparse=always -- "$DB_FILE" "$TEMP_DB"
else
  rm -f -- "$TEMP_DB"
fi

"$UV_BIN" run \
  --with-requirements "$REQUIREMENTS_FILE" \
  python -m data.ingest.cli \
  --root /mnt/hdd/daily/data \
  --db "$TEMP_DB" \
  --force

mv -f -- "$TEMP_DB" "$DB_FILE"
TEMP_DB=""
