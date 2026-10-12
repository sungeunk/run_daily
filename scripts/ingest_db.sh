#!/usr/bin/env bash
# Refresh the central DuckDB from daily result artifacts. The viewer executes
# this tracked script directly.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

DB_DIR="${DAILY_DB_DIR:-/mnt/hdd/daily/db}"
DB_FILE="${DAILY_DB_FILE:-${DB_DIR}/daily_llm_benchmark.duckdb}"
DATA_ROOT="${DAILY_DATA_ROOT:-/mnt/hdd/daily/data}"

UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${REPO_ROOT}/daily/requirements.txt}"
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found at $UV_BIN" >&2
  exit 127
fi
if [[ ! -r "$REQUIREMENTS_FILE" ]]; then
  echo "requirements file not found at $REQUIREMENTS_FILE" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}/daily"

exec "$UV_BIN" run \
  --with-requirements "$REQUIREMENTS_FILE" \
  python -u -m data.ingest.refresh \
  --root "$DATA_ROOT" \
  --db "$DB_FILE" \
  "$@"
