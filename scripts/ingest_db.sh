#!/usr/bin/env bash
# Source of truth for the deployed copy at /var/www/html/daily2/ingest_db.sh,
# which is what the viewer's "Refresh database" button executes.

set -Eeuo pipefail

LOCK_FILE="${INGEST_LOCK_FILE:-/var/www/html/daily2/.ingest.lock}"
BUSY_EXIT_CODE=75

# DuckDB takes an exclusive file lock for the read-write connection, so two
# concurrent refreshes would race; re-exec under flock and fail fast instead.
if [ "${INGEST_LOCK_HELD:-0}" != "1" ]; then
  export INGEST_LOCK_HELD=1
  exec flock -n -E "$BUSY_EXIT_CODE" "$LOCK_FILE" "$0" "$@"
fi

CONDA_BIN="${CONDA_BIN:-/home/sungeunk/miniforge3/bin/conda}"
if [ ! -x "$CONDA_BIN" ]; then
  echo "conda not found at $CONDA_BIN" >&2
  exit 127
fi

# Load conda in a non-login shell so subprocesses invoked from Streamlit can
# find the environment hooks reliably.
source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"
conda activate daily.py312

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/home/sungeunk/repo/run_daily/daily"

python -m viewer.ingest.cli --root /var/www/html/daily2 --db /var/www/html/daily2/daily_llm_benchmark.duckdb --force
