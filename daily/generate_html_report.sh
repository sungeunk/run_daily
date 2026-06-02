#!/usr/bin/env bash
# Generate a distribution-aware HTML analysis report from daily run results.
#
# Usage:
#   ./generate_html_report.sh [--machine <name>] [--rebuild-db] [extra args passed to generate_analysis_report.py]
#
# Examples:
#   ./generate_html_report.sh
#   ./generate_html_report.sh --machine dg2alderlake
#   ./generate_html_report.sh --machine MTL-01 --rebuild-db
#   ./generate_html_report.sh --stamp 20260530_0315
#   ./generate_html_report.sh --run-id daily.20260530_0315.report
#   ./generate_html_report.sh --history-window 15 --fluctuation-scale 2.0
#
# Environment overrides:
#   CONDA_ENV      conda environment name (default: skills)
#   DB_PATH        DuckDB path            (default: <repo>/daily/viewer/bench.<machine>.duckdb)
#   DAILY_ROOT     ingest root            (default: /var/www/html/daily/<machine>)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${REPO_ROOT}/scripts/generate_analysis_report.py"

usage() {
    cat <<EOF
Usage: ./generate_html_report.sh [OPTIONS] [EXTRA_ARGS...]

Options:
  -m, --machine NAME   Machine name used for DB path and ingest root
      --rebuild-db     Remove the machine-specific DuckDB before regenerating report
  -h, --help           Show this help message

Any remaining arguments are passed through to generate_analysis_report.py.
EOF
}

CONDA_ENV="${CONDA_ENV:-skills}"
MACHINE=""
REBUILD_DB=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine|-m)
            if [[ -z "${2:-}" ]]; then
                echo "[report] $1 requires a value" >&2
                exit 2
            fi
            MACHINE="$2"
            shift 2
            ;;
        --rebuild-db)
            REBUILD_DB=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

if [[ -z "$MACHINE" ]]; then
    MACHINE="${MACHINE_NAME:-dg2alderlake}"
fi

DB_PATH="${DB_PATH:-${SCRIPT_DIR}/viewer/bench.${MACHINE}.duckdb}"
DAILY_ROOT="${DAILY_ROOT:-/var/www/html/daily/${MACHINE}}"

EXTRA_ARGS=("$@")

if [[ ! -d "$DAILY_ROOT" ]]; then
    echo "[report] ingest root not found: $DAILY_ROOT" >&2
    exit 1
fi

if [[ "$REBUILD_DB" -eq 1 ]]; then
    echo "[report] removing existing DB: $DB_PATH"
    rm -f "$DB_PATH" "$DB_PATH.wal" "$DB_PATH.tmp"
fi

PYTHONPATH="${SCRIPT_DIR}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python "${PYTHON_SCRIPT}" \
    --db "${DB_PATH}" \
    --root "${DAILY_ROOT}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
