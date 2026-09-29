#!/usr/bin/env bash
# Generate a distribution-aware HTML analysis report from the daily_results MCP server.
#
# Usage:
#   ./generate_html_report.sh [--machine <name>] [extra args passed to generate_analysis_report.py]
#
# Examples:
#   ./generate_html_report.sh
#   ./generate_html_report.sh --machine dg2alderlake
#   ./generate_html_report.sh --machine MTL-01
#   ./generate_html_report.sh --stamp 20260530_0315
#   ./generate_html_report.sh --run-id daily.20260530_0315.report
#   ./generate_html_report.sh --history-window 15 --fluctuation-scale 2.0
#
# Environment overrides:
#   UV_BIN         uv executable (default: /usr/local/bin/uv)
#   MCP_URL        daily_results MCP endpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${REPO_ROOT}/scripts/generate_analysis_report.py"

usage() {
    cat <<EOF
Usage: ./generate_html_report.sh [OPTIONS] [EXTRA_ARGS...]

Options:
    -m, --machine NAME   Machine name used to select the latest run
            --mcp-url URL    daily_results MCP endpoint
  -h, --help           Show this help message

Any remaining arguments are passed through to generate_analysis_report.py.
EOF
}

UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
MACHINE="${MACHINE_NAME:-}"
MCP_URL="${MCP_URL:-http://dg2fizz.ikor.intel.com:8090/mcp}"

[[ -x "$UV_BIN" ]] || { echo "uv not found at $UV_BIN" >&2; exit 127; }
[[ -r "$REQUIREMENTS_FILE" ]] || { echo "requirements not found at $REQUIREMENTS_FILE" >&2; exit 1; }

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
        --mcp-url)
            if [[ -z "${2:-}" ]]; then
                echo "[report] --mcp-url requires a value" >&2
                exit 2
            fi
            MCP_URL="$2"
            shift 2
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

EXTRA_ARGS=("$@")
PYTHON_ARGS=(--mcp-url "$MCP_URL")
if [[ -n "$MACHINE" ]]; then
    PYTHON_ARGS+=(--machine "$MACHINE")
fi

PYTHONPATH="${SCRIPT_DIR}" \
    "$UV_BIN" run --with-requirements "$REQUIREMENTS_FILE" python "$PYTHON_SCRIPT" \
    "${PYTHON_ARGS[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
