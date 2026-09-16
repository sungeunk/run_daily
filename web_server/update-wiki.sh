#!/usr/bin/env bash
#
# Build the MkDocs site with uv and reload the user-level Caddy service.

set -Eeuo pipefail
trap 'printf "Error at line %d\n" "$LINENO" >&2' ERR

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WIKI_DIR="${SCRIPT_DIR}/wiki"
readonly REQUIREMENTS_FILE="${WIKI_DIR}/requirements.txt"
readonly CADDY_SERVICE="caddy.service"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build the Wiki with Python 3.12 through uv and reload Caddy after a
successful build.

Options:
  --no-reload  Build the site without reloading Caddy.
  -h, --help   Show this help message.

Exit codes:
  0  Build and optional reload succeeded.
  1  Invalid arguments or a command failed.
EOF
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
    elif [[ -x /usr/local/bin/uv ]]; then
        printf '%s\n' /usr/local/bin/uv
    else
        printf 'uv was not found in PATH or /usr/local/bin/uv\n' >&2
        exit 1
    fi
}

main() {
    local reload_caddy=true
    local uv_command

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-reload)
                reload_caddy=false
                shift
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                printf 'Unknown option: %s\n\n' "$1" >&2
                usage >&2
                return 1
                ;;
        esac
    done

    [[ -r "$REQUIREMENTS_FILE" ]] || {
        printf 'Missing requirements file: %s\n' "$REQUIREMENTS_FILE" >&2
        return 1
    }

    uv_command="$(find_uv)"

    printf 'Building Wiki: %s\n' "$WIKI_DIR"
    (
        cd "$WIKI_DIR"
        "$uv_command" run \
            --python 3.12 \
            --with-requirements "$REQUIREMENTS_FILE" \
            mkdocs build --strict
    )

    if [[ "$reload_caddy" == true ]]; then
        printf 'Reloading user service: %s\n' "$CADDY_SERVICE"
        systemctl --user reload "$CADDY_SERVICE"
    else
        printf 'Skipping Caddy reload\n'
    fi

    printf 'Wiki update completed\n'
}

main "$@"
