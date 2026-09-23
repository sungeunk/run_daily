#!/usr/bin/env bash
# Register, operate, and inspect the user-level web services.

set -Eeuo pipefail
trap 'printf "Error at line %d\n" "$LINENO" >&2' ERR

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WIKI_DIR="${SCRIPT_DIR}/wiki"
readonly WIKI_REQUIREMENTS="${WIKI_DIR}/requirements.txt"
readonly USER_UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
readonly CADDY_UNIT="${SCRIPT_DIR}/caddy/systemd/caddy.service"
readonly VIEWER_UNIT="${SCRIPT_DIR}/caddy/systemd/daily-viewer.service"
readonly JENKINS_UNIT="${SCRIPT_DIR}/caddy/systemd/jenkins-agent.service"
readonly SERVICES=(caddy.service daily-viewer.service jenkins-agent.service)

usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND

Commands:
  install       Register available user service units and reload systemd.
  start         Start all registered services.
  restart       Restart all registered services.
  status        Show the status of all registered services.
  build-wiki    Build the Wiki and reload Caddy after success.
    all           Install services, start them, and build the Wiki.
  -h, --help    Show this help message.

Services:
  caddy.service
  daily-viewer.service
  jenkins-agent.service (optional; unit is not stored in this repository)
EOF
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
    elif [[ -x /usr/local/bin/uv ]]; then
        printf '%s\n' /usr/local/bin/uv
    else
        printf 'uv was not found in PATH or /usr/local/bin/uv\n' >&2
        return 1
    fi
}

unit_source() {
    case "$1" in
        caddy.service) printf '%s\n' "$CADDY_UNIT" ;;
        daily-viewer.service) printf '%s\n' "$VIEWER_UNIT" ;;
        jenkins-agent.service) printf '%s\n' "$JENKINS_UNIT" ;;
        *) return 1 ;;
    esac
}

install_services() {
    local service source target
    mkdir -p "$USER_UNIT_DIR"
    for service in "${SERVICES[@]}"; do
        source="$(unit_source "$service")"
        target="${USER_UNIT_DIR}/${service}"
        if [[ ! -f "$source" ]]; then
            printf 'Skipping %s: unit file is not present in web_server\n' "$service"
            continue
        fi
        ln -sfn "$source" "$target"
        printf 'Registered %s\n' "$service"
    done
    systemctl --user daemon-reload
}

start_services() {
    local service
    for service in "${SERVICES[@]}"; do
        if systemctl --user cat "$service" >/dev/null 2>&1; then
            systemctl --user start "$service"
            printf 'Started %s\n' "$service"
        else
            printf 'Skipping %s: service is not registered\n' "$service"
        fi
    done
}

restart_services() {
    local service
    for service in "${SERVICES[@]}"; do
        if systemctl --user cat "$service" >/dev/null 2>&1; then
            systemctl --user restart "$service"
            printf 'Restarted %s\n' "$service"
        else
            printf 'Skipping %s: service is not registered\n' "$service"
        fi
    done
}

status_services() {
    local service
    for service in "${SERVICES[@]}"; do
        printf '\n--- %s ---\n' "$service"
        if systemctl --user cat "$service" >/dev/null 2>&1; then
            systemctl --user --no-pager --full status "$service" || true
        else
            printf 'Not registered\n'
        fi
    done
}

build_wiki() {
    local uv_command
    [[ -r "$WIKI_REQUIREMENTS" ]] || {
        printf 'Missing requirements file: %s\n' "$WIKI_REQUIREMENTS" >&2
        return 1
    }
    uv_command="$(find_uv)"
    (
        cd "$WIKI_DIR"
        "$uv_command" run --python 3.12 \
            --with-requirements "$WIKI_REQUIREMENTS" \
            mkdocs build --strict
    )
    systemctl --user reload caddy.service
    printf 'Wiki build completed and Caddy reloaded\n'
}

main() {
    local command="${1:-}"
    case "$command" in
        install) install_services ;;
        start) start_services ;;
        restart) restart_services ;;
        status) status_services ;;
        build-wiki) build_wiki ;;
        all)
            install_services
            start_services
            build_wiki
            ;;
        -h|--help) usage ;;
        *) usage >&2; return 1 ;;
    esac
}

main "$@"
