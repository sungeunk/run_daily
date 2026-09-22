#!/usr/bin/env python3
"""Generate and optionally mail one MCP-backed fleet daily report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from common.delivery import send_mail
from common.mcp_client import McpError, McpHttpClient, call_json_tool
from report.fleet import render_fleet_html


DAILY_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DAILY_DIR / "fleet_report.json"
log = logging.getLogger("fleet-report")


@dataclass(frozen=True, slots=True)
class FleetConfig:
    mcp_url: str
    viewer_base_url: str
    html_report_base_url: str
    timezone: str
    purpose: str
    triggered_by: str
    ov_sha: str | None
    expected_machines: tuple[str, ...]
    recipients: tuple[str, ...]
    subject_prefix: str
    relay_server: str | None
    max_wait_minutes: int
    poll_interval_seconds: int
    max_functional_issues: int
    top_regressions: int | None
    top_improvements: int
    output_dir: Path
    report_title: str
    output_prefix: str


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must be a non-empty string array")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _optional_strings(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must be a string array")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "fleet"


def _parse_ov_version(value: str) -> tuple[str, str]:
    """Extract build and commit SHA from an OpenVINO version string.

    PR/custom suffixes after the SHA are intentionally ignored.
    """
    match = re.fullmatch(r"\d+\.\d+\.\d+-(\d+)-([A-Za-z0-9]+)(?:-.+)?", value.strip())
    if match is None:
        raise ValueError("--ov-ver must look like 2026.5.0-23164-749d332ac8b")
    return match.group(1), match.group(2)


def load_config(path: Path) -> FleetConfig:
    """Read and validate the fleet report JSON configuration."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    root = _mapping(payload, "config")
    mail = _mapping(root.get("mail", {}), "mail")
    schedule = _mapping(root.get("schedule", {}), "schedule")
    report = _mapping(root.get("report", {}), "report")
    timezone = str(root.get("timezone", "Asia/Seoul"))
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {timezone}") from exc
    output_dir = Path(str(root.get("output_dir", "../output/fleet")))
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()
    return FleetConfig(
        mcp_url=str(root["mcp_url"]),
        viewer_base_url=str(root["viewer_base_url"]),
        html_report_base_url=str(root.get("html_report_base_url", "")),
        timezone=timezone,
        purpose=str(root["purpose"]),
        triggered_by=str(root["triggered_by"]),
        ov_sha=(str(root["ov_sha"]).strip() if root.get("ov_sha") else None),
        expected_machines=_strings(root.get("expected_machines"), "expected_machines"),
        recipients=_optional_strings(mail.get("recipients"), "mail.recipients"),
        subject_prefix=str(mail.get("subject_prefix", "Daily GPU")),
        relay_server=(str(mail["relay_server"]) if mail.get("relay_server") else None),
        max_wait_minutes=max(0, int(schedule.get("max_wait_minutes", 60))),
        poll_interval_seconds=max(1, int(schedule.get("poll_interval_seconds", 300))),
        max_functional_issues=max(0, int(report.get("max_functional_issues", 20))),
        top_regressions=(
            None if report.get("top_regressions", 10) is None
            else max(0, int(report.get("top_regressions", 10)))
        ),
        top_improvements=max(0, int(report.get("top_improvements", 5))),
        output_dir=output_dir,
        report_title=str(root.get("report_title", "Daily GPU Fleet Summary")),
        output_prefix=str(root.get("output_prefix", "daily-fleet")),
    )


def _apply_cli_overrides(config: FleetConfig, args: argparse.Namespace) -> FleetConfig:
    updates: dict[str, object] = {}
    if purpose := getattr(args, "purpose", None):
        updates["purpose"] = purpose
    if triggered_by := getattr(args, "triggered_by", None):
        updates["triggered_by"] = triggered_by
    if ov_sha := getattr(args, "ov_sha", None):
        updates["ov_sha"] = ov_sha
    if machines := getattr(args, "machines", None):
        updates["expected_machines"] = tuple(dict.fromkeys(machines))
    if exclude_machine := getattr(args, "exclude_machine", None):
        excluded = set(exclude_machine)
        updates["expected_machines"] = tuple(
            machine for machine in updates.get("expected_machines", config.expected_machines)
            if machine not in excluded
        )
    if title := getattr(args, "title", None):
        updates["report_title"] = title
    if output_prefix := getattr(args, "output_prefix", None):
        updates["output_prefix"] = output_prefix
    if not updates:
        return config
    updated = replace(config, **updates)
    if not updated.expected_machines:
        raise ValueError("expected_machines must not be empty after CLI overrides")
    return updated


def _call_json_tool(config: FleetConfig, name: str, arguments: dict[str, object], *, client=None):
    if client is not None:
        return client.call_json_tool(name, arguments)
    return call_json_tool(config.mcp_url, name, arguments, timeout=30.0)


def latest_build(config: FleetConfig, *, client=None) -> str:
    """Newest build the scheduled cycle has produced a run for.

    The fleet is keyed by build rather than by date because machines start
    their nightly run at their own local times and the batch straddles
    midnight. The newest build may still be in flight; `wait_for_digest`
    is what waits for the remaining machines.
    """
    builds = _call_json_tool(
        config,
        "daily_results_list_builds",
        {"purpose": config.purpose,
         "triggered_by": config.triggered_by,
         "limit": 1},
        client=client,
    )
    if not isinstance(builds, list) or not builds:
        raise McpError(
            f"no build found for purpose={config.purpose!r} "
            f"triggered_by={config.triggered_by!r}"
        )
    build = str(builds[0].get("ov_build") or "").strip()
    if not build:
        raise McpError("build listing returned a row without ov_build")
    return build


def _digest_arguments(config: FleetConfig, ov_build: str) -> dict[str, object]:
    arguments: dict[str, object] = {
        "ov_build": ov_build,
        "purpose": config.purpose,
        "triggered_by": config.triggered_by,
        "expected_machines": list(config.expected_machines),
        "max_functional_issues": config.max_functional_issues,
        "top_regressions": 500 if config.top_regressions is None else config.top_regressions,
        "top_improvements": config.top_improvements,
        "html_report_base_url": config.html_report_base_url,
    }
    if config.ov_sha:
        arguments["ov_sha"] = config.ov_sha
    return arguments


def fetch_digest(config: FleetConfig, ov_build: str, *, client=None) -> dict[str, Any]:
    """Fetch and validate one digest payload from the MCP server."""
    payload = _call_json_tool(
        config,
        "daily_results_daily_digest",
        _digest_arguments(config, ov_build),
        client=client,
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise McpError("daily digest returned an unsupported payload")
    return payload


def wait_for_digest(config: FleetConfig, ov_build: str,
                    *, wait: bool = True, client=None) -> dict[str, Any]:
    """Poll until the fleet is ready or the configured deadline expires."""
    deadline = time.monotonic() + config.max_wait_minutes * 60
    while True:
        digest = fetch_digest(config, ov_build, client=client)
        summary = digest.get("summary")
        if isinstance(summary, dict) and summary.get("status") != "incomplete":
            return digest
        if not wait or time.monotonic() >= deadline:
            return digest
        time.sleep(min(config.poll_interval_seconds, max(0.0, deadline - time.monotonic())))


def _delivery_key(config: FleetConfig, ov_build: str, digest: dict[str, Any]) -> str:
    """Identity of what is about to be delivered.

    Keyed on the selected runs, not on the build alone: a build that stays
    current for a second night is re-tested, and that genuinely is a new
    report. Keying on the build would silently swallow it.
    """
    run_ids = sorted(
        str(m.get("run_id"))
        for m in digest.get("machines", [])
        if isinstance(m, dict) and m.get("run_id")
    )
    identity = "\0".join(
        [ov_build, config.purpose, config.triggered_by, *run_ids]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(os.environ.get("DAILY_FLEET_CONFIG", DEFAULT_CONFIG)),
    )
    parser.add_argument(
        "--ov-ver",
        help="OpenVINO version, e.g. 2026.5.0-23164-749d332ac8b; trailing PR suffix is ignored",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write HTML without sending mail")
    parser.add_argument("--force", action="store_true", help="Send an already delivered cycle again")
    parser.add_argument("--purpose", help="Override config purpose, e.g. a PR/custom run label")
    parser.add_argument("--triggered-by", help="Override config trigger identity")
    parser.add_argument(
        "--machine", dest="machines", action="append",
        help="Expected machine for this report; repeat to override config list",
    )
    parser.add_argument(
        "--exclude-machine", action="append", default=[],
        help="Remove a machine from the configured or overridden expected list",
    )
    parser.add_argument("--title", help="HTML report title override")
    parser.add_argument("--output-prefix", help="Output file prefix override")
    parser.add_argument(
        "--wait", action="store_true",
        help="Wait for missing machines even with --dry-run",
    )
    return parser.parse_args()


def main() -> int:
    """Generate one report and optionally deliver it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    try:
        config = _apply_cli_overrides(load_config(args.config), args)
        ov_build = None
        if ov_ver := getattr(args, "ov_ver", None):
            ov_build, ov_sha = _parse_ov_version(ov_ver)
            config = replace(config, ov_sha=ov_sha)
        with McpHttpClient(config.mcp_url, timeout=30.0) as client:
            ov_build = ov_build or latest_build(config, client=client)
            # The key depends on which runs were selected, so it can only be
            # computed once the digest is in hand.
            digest = wait_for_digest(
                config, ov_build,
                wait=(getattr(args, "wait", False) or not args.dry_run),
                client=client,
            )
        config.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = config.output_dir / ".fleet_delivery_state.json"
        state = _state(state_path)
        delivery_key = _delivery_key(config, ov_build, digest)
        sha_suffix = f".{_slug(config.ov_sha)}" if config.ov_sha else ""
        output = config.output_dir / f"{_slug(config.output_prefix)}.{_slug(config.purpose)}.{ov_build}{sha_suffix}.html"
        output.write_text(
            render_fleet_html(
                digest, config.viewer_base_url, config.html_report_base_url,
                title=config.report_title,
            ), encoding="utf-8"
        )
        log.info("wrote %s", output)
        if args.dry_run:
            return 0

        if delivery_key in state and not args.force:
            log.info("report already sent for build %s (same runs)", ov_build)
            return 0

        if not config.recipients:
            raise ValueError("mail.recipients must not be empty when sending mail")

        summary = digest.get("summary") if isinstance(digest.get("summary"), dict) else {}
        status = str(summary.get("status") or "unknown").upper()
        sent = send_mail(
            output,
            ",".join(config.recipients),
            f"{config.subject_prefix} [{status}] {config.purpose} build {ov_build}",
            now_stamp=ov_build,
            relay_server=config.relay_server,
        )
        if not sent:
            return 3
        state[delivery_key] = {
            "ov_build": ov_build,
            "ov_sha": config.ov_sha,
            "purpose": config.purpose,
            "triggered_by": config.triggered_by,
            "report": str(output),
            "sent_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _write_state(state_path, state)
        return 4 if status == "INCOMPLETE" else 0
    except (KeyError, TypeError, ValueError, McpError) as exc:
        log.error("fleet report failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())