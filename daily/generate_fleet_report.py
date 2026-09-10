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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from common.delivery import send_mail
from common.mcp_client import McpError, call_json_tool
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
    day_start_hour: int
    purpose: str
    triggered_by: str
    expected_machines: tuple[str, ...]
    recipients: tuple[str, ...]
    subject_prefix: str
    relay_server: str | None
    max_wait_minutes: int
    poll_interval_seconds: int
    max_functional_issues: int
    top_regressions: int
    top_improvements: int
    output_dir: Path


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
    day_start_hour = int(root.get("day_start_hour", 6))
    if not 0 <= day_start_hour <= 23:
        raise ValueError("day_start_hour must be between 0 and 23")
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
        day_start_hour=day_start_hour,
        purpose=str(root["purpose"]),
        triggered_by=str(root["triggered_by"]),
        expected_machines=_strings(root.get("expected_machines"), "expected_machines"),
        recipients=_optional_strings(mail.get("recipients"), "mail.recipients"),
        subject_prefix=str(mail.get("subject_prefix", "Daily GPU")),
        relay_server=(str(mail["relay_server"]) if mail.get("relay_server") else None),
        max_wait_minutes=max(0, int(schedule.get("max_wait_minutes", 60))),
        poll_interval_seconds=max(1, int(schedule.get("poll_interval_seconds", 300))),
        max_functional_issues=max(0, int(report.get("max_functional_issues", 20))),
        top_regressions=max(0, int(report.get("top_regressions", 10))),
        top_improvements=max(0, int(report.get("top_improvements", 5))),
        output_dir=output_dir,
    )


def _logical_report_date(config: FleetConfig, now: dt.datetime) -> str:
    logical_now = now.astimezone(ZoneInfo(config.timezone)) - dt.timedelta(
        hours=config.day_start_hour
    )
    return logical_now.date().isoformat()


def _default_report_date(config: FleetConfig) -> str:
    now_local = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo(config.timezone))
    if "timer" in config.purpose.lower() and config.triggered_by.lower() == "scheduler":
        return (now_local.date() - dt.timedelta(days=1)).isoformat()
    return _logical_report_date(config, dt.datetime.now(dt.timezone.utc))


def _digest_arguments(config: FleetConfig, report_date: str) -> dict[str, object]:
    return {
        "report_date": report_date,
        "purpose": config.purpose,
        "triggered_by": config.triggered_by,
        "expected_machines": list(config.expected_machines),
        "day_start_hour": config.day_start_hour,
        "max_functional_issues": config.max_functional_issues,
        "top_regressions": config.top_regressions,
        "top_improvements": config.top_improvements,
    }


def fetch_digest(config: FleetConfig, report_date: str) -> dict[str, Any]:
    """Fetch and validate one digest payload from the MCP server."""
    payload = call_json_tool(
        config.mcp_url,
        "daily_results_daily_digest",
        _digest_arguments(config, report_date),
        timeout=30.0,
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise McpError("daily digest returned an unsupported payload")
    return payload


def wait_for_digest(config: FleetConfig, report_date: str,
                    *, wait: bool = True) -> dict[str, Any]:
    """Poll until the fleet is ready or the configured deadline expires."""
    deadline = time.monotonic() + config.max_wait_minutes * 60
    while True:
        digest = fetch_digest(config, report_date)
        summary = digest.get("summary")
        if isinstance(summary, dict) and summary.get("status") != "incomplete":
            return digest
        if not wait or time.monotonic() >= deadline:
            return digest
        time.sleep(min(config.poll_interval_seconds, max(0.0, deadline - time.monotonic())))


def _delivery_key(config: FleetConfig, report_date: str) -> str:
    identity = f"{report_date}\0{config.purpose}\0{config.triggered_by}"
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
    parser.add_argument("--date", help="Logical daily date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Write HTML without sending mail")
    parser.add_argument("--force", action="store_true", help="Send an already delivered cycle again")
    return parser.parse_args()


def main() -> int:
    """Generate one report and optionally deliver it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    try:
        config = load_config(args.config)
        report_date = args.date or _default_report_date(config)
        dt.date.fromisoformat(report_date)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        delivery_key = _delivery_key(config, report_date)
        state_path = config.output_dir / ".fleet_delivery_state.json"
        state = _state(state_path)
        if delivery_key in state and not args.force and not args.dry_run:
            log.info("report already sent for %s", report_date)
            return 0

        digest = wait_for_digest(config, report_date, wait=not args.dry_run)
        output = config.output_dir / f"daily-fleet.{report_date}.html"
        output.write_text(
            render_fleet_html(
                digest, config.viewer_base_url, config.html_report_base_url
            ), encoding="utf-8"
        )
        log.info("wrote %s", output)
        if args.dry_run:
            return 0

        if not config.recipients:
            raise ValueError("mail.recipients must not be empty when sending mail")

        summary = digest.get("summary") if isinstance(digest.get("summary"), dict) else {}
        status = str(summary.get("status") or "unknown").upper()
        sent = send_mail(
            output,
            ",".join(config.recipients),
            f"{config.subject_prefix} [{status}]",
            now_stamp=report_date,
            relay_server=config.relay_server,
        )
        if not sent:
            return 3
        state[delivery_key] = {
            "report_date": report_date,
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