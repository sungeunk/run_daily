"""Render a compact multi-machine daily digest as email-safe HTML."""

from __future__ import annotations

import datetime as dt
import html
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode


def _text(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _status_dot(status: object) -> str:
    value = str(status or "unknown").lower()
    icon = {
        "green": "🟢",
        "success": "🟢",
        "yellow": "🟡",
        "red": "🔴",
        "failed": "🔴",
        "incomplete": "🟡",
        "missing": "⚪",
    }.get(value, "⚪")
    return (
        f'<span title="{_text(value)}" aria-label="{_text(value)}">{icon}</span>'
    )


def _viewer_url(base_url: str, run_id: object) -> str:
    if "?" in base_url:
        return f"{base_url}&{urlencode({'run_id': str(run_id)})}"
    return f"{base_url.rstrip('/')}/?{urlencode({'run_id': str(run_id)})}"


def _html_report_url(base_url: str, row: Mapping[str, Any]) -> str:
    report_file = str(row.get("report_file") or "")
    machine = str(row.get("machine") or "")
    timestamp = str(row.get("ts") or "")
    if not base_url or not report_file or not machine or not timestamp:
        return ""
    if report_file.endswith(".summary.json"):
        report_file = report_file.removesuffix(".summary.json") + ".html"
    month = timestamp[:7].replace("-", ".")
    return (
        f"{base_url.rstrip('/')}/daily2/{quote(machine)}/{quote(month)}/"
        f"{quote(report_file)}"
    )


def _details_link(viewer_base_url: str, row: Mapping[str, Any],
                  html_report_base_url: str) -> str:
    report_url = _html_report_url(html_report_base_url, row)
    if report_url:
        url = html.escape(report_url, quote=True)
        return f'<a href="{url}" style="color:#075985">HTML report</a>'
    run_id = row.get("run_id")
    if not viewer_base_url or not run_id:
        return ""
    url = html.escape(_viewer_url(viewer_base_url, run_id), quote=True)
    return f'<a href="{url}" style="color:#075985">HTML report</a>'


def _raw_log_link(base_url: str, row: Mapping[str, Any]) -> str:
    rawlog_path = str(row.get("rawlog_path") or "")
    prefix = "/var/www/html/"
    if not base_url or not rawlog_path.startswith(prefix):
        return ""
    relative = quote(rawlog_path.removeprefix(prefix), safe="/")
    url = html.escape(f"{base_url.rstrip('/')}/{relative}", quote=True)
    return f'<a href="{url}" style="color:#075985">Raw log</a>'


def _format_timestamp(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(value)[:16].replace("T", " ")


def _format_duration(value: object) -> str:
    try:
        seconds = max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return "-"
    return f"{seconds // 60}m"


def _machine_rows(machines: Sequence[Mapping[str, Any]], viewer_base_url: str,
                  html_report_base_url: str) -> str:
    rows = []
    for row in machines:
        status = str(row.get("status") or "unknown")
        series = "-" if status == "missing" else (
            f'Total: {int(row.get("series_total") or 0)} / '
            f'Skip: {int(row.get("series_skipped") or 0)} / '
            f'Success: {int(row.get("series_success") or 0)} / '
            f'Failed: {int(row.get("series_failed") or 0)}'
        )
        performance = row.get("performance")
        regression_count = (
            int(performance.get("regressed", 0))
            if isinstance(performance, Mapping) else 0
        )
        rows.append(
            "<tr>"
            f'<td>{_text(row.get("machine"))}</td>'
            f'<td>{_status_dot(status)}</td>'
            f'<td>{_text(_format_timestamp(row.get("ts")))}</td>'
            f'<td>{_text(_format_duration(row.get("duration_sec")))}</td>'
            f'<td>{_text(row.get("ov_version") or "-")}</td>'
            f'<td>{_text(series)}</td>'
            f'<td>{regression_count}</td>'
            f'<td>{_details_link(viewer_base_url, row, html_report_base_url)}</td>'
            "</tr>"
        )
    return "".join(rows)


def _issue_rows(issues: Sequence[Mapping[str, Any]], viewer_base_url: str,
                html_report_base_url: str) -> str:
    rows = []
    for issue in issues:
        rows.append(
            "<tr>"
            f'<td>{_text(issue.get("machine"))}</td>'
            f'<td>{_text(issue.get("failed_series") or 0)}</td>'
            f'<td>{_text(issue.get("model") or "-")}</td>'
            f'<td>{_text(issue.get("precision") or "-")}</td>'
            f'<td>{_raw_log_link(html_report_base_url, issue)}</td>'
            "</tr>"
        )
    return "".join(rows)


def _regression_rows(rows: Sequence[Mapping[str, Any]], viewer_base_url: str,
                     html_report_base_url: str) -> str:
    rendered = []
    for row in rows:
        improvement = row.get("improvement_pct")
        change = f"{float(improvement) * 100:+.1f}%" if improvement is not None else "-"
        shape = f'{int(row.get("in_token") or 0)} → {int(row.get("out_token") or 0)}'
        rendered.append(
            "<tr>"
            f'<td>{_text(row.get("machine"))}</td>'
            f'<td>{_text(row.get("model"))}</td>'
            f'<td>{_text(row.get("precision"))}</td>'
            f'<td>{_text(shape)}</td>'
            f'<td>{_text(row.get("exec_mode"))}</td>'
            f'<td>{_text(change)}</td>'
            f'<td>{_details_link(viewer_base_url, row, html_report_base_url)}</td>'
            "</tr>"
        )
    return "".join(rendered)


def render_fleet_html(digest: Mapping[str, Any], viewer_base_url: str,
                      html_report_base_url: str = "") -> str:
    """Return a compact HTML email for one fleet digest."""
    summary = digest.get("summary") if isinstance(digest.get("summary"), Mapping) else {}
    selection = digest.get("selection") if isinstance(digest.get("selection"), Mapping) else {}
    machines = digest.get("machines") if isinstance(digest.get("machines"), list) else []
    issues = (
        digest.get("functional_issues")
        if isinstance(digest.get("functional_issues"), list) else []
    )
    warnings = digest.get("warnings") if isinstance(digest.get("warnings"), list) else []
    regressions = (
        digest.get("top_regressions")
        if isinstance(digest.get("top_regressions"), list) else []
    )
    status = str(summary.get("status") or "unknown")
    warning_html = "".join(f"<li>{_text(item)}</li>" for item in warnings)
    issue_section = ""
    if issues:
        issue_section = f"""
        <h2>Issues Requiring Attention</h2>
        <table><thead><tr><th>Machine</th><th>Failed series</th><th>Model</th><th>Precision</th><th>Raw log</th></tr></thead>
        <tbody>{_issue_rows(issues, viewer_base_url, html_report_base_url)}</tbody></table>
        """
    regression_section = ""
    if regressions:
        regression_section = f"""
        <h2>Top Performance Regressions</h2>
        <table><thead><tr><th>Machine</th><th>Model</th><th>Precision</th><th>Tokens</th><th>Mode</th><th>Change</th><th>Report</th></tr></thead>
        <tbody>{_regression_rows(regressions, viewer_base_url, html_report_base_url)}</tbody></table>
        """

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body{{font-family:Segoe UI,Arial,sans-serif;color:#202124;margin:24px;line-height:1.4}}
h1{{font-size:22px;margin:0 0 8px}}h2{{font-size:17px;margin:24px 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #d1d5db;padding:7px;text-align:left;vertical-align:top}}
th{{background:#f3f4f6}}.summary{{margin:10px 0 18px;color:#4b5563}}
</style></head><body>
<h1>Daily GPU Fleet Summary {_status_dot(status)}</h1>
<div class="summary">Report date: <strong>{_text(selection.get("report_date"))}</strong> · 
Completed: <strong>{_text(summary.get("completed_machines"))}/{_text(summary.get("expected_machines"))}</strong> · 
Failed machines: <strong>{_text(summary.get("failed_machines"))}</strong> · 
Purpose: {_text(selection.get("purpose"))}</div>
<h2>Machine Summary</h2>
<table><thead><tr><th>Machine</th><th>Status</th><th>Execution time</th><th>Duration</th><th>OpenVINO</th><th>Series</th><th>Regression</th><th>HTML report</th></tr></thead>
<tbody>{_machine_rows(machines, viewer_base_url, html_report_base_url)}</tbody></table>
{issue_section}
{regression_section}
{f'<h2>Warnings</h2><ul>{warning_html}</ul>' if warning_html else ''}
</body></html>"""