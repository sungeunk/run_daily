"""Render a compact multi-machine daily digest as email-safe HTML."""

from __future__ import annotations

import datetime as dt
import html
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode, urlparse


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


def _html_report_url(row: Mapping[str, Any]) -> str:
    value = str(row.get("html_report_url") or "")
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _fallback_html_report_url(base_url: str, row: Mapping[str, Any]) -> str:
    parsed = urlparse(base_url)
    report_file = str(row.get("report_file") or "")
    machine = str(row.get("machine") or "")
    match = re.fullmatch(r"daily\.(\d{8}_\d{4})\.summary\.json", report_file)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or match is None or not machine:
        return ""
    stamp = match.group(1)
    return (
        f"{base_url.rstrip('/')}/daily2/{quote(machine, safe='')}/"
        f"{stamp[:4]}.{stamp[4:6]}/daily.{stamp}.html"
    )


def _details_link(viewer_base_url: str, row: Mapping[str, Any],
                  html_report_base_url: str) -> str:
    report_url = _html_report_url(row)
    if not report_url:
        report_url = _fallback_html_report_url(html_report_base_url, row)
    if report_url:
        url = html.escape(report_url, quote=True)
        return f'<a href="{url}" style="color:#075985">HTML report</a>'
    return ""


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
            f'<td class="num">{_text(_format_timestamp(row.get("ts")))}</td>'
            f'<td class="num">{_text(_format_duration(row.get("duration_sec")))}</td>'
            f'<td>{_text(row.get("ov_version") or "-")}</td>'
            f'<td class="num">{_text(series)}</td>'
            f'<td class="num">{regression_count}</td>'
            f'<td>{_details_link(viewer_base_url, row, html_report_base_url)}</td>'
            "</tr>"
        )
    return "".join(rows)


def _issue_rows(issues: Sequence[Mapping[str, Any]], viewer_base_url: str,
                html_report_base_url: str) -> str:
    def attention_reason(outcome: object) -> str:
        labels = {
            "failed": "Test assertion failed",
            "error": "Test execution error",
            "timeout": "Test timed out",
        }
        return labels.get(str(outcome or "").lower(), "Test requires investigation")

    rows = []
    for issue in issues:
        model = issue.get("model") or "-"
        precision = issue.get("precision") or "-"
        last_good_url = _html_report_url({"html_report_url": issue.get("last_good_html_report_url")})
        last_good_link = (
            f'<a href="{html.escape(last_good_url, quote=True)}" style="color:#075985">HTML report</a>'
            if issue.get("last_good_run_id") and last_good_url else ""
        )
        rows.append(
            "<tr>"
            f'<td>{_text(issue.get("machine"))}</td>'
            f'<td>{_text(f"{model} / {precision}")}</td>'
            f'<td>{_text(attention_reason(issue.get("outcome")))}</td>'
            f'<td>{last_good_link}</td>'
            "</tr>"
        )
    return "".join(rows)


def _regression_rows(rows: Sequence[Mapping[str, Any]], viewer_base_url: str,
                     html_report_base_url: str) -> str:
    def latency_mode(value: object) -> str:
        labels = {
            "1st": "First token latency",
            "2nd": "Second token latency",
            "1st-infer": "First token inference latency",
            "2nd-infer": "Second token inference latency",
        }
        return labels.get(str(value), str(value or "-"))

    def measurement(value: object, unit: object) -> str:
        try:
            return f"{float(value):.3f} {unit or ''}".strip()
        except (TypeError, ValueError):
            return "-"

    rendered = []
    for row in rows:
        improvement = row.get("improvement_pct")
        try:
            regression_pct = -float(improvement) * 100
            change = f"{regression_pct:.1f}% regression"
        except (TypeError, ValueError):
            regression_pct = None
            change = "-"
        unit = row.get("unit") or ""
        baseline = measurement(row.get("baseline_value"), unit)
        current = measurement(row.get("current_value"), unit)
        in_token = int(row.get("in_token") or 0)
        out_token = int(row.get("out_token") or 0)
        tokens = f"in: {in_token} / out: {out_token}"
        rendered.append(
            "<tr>"
            f'<td data-sort="{_text(row.get("machine"))}">{_text(row.get("machine"))}</td>'
            f'<td data-sort="{_text(row.get("model"))}">{_text(row.get("model"))}</td>'
            f'<td data-sort="{_text(row.get("precision"))}">{_text(row.get("precision"))}</td>'
            f'<td class="num" data-sort="{in_token * 1000000 + out_token}">{_text(tokens)}</td>'
            f'<td data-sort="{_text(row.get("exec_mode"))}">{_text(latency_mode(row.get("exec_mode")))}</td>'
            f'<td class="num" data-sort="{regression_pct if regression_pct is not None else ""}">{_text(change)}</td>'
            f'<td class="num" data-sort="{_text(row.get("baseline_value"))}">{_text(baseline)}</td>'
            f'<td class="num" data-sort="{_text(row.get("current_value"))}">{_text(current)}</td>'
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
    machine_by_name = {
        str(machine.get("machine")): machine
        for machine in machines if isinstance(machine, Mapping)
    }
    regressions = [
        {**machine_by_name.get(str(row.get("machine")), {}), **row}
        for row in regressions if isinstance(row, Mapping)
    ]
    status = str(summary.get("status") or "unknown")
    warning_html = "".join(f"<li>{_text(item)}</li>" for item in warnings)
    issue_section = ""
    if issues:
        issue_section = f"""
        <h2>Issues Requiring Attention</h2>
        <table><thead><tr><th>Machine</th><th>Models</th><th>Attention reason</th><th>Last good</th></tr></thead>
        <tbody>{_issue_rows(issues, viewer_base_url, html_report_base_url)}</tbody></table>
        """
    regression_section = ""
    if regressions:
        regression_section = f"""
        <h2>Performance Regressions</h2>
        <table id="performance-regressions"><thead><tr><th data-sort-type="text">Machine</th><th data-sort-type="text">Model</th><th data-sort-type="text">Precision</th><th data-sort-type="number">Tokens</th><th data-sort-type="text">Mode</th><th data-sort-type="number">Regression</th><th data-sort-type="number">Baseline</th><th data-sort-type="number">Current</th><th>Report</th></tr></thead>
        <tbody>{_regression_rows(regressions, viewer_base_url, html_report_base_url)}</tbody></table>
        """

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body{{font-family:Segoe UI,Arial,sans-serif;color:#202124;margin:24px;line-height:1.4}}
h1{{font-size:22px;margin:0 0 8px}}h2{{font-size:17px;margin:24px 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #d1d5db;padding:7px;text-align:left;vertical-align:top}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
th{{background:#f3f4f6}}th[data-sort-type]{{cursor:pointer;text-decoration:underline}}
.summary{{margin:10px 0 18px;color:#4b5563}}
</style></head><body>
<h1>Daily GPU Fleet Summary {_status_dot(status)}</h1>
<div class="summary">Build: <strong>{_text(selection.get("ov_build"))}</strong> ·
Completed: <strong>{_text(summary.get("completed_machines"))}/{_text(summary.get("expected_machines"))}</strong> ·
Failed machines: <strong>{_text(summary.get("failed_machines"))}</strong> · 
Purpose: {_text(selection.get("purpose"))}</div>
<h2>Machine Summary</h2>
<table><thead><tr><th>Machine</th><th>Status</th><th>Execution time</th><th>Duration</th><th>OpenVINO</th><th>Series</th><th>Regression</th><th>HTML report</th></tr></thead>
<tbody>{_machine_rows(machines, viewer_base_url, html_report_base_url)}</tbody></table>
{issue_section}
{regression_section}
{f'<h2>Warnings</h2><ul>{warning_html}</ul>' if warning_html else ''}
<script>
function sortFleetTable(header) {{
    const table = header.closest('table');
    const index = Array.prototype.indexOf.call(header.parentNode.children, header);
    const ascending = header.getAttribute('aria-sort') !== 'ascending';
    const type = header.dataset.sortType;
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((left, right) => {{
        const a = left.cells[index].dataset.sort || '';
        const b = right.cells[index].dataset.sort || '';
        if (!a) return 1;
        if (!b) return -1;
        const comparison = type === 'number' ? Number(a) - Number(b) : a.localeCompare(b);
        return ascending ? comparison : -comparison;
    }});
    rows.forEach((row) => table.tBodies[0].appendChild(row));
    Array.from(header.parentNode.children).forEach((cell) => cell.removeAttribute('aria-sort'));
    header.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
}}
document.querySelectorAll('th[data-sort-type]').forEach((header) => {{
    header.tabIndex = 0;
    header.setAttribute('role', 'button');
    header.setAttribute('aria-sort', 'none');
    header.addEventListener('click', () => sortFleetTable(header));
    header.addEventListener('keydown', (event) => {{
        if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); sortFleetTable(header); }}
    }});
}});
</script>
</body></html>"""