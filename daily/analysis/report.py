"""Text report renderer for AnalysisResult.

Produces the ``[ Analysis summary ]`` block that is prepended to the
daily text report.  All formatting lives here so the engine and
persistence layer stay format-agnostic.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

from data import expected_cases

from .types import AnalysisResult


def _gpu_memory_text(summary: dict | None) -> tuple[str | None, str]:
    """Return ``(dedicated, shared)`` display strings from the run metadata.

    ``dedicated`` is ``None`` on an iGPU, where the DXGI figure is only the
    BIOS carve-out and not a meaningful amount of usable memory.
    """
    meta = (summary or {}).get("meta") or {}

    def _mb(value) -> str:
        if value in (None, ""):
            return "—"
        return f"{float(value) / 1024:.2f} GB ({float(value):,.0f} MB)"

    shared = _mb(meta.get("gpu_shared_memory_mb"))
    override = meta.get("gpu_shared_memory_override")
    if shared != "—" and override:
        shared += f" — overridden (IncreaseFixedSegment={int(override)})"

    dedicated_mb = meta.get("gpu_dedicated_memory_mb")
    if _is_carve_out(dedicated_mb, meta.get("gpu_shared_memory_mb")):
        return None, shared
    return _mb(dedicated_mb), shared


def _is_carve_out(dedicated_mb, shared_mb) -> bool:
    try:
        return float(dedicated_mb) < 1024 and float(shared_mb) > float(dedicated_mb)
    except (TypeError, ValueError):
        return False


def _series_counts(summary: dict | None, result: AnalysisResult) -> tuple[int, int, int]:
    """Return ``(skipped, success, failed)`` series counts for the run.

    Counted per ``expected_series`` rather than per test function, matching
    ``common.delivery.mail_title_suffix``: one test function can stand for
    several benchmark series.
    """
    if summary is None:
        functional = result.functional
        return functional.skipped, functional.passed, functional.failed + functional.error
    return (expected_cases(summary, {"skipped"}),
            expected_cases(summary, {"passed"}),
            expected_cases(summary, {"failed", "error"}))


def _render_run_summary(result: AnalysisResult, summary: dict | None,
                        baseline_meta: dict | None) -> tuple[str, int, bool]:
    """Return ``(rows_html, changed_count, has_baseline)`` for the Run Summary
    card, flagging every value that differs from the baseline run."""
    current = result.current_run
    cur_dedicated, cur_shared = _gpu_memory_text(summary)
    base_dedicated, base_shared = _gpu_memory_text(
        {"meta": baseline_meta} if baseline_meta else None)

    def _base(key: str, fmt=None) -> str | None:
        value = (baseline_meta or {}).get(key)
        if value in (None, ""):
            return None
        return fmt(value) if fmt else str(value)

    fields: list[tuple[str, str | None, str | None]] = [
        ("Current OV", current.ov_version if current else None, _base("ov_version")),
        ("Current purpose", current.purpose if current else None, _base("purpose")),
        ("Machine", current.machine_name if current else None, _base("machine")),
        ("GPU driver", current.gpu_driver_version if current else None,
         _base("gpu_driver_version")),
        ("GPU info", current.gpu_info if current else None, _base("gpu_info")),
    ]
    if cur_dedicated is not None:
        fields.append(("GPU dedicated memory", cur_dedicated, base_dedicated))
    fields += [
        ("GPU shared memory", cur_shared, base_shared),
        ("Host info", current.host_info if current else None, _base("host_info")),
        ("Memory size", current.memory_size if current else None,
         _base("host_memory_size_gb", lambda v: f"{float(v):.1f} GB")),
        ("Memory speed", current.memory_speed if current else None,
         _base("host_memory_speed_mhz", lambda v: f"{float(v):.0f} MHz")),
    ]

    has_baseline = bool(baseline_meta)
    rows: list[str] = []
    changed = 0
    for label, cur, base in fields:
        if cur is None and base is None:
            continue
        cur_text = html.escape(cur or "n/a")
        if not has_baseline:
            rows.append(f'<tr><td class="k">{label}</td><td>{cur_text}</td></tr>')
            continue
        differs = base is not None and base != cur
        changed += differs
        if base is None:
            base_cell = "<span class='muted'>n/a</span>"
        elif differs:
            base_cell = f"<span style='color:#a05a00'>{html.escape(base)}</span>"
        else:
            base_cell = "<span class='muted'>same</span>"
        cur_style = " style='color:#a05a00;font-weight:700'" if differs else ""
        rows.append(
            f'<tr><td class="k">{label}</td>'
            f'<td{cur_style}>{cur_text}</td>'
            f'<td>{base_cell}</td></tr>'
        )
    return "\n".join(rows), changed, has_baseline


def _mcp_banner(result: AnalysisResult) -> str:
    """Loud banner when daily_results could not be queried.

    A missing reference is an infrastructure fault, not a benign "no data":
    the whole comparison table is meaningless without it, so it is called out
    at the top of the report instead of being buried in a table cell.
    """
    problems: list[str] = []
    if result.baseline.status == "unavailable":
        problems.append(f"Reference: {result.baseline.detail or 'query failed'}")
    release = result.release
    if release is not None and release.status == "unavailable":
        problems.append(f"Release: {release.detail or 'query failed'}")
    if not problems:
        return ""

    url = result.baseline.source_url or (release.source_url if release else "") or ""
    items = "".join(f"<li>{html.escape(p)}</li>" for p in problems)
    return (
        "<div style='margin-bottom:14px;padding:12px 16px;border:2px solid #b42318;"
        "border-radius:8px;background:#fef3f2;color:#7a271a'>"
        "<div style='font-weight:700;font-size:14px'>"
        "daily_results MCP server unreachable — this report has no valid comparison</div>"
        f"<ul style='margin:6px 0 0 18px;font-size:12px'>{items}</ul>"
        f"<div style='margin-top:6px;font-size:12px'>Fix the server at "
        f"{html.escape(url)} and regenerate the report; the numbers below are "
        "<b>not</b> compared against a reference.</div>"
        "</div>"
    )


def _render_output_quality(result: AnalysisResult, *, include_image_sample: bool = False) -> str:
    quality = result.output_quality
    rows = quality.rows if quality else []
    counts = {status: sum(row.status == status for row in rows) for status in ("pass", "warning", "unavailable")}
    detail = quality.detail if quality else "Output quality checks were not run."
    table_rows: list[str] = []
    comparison_warnings = sum(row.baseline_warning + row.release_warning for row in rows)
    comparisons = ("baseline",) if result.release and result.release.status == "disabled" else ("baseline", "release")
    column_count = 5 + len(comparisons)
    visible_rows = [row for row in rows if row.status != "pass"]
    image_sample = next(
        (row for row in rows if row.kind == "image" and row.status == "pass" and row.current_preview), None,
    ) if include_image_sample else None
    if image_sample is not None:
        visible_rows.append(image_sample)
    for index, row in enumerate(visible_rows):
        background = "#f3f6fa" if index % 2 == 0 else "#ffffff"
        separator = "3px solid #a8b5c5"
        previews: list[str] = []
        for label, preview in (("Current", row.current_preview), ("Baseline", row.baseline_preview), ("Release", row.release_preview)):
            if label == "Release" and "release" not in comparisons:
                continue
            if not preview:
                continue
            if row.kind == "image" and preview.startswith("data:image/jpeg;base64,"):
                content = f'<img alt="{label} output" src="{html.escape(preview, quote=True)}" width="128" style="display:block;max-width:128px;max-height:128px;height:auto" />'
            else:
                excerpt = preview[:200] + ("..." if len(preview) > 200 else "")
                content = f'<pre style="white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;margin:4px 0;font-size:12px">{html.escape(excerpt)}</pre>'
            previews.append(
                f'<div class="quality-preview">'
                f'<div style="font-size:11px;font-weight:700;margin-bottom:4px">{label}</div>{content}</div>'
            )
        metric = "IoU" if row.kind == "text" else "SSIM"
        cell_style = (
            f"background:{background};vertical-align:top;padding:10px 8px;"
            f"border-bottom:{'0' if previews else separator}"
        )
        score_cells: list[str] = []
        for label in comparisons:
            score = getattr(row, f"{label}_score")
            condition = getattr(row, f"{label}_comparison")
            low_similarity = getattr(row, f"{label}_warning")
            condition_text = {
                "verified": "Matched conditions",
                "unverified": "Unverified conditions; reference only",
                "different": "Different conditions; reference only",
                "unavailable": "Comparison unavailable",
            }.get(condition, "Comparison unavailable")
            measurement = f"{metric} {score:.3f}" if score is not None else "Not evaluated"
            if score is None:
                explanations = [item.split(": ", 1)[1] for item in row.reasons if item.startswith(f"{label}: ")]
                condition_text = "; ".join(explanations) or "No comparable output"
            warning_text = '<div style="color:#a05a00">Low similarity</div>' if low_similarity else ""
            score_cells.append(
                f'<td data-comparison-condition="{condition}" style="{cell_style}">{measurement}'
                f'<div class="muted" style="font-size:11px">{html.escape(condition_text)}</div>{warning_text}</td>'
            )
        reason = "<br>".join(
            html.escape(item) for item in row.reasons
            if not (item.startswith("baseline: ") and row.baseline_score is None)
            and not (item.startswith("release: ") and row.release_score is None)
        )
        sample_attribute = ' data-quality-sample="true"' if row is image_sample else ""
        status_text = row.status.upper()
        if row is image_sample:
            status_text += " (sample)"
            reason = "Image preview sample; no quality issue."
        table_rows.append(
            f'<tr data-quality-status="{row.status}"{sample_attribute}><td style="{cell_style}"><strong>{html.escape(row.model)}</strong><br>{html.escape(row.precision)}</td>'
            f'<td style="{cell_style}">{html.escape(row.prompt)}</td><td style="{cell_style}">{html.escape(row.kind)}</td>'
            f'<td style="{cell_style}">{status_text}</td>'
            f'{"".join(score_cells)}'
            f'<td style="{cell_style};overflow-wrap:anywhere">{reason}</td></tr>'
        )
        if previews:
            preview_width = 100 / len(previews)
            preview_cells = "".join(
                f'<td class="quality-preview-cell" width="{preview_width:.2f}%" '
                f'style="width:{preview_width:.2f}%;vertical-align:top;border:0;padding:8px 12px;box-sizing:border-box;background:{background}">{preview}</td>'
                for preview in previews
            )
            table_rows.append(
                f'<tr class="quality-preview-row"><td colspan="{column_count}" style="padding:0 0 12px;background:{background};border-bottom:{separator}">'
                '<table class="quality-preview-table" role="presentation" '
                'style="width:100%;table-layout:fixed;border-collapse:collapse"><tbody><tr>'
                f'{preview_cells}</tr></tbody></table></td></tr>'
            )
    if not table_rows:
        if rows:
            return ""
        message = detail or "No outputs available; quality not verified."
        table_rows.append(f'<tr><td colspan="{column_count}">{html.escape(message)}</td></tr>')
    comparison_headers = "".join(f'<th>vs {label.title()}</th>' for label in comparisons)
    return (
        '<section style="margin:18px 0" id="output-quality-checks"><h2>Output Quality Checks</h2>'
        f'<div class="muted" style="margin-bottom:8px">Output checks: {len(rows)} | PASS: {counts["pass"]} | '
        f'WARNING: {counts["warning"]} | UNAVAILABLE: {counts["unavailable"]} | Low similarity: {comparison_warnings}</div>'
        '<div style="overflow-x:auto"><table class="quality-results"><thead><tr><th>Model / Precision</th><th>Prompt / Iteration</th>'
        f'<th>Output</th><th>Output Status</th>{comparison_headers}<th>Findings</th>'
        f'</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div></section>'
    )


def render_analysis_html(result: AnalysisResult, summary: dict | None = None,
                         image_assets: dict[str, dict] | None = None,
                         baseline_meta: dict | None = None, *, include_image_sample: bool = False) -> str:
    """Return a standalone HTML report for analysis-focused review.

    ``summary`` is the normalised daily summary dict (same shape as
    ``daily.*.summary.json``). Image previews are included with output-quality
    findings. ``image_assets`` is retained for compatibility with existing callers.
    ``baseline_meta`` is the baseline run's ``meta`` block, used to flag
    environment differences in the Run Summary card.
    ``include_image_sample`` adds one labeled PASS image for layout review only;
    normal reports keep PASS outputs hidden.
    """
    from datetime import datetime as _dt  # noqa: PLC0415

    output_quality_table = _render_output_quality(result, include_image_sample=include_image_sample)

    improved_rows = sorted(
        [r for r in result.rows if r.verdict == "improved" and r.improvement_pct is not None],
        key=lambda r: r.improvement_pct,
        reverse=True,
    )[:10]
    regressed_rows = sorted(
        [r for r in result.rows if r.verdict == "regressed" and r.improvement_pct is not None],
        key=lambda r: r.improvement_pct,
    )[:10]
    # Keep original engine order so this table matches the main report table order.
    all_rows = list(result.rows)
    show_release = any(
        row.release_value is not None and math.isfinite(row.release_value)
        for row in all_rows
    )
    column_count = 13 if show_release else 11
    fluctuation_same = sum(1 for r in result.rows if r.within_fluctuation)
    # Top table counts one benchmark series as one unit, including the series
    # skipped tests would have produced.
    series_skipped, series_success, series_failed = _series_counts(summary, result)
    series_total = series_skipped + series_success + series_failed

    baseline_text = "not found"
    if result.baseline.status == "found":
        baseline_text = f"{result.baseline.stamp or ''} / {result.baseline.ov_version or 'unknown'}"
    elif result.baseline.status == "unavailable":
        baseline_text = f"UNAVAILABLE — {result.baseline.detail or 'daily_results query failed'}"

    summary_rows, changed_fields, has_baseline_meta = _render_run_summary(
        result, summary, baseline_meta)
    # Without the baseline column there is nowhere else the baseline is named.
    summary_head = ""
    summary_note = ""
    baseline_row = f'<tr><td class="k">Baseline</td><td>{html.escape(baseline_text)}</td></tr>'
    release_row = ""
    rel = result.release
    if rel is not None and rel.status != "disabled":
        if rel.status == "found":
            release_text = (f"{rel.stamp or ''} / {rel.ov_version or 'unknown'} "
                            f"({rel.matched_count} series)")
        elif rel.status == "not_found":
            release_text = rel.detail or "no release run published yet"
        else:
            release_text = f"unavailable ({rel.detail or 'query failed'})"
        release_row = f'<tr><td class="k">Release</td><td>{html.escape(release_text)}</td></tr>'

    mcp_banner = _mcp_banner(result)
    if has_baseline_meta:
        baseline_row = ""
        summary_head = (
            "<tr>"
            "<th style='background:transparent;font-size:11px;color:#6b7280;padding:0 12px 6px 0'>Field</th>"
            "<th style='background:transparent;font-size:11px;color:#6b7280;padding:0 0 6px'>Current</th>"
            "<th style='background:transparent;font-size:11px;color:#6b7280;padding:0 0 6px'>"
            f"Baseline {html.escape(result.baseline.stamp or '')}</th>"
            "</tr>"
        )
        summary_note = (
            "<div style='font-size:12px;color:#a05a00;margin-bottom:8px'>"
            f"{changed_fields} field(s) differ from the baseline run.</div>"
            if changed_fields else
            "<div style='font-size:12px;color:#18794e;margin-bottom:8px'>"
            "Environment matches the baseline run.</div>"
        )

    badges = {
        "green":  ("GREEN",  "#18794e"),
        "yellow": ("YELLOW", "#a05a00"),
        "red":    ("RED",    "#b42318"),
        "gray":   ("GRAY",   "#475467"),
    }
    badge = badges.get(result.overall_status, (result.overall_status.upper(), "#475467"))
    release_status_html = "".join(
        '<td class="comparison-status" style="border:0;padding:0 0 0 20px;vertical-align:top">'
        f'<div class="muted" style="font-size:12px;margin-bottom:4px">{label}</div>'
        f'<span class="badge" data-comparison="{comparison}" data-status="{status}" '
        f'style="background:{badges.get(status, (str(status).upper(), "#475467"))[1]};font-size:16px;padding:6px 18px">'
        f'{badges.get(status, (str(status).upper(), "#475467"))[0]}</span></td>'
        for label, comparison, status in (
            ("Current vs Release", "release", result.release_status or "gray"),
            ("Baseline vs Release", "baseline-release", result.baseline_release_status or "gray"),
        )
    ) if rel is None or rel.status != "disabled" else ""

    generated_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    def _fmt_pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:+.2f}%"

    def _fmt_num(v: float | None, unit: str = "") -> str:
        if v is None or not math.isfinite(v):
            return "n/a"
        s = f"{v:.3f}"
        return f"{s} {html.escape(unit)}".strip() if unit else s

    def _fmt_cv(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:.2f}%"

    def _delta_style(verdict: str, within_fluct: bool) -> str:
        if within_fluct:
            return "color:#6b7280"          # muted gray — same by fluctuation
        if verdict == "regressed":
            return "color:#b42318;font-weight:700"
        if verdict == "improved":
            return "color:#18794e;font-weight:700"
        return ""

    def _cv_style(v: float | None) -> str:
        if v is None:
            return ""
        if v > 0.10:
            return "color:#b42318"          # >10% CV → noisy
        if v > 0.05:
            return "color:#a05a00"          # 5–10% → moderate
        return "color:#18794e"              # ≤5% → stable

    def _fluct_badge(within: bool) -> str:
        if within:
            return "<span title='Delta is within historical fluctuation range — treated as same' style='font-size:11px;background:#e5e7eb;color:#374151;padding:1px 6px;border-radius:999px'>fluct</span>"
        return ""

    def _release_delta_style(v: float | None) -> str:
        if v is None:
            return ""
        if v < 0:
            return "color:#b42318"
        if v > 0:
            return "color:#18794e"
        return ""

    def _row_html(row, show_fluct: bool = True) -> str:
        k = row.key
        unit = row.unit or ""
        delta_s = _delta_style(row.verdict, row.within_fluctuation)
        cv_s = _cv_style(row.history_cv)
        fluct = _fluct_badge(row.within_fluctuation) if show_fluct else ""
        delta_style = f"text-align:right;{delta_s};white-space:nowrap" if delta_s else "text-align:right;white-space:nowrap"
        cv_style = f"text-align:right;{cv_s};white-space:nowrap" if cv_s else "text-align:right;white-space:nowrap"
        rel_s = _release_delta_style(row.release_improvement_pct)
        rel_style = f"text-align:right;{rel_s};white-space:nowrap" if rel_s else "text-align:right;white-space:nowrap"
        release_cells = (
            f"<td class='num' style='text-align:right;white-space:nowrap'>{_fmt_num(row.release_value, unit)}</td>\n"
            f"<td class='num' style='{rel_style}'>{_fmt_pct(row.release_improvement_pct)}</td>\n"
        ) if show_release else ""
        return (
            "<tr>\n"
            f"<td>{html.escape(k.model)}</td>\n"
            f"<td>{html.escape(k.precision)}</td>\n"
            f"<td style='white-space:nowrap'>{k.in_token}&nbsp;/&nbsp;{k.out_token}</td>\n"
            f"<td>{html.escape(k.exec_mode)}</td>\n"
            f"<td class='num' style='text-align:right;white-space:nowrap'>{_fmt_num(row.current_value, unit)}</td>\n"
            f"<td class='num' style='text-align:right;white-space:nowrap'>{_fmt_num(row.baseline_value, unit)}</td>\n"
            f"<td class='num' style='{delta_style}'>{_fmt_pct(row.improvement_pct)}{fluct}</td>\n"
            f"{release_cells}"
            f"<td class='num' style='text-align:right'>{row.history_count}</td>\n"
            f"<td class='num' style='text-align:right;white-space:nowrap'>{_fmt_num(row.history_sigma, unit)}</td>\n"
            f"<td class='num' style='{cv_style}'>{_fmt_cv(row.history_cv)}</td>\n"
            f"<td style='font-size:11px;color:#6b7280'>{html.escape(row.reference_source)}</td>\n"
            "</tr>"
        )

    improved_table  = "\n".join(_row_html(r) for r in improved_rows)  or f"<tr><td colspan='{column_count}' style='color:#6b7280;text-align:center'>No improved rows</td></tr>"
    regressed_table = "\n".join(_row_html(r) for r in regressed_rows) or f"<tr><td colspan='{column_count}' style='color:#6b7280;text-align:center'>No regressed rows</td></tr>"
    all_table       = "\n".join(_row_html(r, show_fluct=True) for r in all_rows)
    failed_rows = ""
    if result.functional.issues:
        rows: list[str] = []
        for issue in result.functional.issues[:10]:
            msg = issue.message or "(no message captured)"
            rows.append(
                "<tr>"
                f"<td style='font-family:Consolas,Monaco,monospace;font-size:12px'>{html.escape(issue.nodeid)}</td>"
                f"<td>{html.escape(issue.outcome)}</td>"
                f"<td style='white-space:pre-wrap'>{html.escape(msg)}</td>"
                "</tr>"
            )
        failed_rows = "\n".join(rows)

    # Column header definitions — (label, tooltip)
    COL_DEFS = [
        ("Model",      "Model name and architecture (e.g. llama-3.1-8b)"),
        ("Precision",  "Weight/activation data type used for inference (e.g. FP16, INT4, INT8)"),
        ("In / Out",   "Input token count / Output token count used in the benchmark run"),
        ("Mode",       "Execution mode: 'latency' = single-request, 'throughput' = concurrent batches"),
        ("Current",    "Measured value from today's run (unit shown alongside the number)"),
        ("Reference",  "Value of the most recent successful timer-scheduled run with performance data on this machine"),
        ("Delta",      "Relative change vs reference (+% = improved, -% = regressed). "
                       "Grayed-out 'fluct' badge means the delta is within historical noise — treated as same."),
        ("Release",    "Value of the newest published release build for this machine, read from the "
                       "central daily_results server; 'n/a' means no release run covers this series yet"),
        ("Δ Release",  "Relative change vs the release build (+% = better than release). "
                       "Informational only — it never changes the verdict"),
        ("N",          "Number of historical comparable runs (same machine / model / precision / mode) "
                       "used to build the reference distribution"),
        ("Sigma (σ)",  "Standard deviation of historical values — larger σ means the machine is noisier "
                       "for this series; a 1 ms delta on a series with σ=2 ms is not meaningful"),
        ("CV",         "Coefficient of Variation = σ / mean.  ≤5% (green) = stable, "
                       "5–10% (orange) = moderate noise, >10% (red) = high noise — be cautious with verdicts"),
        ("Ref Source", "How the reference value was chosen: "
                       "'baseline' = latest successful timer-scheduled run with performance data, "
                       "'no_baseline' / 'unit_mismatch' = nothing comparable was found"),
    ]
    if not show_release:
        COL_DEFS = [(label, tip) for label, tip in COL_DEFS if label not in {"Release", "Δ Release"}]

    def _th(label: str, tip: str) -> str:
        numeric_headers = {"Current", "Reference", "Delta", "Release", "Δ Release", "N", "Sigma (σ)", "CV"}
        th_class = "num-h" if label in numeric_headers else ""
        return (f"<th title='{html.escape(tip)}' "
            f"class='{th_class}' style='cursor:help;border-bottom:2px solid #bcd0f0'>{label} "
                f"<span style='font-weight:400;font-size:10px;color:#6b7280'>(?)</span></th>")

    thead = "<tr>" + "".join(_th(l, t) for l, t in COL_DEFS) + "</tr>"

    col_legend_rows = "\n".join(
        f"<tr><td style='font-weight:700;white-space:nowrap;padding:5px 10px 5px 0'>{l}</td>"
        f"<td style='color:#374151;padding:5px 0'>{html.escape(t)}</td></tr>"
        for l, t in COL_DEFS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Daily Analysis Report</title>
    <style>
        :root {{
            --bg: #f6f8fb;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --line: #d9dee7;
            --accent: #0f4c81;
        }}
        body {{ margin: 0; background: radial-gradient(circle at top right, #e7eef9 0%, var(--bg) 38%); color: var(--text); font-family: "Segoe UI", "Noto Sans", sans-serif; }}
        .wrap {{ max-width: 1380px; margin: 0 auto; padding: 24px; }}
        .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 20px; box-shadow: 0 8px 28px rgba(21, 34, 56, 0.06); }}
        h1 {{ margin: 0 0 4px; font-size: 26px; letter-spacing: 0.2px; }}
        h2 {{ margin: 0 0 10px; font-size: 16px; color: var(--accent); }}
        h3 {{ margin: 0 0 8px; font-size: 14px; font-weight: 700; }}
        .kvs-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .kvs-table td {{ border: 0; padding: 4px 0; vertical-align: top; }}
        .kvs-table .k {{ color: var(--muted); white-space: nowrap; width: 170px; padding-right: 12px; }}
        .muted {{ color: var(--muted); }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; color: #fff; font-weight: 700; font-size: 13px; letter-spacing: 0.5px; }}
        .stat-block {{ text-align: center; padding: 10px 6px; }}
        .stat-block .val {{ font-size: 28px; font-weight: 700; }}
        .stat-block .lbl {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
        .stat-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; text-align: center; }}
        .stat-table th {{ text-align: center; font-size: 11px; letter-spacing: 0.4px; text-transform: uppercase; color: var(--muted); }}
        .stat-table td {{ text-align: center; font-size: 26px; font-weight: 700; border-bottom: 0; padding: 8px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f3f7fe; font-weight: 700; padding: 9px 8px; text-align: left; }}
        td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; }}
        .num, .num-h {{ text-align: right; font-variant-numeric: tabular-nums; }}
        tr:hover td {{ background: #f8faff; }}
        .legend-table {{ font-size: 13px; width: 100%; border-collapse: collapse; }}
        .legend-table tr:nth-child(even) td {{ background: #f8fafc; }}
        @media (max-width: 640px) {{
            .quality-results {{ table-layout: fixed; width: 100%; }}
            .quality-results > thead > tr > th, .quality-results > tbody > tr > td {{ overflow-wrap: anywhere; word-break: break-word; }}
            .quality-preview-table, .quality-preview-table tbody, .quality-preview-table tr {{ display: block; width: 100%; }}
            .quality-preview-cell {{ display: block; width: 100% !important; }}
        }}
        @media (max-width: 980px) {{
            .wrap {{ padding: 14px; }}
        }}
    </style>
</head>
<body>
<div class="wrap">

    <!-- Header -->
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:14px">
        <div>
            <h1>Daily Analysis Report</h1>
            <div class="muted" style="font-size:12px">Generated {generated_at}</div>
        </div>
        <table role="presentation" style="width:auto;border-collapse:collapse"><tr>
            <td class="comparison-status" style="border:0;padding:0;vertical-align:top">
                <div class="muted" style="font-size:12px;margin-bottom:4px">Current vs Baseline</div>
                <span class="badge" data-comparison="baseline" data-status="{result.overall_status}" style="background:{badge[1]};font-size:16px;padding:6px 18px">{badge[0]}</span>
            </td>
            {release_status_html}
        </tr></table>
    </div>

    {mcp_banner}

    <!-- Top stat row -->
    <div class="card" style="margin-bottom:14px;padding:0;overflow:hidden">
        <table role="presentation" class="stat-table">
            <tr>
                <th>Total</th>
                <th>Skip</th>
                <th>Success</th>
                <th>Fail</th>
            </tr>
            <tr>
                <td>{series_total}</td>
                <td style="color:{'#a05a00' if series_skipped else '#6b7280'}">{series_skipped}</td>
                <td style="color:{'#18794e' if series_success else '#6b7280'}">{series_success}</td>
                <td style="color:{'#b42318' if series_failed else '#18794e'}">{series_failed}</td>
            </tr>
        </table>
    </div>

    <!-- Summary -->
    <div class="card" style="margin-bottom:14px">
        <h2>Run Summary</h2>
        {summary_note}
        <table role="presentation" class="kvs-table">
            {summary_head}
            {summary_rows}
            {baseline_row}
            {release_row}
        </table>
    </div>

    <!-- Functional issues -->
    <div class="card" style="margin-bottom:14px">
        <h2>Failed Tests</h2>
        <div style="font-size:12px;color:#6b7280;margin-bottom:8px">
            Showing up to 10 failed/error tests from this run.
        </div>
        <div style="overflow-x:auto">
            <table>
                <thead>
                    <tr>
                        <th style="border-bottom:2px solid #bcd0f0">Node ID</th>
                        <th style="border-bottom:2px solid #bcd0f0">Outcome</th>
                        <th style="border-bottom:2px solid #bcd0f0">Message</th>
                    </tr>
                </thead>
                <tbody>{failed_rows or "<tr><td colspan='3' style='color:#6b7280;text-align:center'>No functional issues</td></tr>"}</tbody>
            </table>
        </div>
    </div>

    <!-- Top Regressions -->
    <div class="card" style="margin-bottom:14px">
        <h2>Top Regressions</h2>
        <div style="overflow-x:auto">
            <table>
                <thead>{thead}</thead>
                <tbody>{regressed_table}</tbody>
            </table>
        </div>
    </div>

    <!-- Top Improvements -->
    <div class="card" style="margin-bottom:14px">
        <h2>Top Improvements</h2>
        <div style="overflow-x:auto">
            <table>
                <thead>{thead}</thead>
                <tbody>{improved_table}</tbody>
            </table>
        </div>
    </div>

    {output_quality_table}

    <!-- All rows -->
    <div class="card" style="margin-bottom:14px">
        <h2>All Performance Results ({len(all_rows)} series)</h2>
        <div style="margin-top:10px;overflow-x:auto">
            <table>
                <thead>{thead}</thead>
                <tbody>{all_table}</tbody>
            </table>
        </div>
    </div>

    <!-- Reference material: kept last, it is only needed while learning the report -->
    <div class="card" style="margin-bottom:14px">
        <h2>Analysis Methodology</h2>
        <div style="font-size:13px;line-height:1.65;color:#374151">
            <b>Reference</b> = the latest earlier successful timer-scheduled run with performance data on the same machine, matched by model, precision, input/output tokens, and mode. Historical runs provide noise statistics; they do not replace a missing reference.<br>
            <b>Fluctuation guard</b>: if |delta| ≤ 1.5&nbsp;×&nbsp;σ the series is treated as <em>same</em> regardless of sign, because the change is within normal machine noise.<br>
            <b>CV</b> (Coefficient of Variation) shows how noisy each individual series is — high CV means even large deltas may not be reliable.
        </div>
    </div>

    <div class="card">
        <h2>Column Reference Guide</h2>
        <div style="font-size:12px;color:#6b7280;margin-bottom:8px">
            Outlook compatibility mode: this section is always expanded.
        </div>
        <div style="margin-top:10px;overflow-x:auto">
            <table class="legend-table">
                <thead><tr>
                    <th style="width:110px;background:#f3f7fe">Column</th>
                    <th style="background:#f3f7fe">Description</th>
                </tr></thead>
                <tbody>{col_legend_rows}</tbody>
            </table>
        </div>
    </div>

</div>
</body>
</html>
"""


def write_analysis_html(html_path: Path, result: AnalysisResult,
                        summary: dict | None = None,
                        image_assets: dict[str, dict] | None = None,
                        baseline_meta: dict | None = None) -> Path:
        """Write the analysis-focused HTML report."""
        html_path.write_text(
            render_analysis_html(result, summary, image_assets, baseline_meta),
            encoding="utf-8")
        return html_path
