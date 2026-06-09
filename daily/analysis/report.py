"""Text report renderer for AnalysisResult.

Produces the ``[ Analysis summary ]`` block that is prepended to the
daily text report.  All formatting lives here so the engine and
persistence layer stay format-agnostic.
"""

from __future__ import annotations

import html
from pathlib import Path

from .types import AnalysisResult


def render_analysis_summary(result: AnalysisResult) -> str:
    """Return the full ``[ Analysis summary ]`` text block."""
    lines = ["[ Analysis summary ]"]

    # --- functional ---
    f = result.functional
    lines.append(
        f"- Functional: total={f.total} passed={f.passed} "
        f"failed={f.failed} error={f.error} skipped={f.skipped}"
    )
    for issue in f.issues[:5]:
        lines.append(f"  ! {issue.nodeid} [{issue.outcome}]: {issue.message}")

    # --- baseline ---
    b = result.baseline
    if b.status == "not_found":
        lines.append("- Baseline comparison: no older run found for this machine.")
    else:
        lines.append(
            f"- Baseline: stamp={b.stamp} ov={b.ov_version or 'unknown'}"
            + (f" ({b.selection_reason})" if b.selection_reason else "")
        )

    lkg = result.last_known_good
    if lkg is not None:
        if lkg.status == "found":
            lines.append(
                "- Last known good: "
                f"stamp={lkg.stamp} ov={lkg.ov_version or 'unknown'}"
            )
        else:
            lines.append("- Last known good: not found")

    if b.status != "not_found":
        # --- performance aggregate ---
        p = result.performance
        fluctuation_same = sum(1 for r in result.rows if r.within_fluctuation)
        lines.append(
            f"- Performance: compared={p.compared} improved={p.improved} "
            f"same={p.same} regressed={p.regressed}"
        )
        if fluctuation_same > 0:
            lines.append(
                f"- Fluctuation guard: {fluctuation_same} series were treated as same "
                "because the delta stayed within historical variation."
            )

        # --- model deltas (top movers) ---
        if result.models:
            lines.append("- Model deltas:")
            for m in result.models[:10]:
                avg_str = (
                    f"{m.avg_improvement_pct * 100:+.2f}%"
                    if m.avg_improvement_pct is not None
                    else "n/a"
                )
                lines.append(
                    f"  {m.model}: avg {avg_str} | "
                    f"improved={m.improved} same={m.same} regressed={m.regressed}"
                )

        # --- top regressions ---
        if result.top_regressions:
            lines.append("- Top regressions:")
            for row in result.top_regressions:
                pct_str = (
                    f"{row.improvement_pct * 100:+.2f}%"
                    if row.improvement_pct is not None
                    else "n/a"
                )
                k = row.key
                lines.append(
                    f"  {k.model} | {k.precision} | "
                    f"in={k.in_token} out={k.out_token} | "
                    f"{k.exec_mode} [{row.unit or ''}]: "
                    f"{pct_str} "
                    f"(cur={row.current_value:.3f}, ref={row.baseline_value:.3f}, "
                    f"src={row.reference_source}, n={row.history_count})"
                )

    # --- overall verdict ---
    _verdict_label = {
        "green":  "No issues detected.",
        "yellow": "Performance regression detected.",
        "red":    "Functional issues detected.",
        "gray":   "Baseline comparison unavailable.",
    }
    lines.append(
        f"- Overall verdict: {_verdict_label.get(result.overall_status, result.overall_status)}"
    )

    return "\n".join(lines)


def prepend_to_report(report_path: Path, result: AnalysisResult) -> None:
    """Prepend the analysis summary block to an existing text report file."""
    block = render_analysis_summary(result)
    current = report_path.read_text(encoding="utf-8")
    report_path.write_text(block + "\n\n" + current, encoding="utf-8")


def render_analysis_html(result: AnalysisResult) -> str:
    """Return a standalone HTML report for analysis-focused review."""
    from datetime import datetime as _dt  # noqa: PLC0415

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
    fluctuation_same = sum(1 for r in result.rows if r.within_fluctuation)

    badge = {
        "green":  ("GREEN",  "#18794e"),
        "yellow": ("YELLOW", "#a05a00"),
        "red":    ("RED",    "#b42318"),
        "gray":   ("GRAY",   "#475467"),
    }.get(result.overall_status, (result.overall_status.upper(), "#475467"))

    generated_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    def _fmt_pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:+.2f}%"

    def _fmt_num(v: float | None, unit: str = "") -> str:
        if v is None:
            return "n/a"
        s = f"{v:.3f}"
        return f"{s} {unit}".strip() if unit else s

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

    def _row_html(row, show_fluct: bool = True) -> str:
        k = row.key
        unit = row.unit or ""
        delta_s = _delta_style(row.verdict, row.within_fluctuation)
        cv_s = _cv_style(row.history_cv)
        fluct = _fluct_badge(row.within_fluctuation) if show_fluct else ""
        return (
            "<tr>"
            f"<td>{html.escape(k.model)}</td>"
            f"<td>{html.escape(k.precision)}</td>"
            f"<td style='white-space:nowrap'>{k.in_token}&nbsp;/&nbsp;{k.out_token}</td>"
            f"<td>{html.escape(k.exec_mode)}</td>"
            f"<td class='num' style='white-space:nowrap'>{_fmt_num(row.current_value, unit)}</td>"
            f"<td class='num' style='white-space:nowrap'>{_fmt_num(row.baseline_value, unit)}</td>"
            f"<td class='num' style='{delta_s};white-space:nowrap'>{_fmt_pct(row.improvement_pct)}{fluct}</td>"
            f"<td class='num'>{row.history_count}</td>"
            f"<td class='num' style='white-space:nowrap'>{_fmt_num(row.history_sigma, unit)}</td>"
            f"<td class='num' style='{cv_s};white-space:nowrap'>{_fmt_cv(row.history_cv)}</td>"
            f"<td style='font-size:11px;color:#6b7280'>{html.escape(row.reference_source)}</td>"
            "</tr>"
        )

    improved_table  = "".join(_row_html(r) for r in improved_rows)  or "<tr><td colspan='11' style='color:#6b7280;text-align:center'>No improved rows</td></tr>"
    regressed_table = "".join(_row_html(r) for r in regressed_rows) or "<tr><td colspan='11' style='color:#6b7280;text-align:center'>No regressed rows</td></tr>"
    all_table       = "".join(_row_html(r, show_fluct=True) for r in all_rows)
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
        failed_rows = "".join(rows)

    baseline_text = "not found"
    if result.baseline.status == "found":
        baseline_text = f"{result.baseline.stamp or ''} / {result.baseline.ov_version or 'unknown'}"

    current = result.current_run

    def _safe_text(value: str | None) -> str:
        return html.escape(value) if value else "n/a"

    # Column header definitions — (label, tooltip)
    COL_DEFS = [
        ("Model",      "Model name and architecture (e.g. llama-3.1-8b)"),
        ("Precision",  "Weight/activation data type used for inference (e.g. FP16, INT4, INT8)"),
        ("In / Out",   "Input token count / Output token count used in the benchmark run"),
        ("Mode",       "Execution mode: 'latency' = single-request, 'throughput' = concurrent batches"),
        ("Current",    "Measured value from today's run (unit shown alongside the number)"),
        ("Reference",  "Statistical reference: mean of the top-K best runs from the recent history window"),
        ("Delta",      "Relative change vs reference (+% = improved, -% = regressed). "
                       "Grayed-out 'fluct' badge means the delta is within historical noise — treated as same."),
        ("N",          "Number of historical comparable runs (same machine / model / precision / mode) "
                       "used to build the reference distribution"),
        ("Sigma (σ)",  "Standard deviation of historical values — larger σ means the machine is noisier "
                       "for this series; a 1 ms delta on a series with σ=2 ms is not meaningful"),
        ("CV",         "Coefficient of Variation = σ / mean.  ≤5% (green) = stable, "
                       "5–10% (orange) = moderate noise, >10% (red) = high noise — be cautious with verdicts"),
        ("Ref Source", "How the reference value was chosen: "
                       "'topk_mean' = top-K best historical runs averaged (preferred), "
                       "'baseline' = direct previous run (fallback when history is short)"),
    ]

    def _th(label: str, tip: str) -> str:
        numeric_headers = {"Current", "Reference", "Delta", "N", "Sigma (σ)", "CV"}
        th_class = "num-h" if label in numeric_headers else ""
        return (f"<th title='{html.escape(tip)}' "
            f"class='{th_class}' style='cursor:help;border-bottom:2px solid #bcd0f0'>{label} "
                f"<span style='font-weight:400;font-size:10px;color:#6b7280'>(?)</span></th>")

    thead = "<tr>" + "".join(_th(l, t) for l, t in COL_DEFS) + "</tr>"

    col_legend_rows = "".join(
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
        .grid2 {{ display: grid; gap: 14px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .grid3 {{ display: grid; gap: 14px; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
        h1 {{ margin: 0 0 4px; font-size: 26px; letter-spacing: 0.2px; }}
        h2 {{ margin: 0 0 10px; font-size: 16px; color: var(--accent); }}
        h3 {{ margin: 0 0 8px; font-size: 14px; font-weight: 700; }}
        .kvs {{ display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; font-size: 13px; align-items: baseline; }}
        .kvs .k {{ color: var(--muted); white-space: nowrap; }}
        .muted {{ color: var(--muted); }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; color: #fff; font-weight: 700; font-size: 13px; letter-spacing: 0.5px; }}
        .stat-block {{ text-align: center; padding: 10px 6px; }}
        .stat-block .val {{ font-size: 28px; font-weight: 700; }}
        .stat-block .lbl {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f3f7fe; font-weight: 700; padding: 9px 8px; text-align: left; position: sticky; top: 0; }}
        td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; }}
        .num, .num-h {{ text-align: right; font-variant-numeric: tabular-nums; }}
        tr:hover td {{ background: #f8faff; }}
        details > summary {{ cursor: pointer; font-weight: 700; font-size: 14px; padding: 6px 2px; color: var(--accent); user-select: none; }}
        details > summary:hover {{ opacity: 0.75; }}
        .legend-table {{ font-size: 13px; width: 100%; border-collapse: collapse; }}
        .legend-table tr:nth-child(even) td {{ background: #f8fafc; }}
        @media (max-width: 980px) {{ .grid2, .grid3 {{ grid-template-columns: 1fr; }} .wrap {{ padding: 14px; }} }}
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
        <span class="badge" style="background:{badge[1]};font-size:16px;padding:6px 18px">{badge[0]}</span>
    </div>

    <!-- Top stat row -->
    <div class="grid3" style="margin-bottom:14px">
        <div class="card stat-block">
            <div class="val" style="color:{'#b42318' if result.performance.regressed else '#18794e'}">{result.performance.regressed}</div>
            <div class="lbl">Regressions</div>
        </div>
        <div class="card stat-block">
            <div class="val" style="color:{'#18794e' if result.performance.improved else '#6b7280'}">{result.performance.improved}</div>
            <div class="lbl">Improvements</div>
        </div>
        <div class="card stat-block">
            <div class="val">{result.performance.compared}</div>
            <div class="lbl">Series Compared</div>
        </div>
    </div>

    <!-- Summary + Methodology -->
    <div class="grid2" style="margin-bottom:14px">
        <div class="card">
            <h2>Run Summary</h2>
            <div class="kvs">
                <div class="k">Current OV</div><div>{_safe_text(current.ov_version if current else None)}</div>
                <div class="k">Current purpose</div><div>{_safe_text(current.purpose if current else None)}</div>
                <div class="k">Machine</div><div>{_safe_text(current.machine_name if current else None)}</div>
                <div class="k">GPU driver</div><div>{_safe_text(current.gpu_driver_version if current else None)}</div>
                <div class="k">GPU info</div><div>{_safe_text(current.gpu_info if current else None)}</div>
                <div class="k">Host info</div><div>{_safe_text(current.host_info if current else None)}</div>
                <div class="k">Memory size</div><div>{_safe_text(current.memory_size if current else None)}</div>
                <div class="k">Memory speed</div><div>{_safe_text(current.memory_speed if current else None)}</div>
                <div class="k">Baseline</div><div>{html.escape(baseline_text)}</div>
                <div class="k">Selection reason</div><div>{html.escape(result.baseline.selection_reason or "—")}</div>
                <div class="k">Functional</div><div>failed={result.functional.failed}&nbsp;&nbsp;error={result.functional.error}&nbsp;&nbsp;skipped={result.functional.skipped}</div>
                <div class="k">Perf same</div><div>{result.performance.same} ({fluctuation_same} by fluctuation guard)</div>
            </div>
        </div>
        <div class="card">
            <h2>Analysis Methodology</h2>
            <div style="font-size:13px;line-height:1.65;color:#374151">
                <b>Reference</b> = mean of the best <b>top-5</b> runs from a <b>10-run history window</b> (same machine · model · precision · mode).<br>
                <b>Fluctuation guard</b>: if |delta| ≤ 1.5&nbsp;×&nbsp;σ the series is treated as <em>same</em> regardless of sign, because the change is within normal machine noise.<br>
                <b>CV</b> (Coefficient of Variation) shows how noisy each individual series is — high CV means even large deltas may not be reliable.
            </div>
        </div>
    </div>

    <!-- Column legend (collapsible) -->
    <div class="card" style="margin-bottom:14px">
        <details>
            <summary>Column Reference Guide</summary>
            <div style="margin-top:10px;overflow-x:auto">
                <table class="legend-table">
                    <thead><tr>
                        <th style="width:110px;background:#f3f7fe">Column</th>
                        <th style="background:#f3f7fe">Description</th>
                    </tr></thead>
                    <tbody>{col_legend_rows}</tbody>
                </table>
            </div>
        </details>
    </div>

    <!-- Functional issues -->
    <div class="card" style="margin-bottom:14px">
        <h2>&#x1F6A8; Failed Tests</h2>
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
        <h2>&#x26A0;&#xFE0F; Top Regressions</h2>
        <div style="overflow-x:auto">
            <table>
                <thead>{thead}</thead>
                <tbody>{regressed_table}</tbody>
            </table>
        </div>
    </div>

    <!-- Top Improvements -->
    <div class="card" style="margin-bottom:14px">
        <h2>&#x2705; Top Improvements</h2>
        <div style="overflow-x:auto">
            <table>
                <thead>{thead}</thead>
                <tbody>{improved_table}</tbody>
            </table>
        </div>
    </div>

    <!-- All rows (collapsible) -->
    <div class="card">
        <details>
            <summary>All Performance Results ({len(all_rows)} series)</summary>
            <div style="margin-top:10px;overflow-x:auto">
                <table>
                    <thead>{thead}</thead>
                    <tbody>{all_table}</tbody>
                </table>
            </div>
        </details>
    </div>

</div>
</body>
</html>
"""


def write_analysis_html(report_path: Path, result: AnalysisResult) -> Path:
        """Write an analysis-focused HTML report next to the text report."""
        html_path = report_path.with_suffix(".html")
        html_path.write_text(render_analysis_html(result), encoding="utf-8")
        return html_path
