"""Text report renderer for AnalysisResult.

Produces the ``[ Analysis summary ]`` block that is prepended to the
daily text report.  All formatting lives here so the engine and
persistence layer stay format-agnostic.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from .types import AnalysisResult


# Longest edge of the inline preview. Outlook blocks remote images, so the
# thumbnail must be embedded while the full-size file stays on the relay.
THUMBNAIL_PX = 240


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


def _series_counts(summary: dict | None) -> tuple[int, int, int]:
    """Return ``(skipped, success, failed)`` series counts for the run.

    Counted per ``expected_series`` rather than per test function, matching
    ``common.delivery.mail_title_suffix``: one test function can stand for
    several benchmark series.
    """
    skipped = success = failed = 0
    for test in (summary or {}).get("tests", []):
        expected = int((test.get("metrics") or {}).get("expected_series") or 0)
        outcome = test.get("outcome")
        if outcome == "skipped":
            skipped += expected
        elif outcome == "passed":
            success += expected
        elif outcome in ("failed", "error"):
            failed += expected
    return skipped, success, failed


def _thumbnail_data_uri(path: Path) -> str | None:
    """Return a downscaled JPEG data URI, or None when it can't be produced."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=80)
    except (OSError, ValueError):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_cell(path: Path | None, url: str | None, caption: str) -> str:
    """One thumbnail with its caption, or a placeholder when nothing is available."""
    src = _thumbnail_data_uri(path) if path else None
    src = src or url
    if not src:
        return (
            "<div style='display:inline-block;text-align:center;margin:0 8px'>"
            "<div style='width:150px;height:150px;line-height:150px;border:1px dashed #d9dee7;"
            "border-radius:8px;color:#9ca3af;font-size:12px'>none</div>"
            f"<div style='font-size:11px;color:#6b7280;margin-top:4px'>{html.escape(caption)}</div>"
            "</div>"
        )
    img = (
        f"<img src='{html.escape(src, quote=True)}' "
        "style='max-width:150px;border:1px solid #d9dee7;border-radius:8px' />"
    )
    if url:
        img = (f"<a href='{html.escape(url, quote=True)}' "
               f"title='Open full-size image'>{img}</a>")
    return (
        "<div style='display:inline-block;text-align:center;margin:0 8px'>"
        f"{img}"
        f"<div style='font-size:11px;color:#6b7280;margin-top:4px'>{html.escape(caption)}</div>"
        "</div>"
    )


def _render_image_gallery(summary: dict | None,
                          image_assets: dict[str, dict] | None = None,
                          baseline_stamp: str | None = None) -> str:
    """Render each generated image next to the baseline run's image for the
    same slot, as inline thumbnails linked to their full-size copies.

    ``image_assets`` maps a test's ``image_path`` to ``url`` (published
    full-size copy), ``baseline_path`` (local baseline image) and
    ``baseline_url``.
    """
    if not summary:
        return ""

    image_assets = image_assets or {}
    published = any(a.get("url") for a in image_assets.values())
    cards: list[str] = []
    for test in summary.get("tests", []):
        m = test.get("metrics", {})
        if m.get("test_type") != "image_generation" or test.get("outcome") != "passed":
            continue
        for d in m.get("data", []):
            image_path = d.get("image_path")
            if not image_path:
                continue
            asset = image_assets.get(image_path) or {}
            current = _image_cell(Path(image_path), asset.get("url"), "Current")
            baseline = _image_cell(asset.get("baseline_path"),
                                   asset.get("baseline_url"),
                                   f"Baseline {baseline_stamp}" if baseline_stamp else "Baseline")
            label = f"{m.get('model', '')} / {m.get('precision', '')}"
            if d.get("input_token_size") is not None:
                label += f" (in={d['input_token_size']})"
            cards.append(
                "<div style='display:inline-block;vertical-align:top;text-align:center;"
                "border:1px solid #e5e7eb;border-radius:10px;padding:10px;margin:0 14px 14px 0'>"
                f"<div style='font-size:12px;font-weight:700;margin-bottom:6px'>{html.escape(label)}</div>"
                f"{current}{baseline}"
                "</div>"
            )

    if not cards:
        return ""

    hint = ("Each pair shows this run's image next to the baseline run's image for the "
            "same model/precision slot.")
    if published:
        hint += " Click a thumbnail to open the full-size file."
    return f"""
    <div class="card" style="margin-bottom:14px">
        <h2>Generated Images</h2>
        <div style="font-size:12px;color:#6b7280;margin-bottom:8px">
            {hint}
        </div>
        <div>{"".join(cards)}</div>
    </div>
    """


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


def render_analysis_html(result: AnalysisResult, summary: dict | None = None,
                         image_assets: dict[str, dict] | None = None,
                         baseline_meta: dict | None = None) -> str:
    """Return a standalone HTML report for analysis-focused review.

    ``summary`` is the normalised daily summary dict (same shape as
    ``daily.*.summary.json``); when given, generated images from
    image_generation tests are shown as thumbnails. ``image_assets`` carries
    the published URL and the matching baseline image for each of them.
    ``baseline_meta`` is the baseline run's ``meta`` block, used to flag
    environment differences in the Run Summary card.
    """
    from datetime import datetime as _dt  # noqa: PLC0415

    image_gallery = _render_image_gallery(summary, image_assets,
                                          result.baseline.stamp)

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
    # Top table counts one benchmark series as one unit, including the series
    # skipped tests would have produced.
    series_skipped, series_success, series_failed = _series_counts(summary)
    series_total = series_skipped + series_success + series_failed

    baseline_text = "not found"
    if result.baseline.status == "found":
        baseline_text = f"{result.baseline.stamp or ''} / {result.baseline.ov_version or 'unknown'}"

    summary_rows, changed_fields, has_baseline_meta = _render_run_summary(
        result, summary, baseline_meta)
    # Without the baseline column there is nowhere else the baseline is named.
    summary_head = ""
    summary_note = ""
    baseline_row = f'<tr><td class="k">Baseline</td><td>{html.escape(baseline_text)}</td></tr>'
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
        delta_style = f"text-align:right;{delta_s};white-space:nowrap" if delta_s else "text-align:right;white-space:nowrap"
        cv_style = f"text-align:right;{cv_s};white-space:nowrap" if cv_s else "text-align:right;white-space:nowrap"
        return (
            "<tr>\n"
            f"<td>{html.escape(k.model)}</td>\n"
            f"<td>{html.escape(k.precision)}</td>\n"
            f"<td style='white-space:nowrap'>{k.in_token}&nbsp;/&nbsp;{k.out_token}</td>\n"
            f"<td>{html.escape(k.exec_mode)}</td>\n"
            f"<td class='num' style='text-align:right;white-space:nowrap'>{_fmt_num(row.current_value, unit)}</td>\n"
            f"<td class='num' style='text-align:right;white-space:nowrap'>{_fmt_num(row.baseline_value, unit)}</td>\n"
            f"<td class='num' style='{delta_style}'>{_fmt_pct(row.improvement_pct)}{fluct}</td>\n"
            f"<td class='num' style='text-align:right'>{row.history_count}</td>\n"
            f"<td class='num' style='text-align:right;white-space:nowrap'>{_fmt_num(row.history_sigma, unit)}</td>\n"
            f"<td class='num' style='{cv_style}'>{_fmt_cv(row.history_cv)}</td>\n"
            f"<td style='font-size:11px;color:#6b7280'>{html.escape(row.reference_source)}</td>\n"
            "</tr>"
        )

    improved_table  = "\n".join(_row_html(r) for r in improved_rows)  or "<tr><td colspan='11' style='color:#6b7280;text-align:center'>No improved rows</td></tr>"
    regressed_table = "\n".join(_row_html(r) for r in regressed_rows) or "<tr><td colspan='11' style='color:#6b7280;text-align:center'>No regressed rows</td></tr>"
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
        <span class="badge" style="background:{badge[1]};font-size:16px;padding:6px 18px">{badge[0]}</span>
    </div>

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

    {image_gallery}

    <!-- Reference material: kept last, it is only needed while learning the report -->
    <div class="card" style="margin-bottom:14px">
        <h2>Analysis Methodology</h2>
        <div style="font-size:13px;line-height:1.65;color:#374151">
            <b>Reference</b> = mean of the best <b>top-5</b> runs from a <b>10-run history window</b> (same machine · model · precision · mode).<br>
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
