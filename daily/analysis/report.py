"""Text report renderer for AnalysisResult.

Produces the ``[ Analysis summary ]`` block that is prepended to the
daily text report.  All formatting lives here so the engine and
persistence layer stay format-agnostic.
"""

from __future__ import annotations

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

        # --- performance aggregate ---
        p = result.performance
        lines.append(
            f"- Performance: compared={p.compared} improved={p.improved} "
            f"same={p.same} regressed={p.regressed}"
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
                    f"(cur={row.current_value:.3f}, base={row.baseline_value:.3f})"
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
