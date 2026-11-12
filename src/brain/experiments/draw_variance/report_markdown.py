"""Render the draw-variance payload as a compact markdown report."""

from __future__ import annotations

from typing import Any

__all__ = ["render_markdown"]

_LABELS = {
    "F17a_merge_harm_synthetic": "F17a merge harm (synthetic corpus, tau=0.85)",
    "F17b_merge_harm_relations_v2": "F17b merge harm (relations-v2 corpus, tau=0.85, long)",
    "F21_memory_precision": "F21 memory precision",
    "F23_consolidation_delta": "F23 consolidation delta (destructive@0.75, expected@full)",
}


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _summary_block(statistic_id: str, summary: dict[str, Any]) -> list[str]:
    label = _LABELS.get(statistic_id, statistic_id)
    lines = [f"## {label}", ""]
    status = summary.get("status")
    if status == "BLOCKED":
        lines += [f"**BLOCKED**: {summary.get('reason')}", "",
                   f"Partial draws collected before blocking: {len(summary.get('partial_draws', []))}", ""]
        return lines
    if status == "pending":
        lines += [f"**pending**: {summary.get('n_draws_so_far', 0)}/{summary.get('n_target', '?')} "
                   "draws so far", ""]
        return lines
    if status == "in_progress":
        lines += [f"**in progress**: {summary['n_draws']}/{summary.get('n_target', '?')} draws so far "
                   "(numbers below are provisional, over the draws collected up to now)", ""]
    values = summary["values"]
    lines += [
        f"- frozen value (seed {summary['frozen_seed']}): **{_fmt(summary['frozen_value'])}**",
        f"- {summary['n_draws']} draws: mean={_fmt(summary['mean'])} sd={_fmt(summary['sd'])} "
        f"min={_fmt(summary['min'])} max={_fmt(summary['max'])}",
        f"- frozen seed rank: **{summary['frozen_rank_label']}**, "
        f"z-score **{_fmt(summary['frozen_z_score'])}**",
        f"- verdict: {summary['draw_robust']}",
        "",
        "| seed | value |",
        "|---|---|",
    ]
    for seed, value in zip(summary["seeds"], values):
        marker = " <- FROZEN" if seed == summary["frozen_seed"] else ""
        lines.append(f"| {seed} | {_fmt(value, 6)}{marker} |")
    lines.append("")
    return lines


def _integrity_lines(integrity_report: dict[str, Any]) -> list[str]:
    if "directories_unchanged" not in integrity_report:
        return [f"- {integrity_report.get('status', 'unverified')} "
                "(verified only on the final write, not on mid-run checkpoints)", ""]
    return [
        f"- directories unchanged: **{integrity_report['directories_unchanged']}**",
        f"- synthetic content SHA: `{integrity_report['synthetic_content_sha256']}`",
        f"- relations v1 content SHA: `{integrity_report['relations_v1_content_sha256']}`",
        f"- relations v2 content SHA: `{integrity_report.get('relations_v2_content_sha256')}`",
        "",
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    prereg = payload["preregistration"]
    lines = [
        "# Draw-to-draw variance: F17, F21, F23", "",
        f"Generated {payload['created_at']}. Command: `{payload['command']}`", "",
        f"Pre-registered N={prereg['n_draws']}, seeds={prereg['seeds']}, "
        f"statistics={[s['id'] for s in prereg['statistics']]}.", "",
        "## Corpus integrity",
        *_integrity_lines(payload["corpus_integrity"]),
    ]
    for statistic_id in ("F23_consolidation_delta", "F17a_merge_harm_synthetic",
                         "F17b_merge_harm_relations_v2", "F21_memory_precision"):
        lines += _summary_block(statistic_id, payload["summaries"][statistic_id])
    return "\n".join(lines) + "\n"
