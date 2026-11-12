"""Render the sweep payload (a plain dict, `SweepResult.to_dict()` plus
`command`/`environment`) as a Markdown report."""

from __future__ import annotations

from typing import Any

__all__ = ["render_markdown"]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NaN"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _ci(block: dict[str, Any]) -> str:
    return f"{_fmt(block['mean'])} [{_fmt(block['lower'])}, {_fmt(block['upper'])}]"


def _metric_table(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| arm | k | recall@k | disambiguation | precision@k | mrr |",
        "|---|---|---|---|---|---|",
    ]
    by_arm = payload["results"]["by_arm"]
    for arm in sorted(by_arm):
        for k in sorted((int(k) for k in by_arm[arm]), key=int):
            cell = by_arm[arm][str(k)]["metrics"]
            lines.append(
                f"| {arm} | {k} | {_ci(cell['recall_at_k'])} | {_ci(cell['disambiguation'])} "
                f"| {_ci(cell['precision_at_k'])} | {_ci(cell['mrr'])} |"
            )
    return lines


def _cluster_size_table(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| arm | k | " + " | ".join(f"m={m}" for m in (2, 3, 4, 6, 8)) + " |",
        "|---|---|" + "---|" * 5,
    ]
    by_arm = payload["results"]["by_arm"]
    for arm in sorted(by_arm):
        for k in sorted((int(k) for k in by_arm[arm]), key=int):
            curve = by_arm[arm][str(k)]["accuracy_by_cluster_size"]
            cells = []
            for m in (2, 3, 4, 6, 8):
                point = curve.get(str(m))
                cells.append(_fmt(point["accuracy"]) if point else "-")
            lines.append(f"| {arm} | {k} | " + " | ".join(cells) + " |")
    return lines


def _precondition_table(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| arm | k | sibling coverage | competitor purity | precondition satisfied |",
        "|---|---|---|---|---|",
    ]
    cov = payload["results"]["precondition_coverage_by_arm"]
    for arm in sorted(cov):
        for k in sorted((int(k) for k in cov[arm]), key=int):
            cell = cov[arm][str(k)]
            lines.append(
                f"| {arm} | {k} | {_ci(cell['sibling_coverage'])} "
                f"| {_ci(cell['competitor_purity'])} | {cell['precondition_satisfiable']} |"
            )
    return lines


def _headroom_table(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| k | A_ctrl (plain_vector) | R@k (plain_vector) | gap available | "
        "H illustrative (bm25, non-canonical) |",
        "|---|---|---|---|---|",
    ]
    for row in payload["results"]["headroom"]:
        lines.append(
            f"| {row['k']} | {_fmt(row['A_ctrl_plain_vector_disambiguation'])} "
            f"| {_fmt(row['R_at_k_plain_vector_recall'])} "
            f"| {_fmt(row['headroom_available_gap'])} "
            f"| {_fmt(row['H_illustrative_non_canonical'])} |"
        )
    return lines


def _answers(payload: dict[str, Any]) -> list[str]:
    lines = ["## Answers"]
    q1 = payload["answers"]["q1_recall_saturation"]
    lines.append("\n**Q1 -- recall@k saturation**\n")
    for arm, row in sorted(q1.items()):
        lines.append(
            f"- `{arm}`: ceiling at k={row['k_grid'][-1]} is {_fmt(row['ceiling_at_max_k'])}; "
            f"plateaus within 1 point at k={row['plateau_k_within_1pt']}. "
            f"Curve: {[round(v, 4) for v in row['recall_by_k']]}"
        )
    q2 = payload["answers"]["q2_disambiguation_monotonicity"]
    lines.append("\n**Q2 -- disambiguation vs k**\n")
    for arm, row in sorted(q2.items()):
        verdict = "non-increasing (as expected: retrieve(k) truncates one fixed order)" if row["non_increasing"] else "VIOLATED monotonicity"
        lines.append(
            f"- `{arm}`: {verdict}. Total drop k={row['k_grid'][0]} to k={row['k_grid'][-1]}: "
            f"{_fmt(row['total_drop'])} points. Curve: {[round(v, 4) for v in row['disambiguation_by_k']]}"
        )
    lines.append(
        "\n**Q3 -- F15/I4 precondition and headroom**\n"
        "\nSee the precondition table (sibling coverage / competitor purity) and the headroom "
        "table above. `H` for the certified operators (`llm_pick`, `verify_veto`) is NOT "
        "computable at any k on this grid: both require a model call, there is no API key in "
        "this environment, and the response cache is keyed on the exact request including k, so "
        "every entry would miss. The bm25 column is an illustrative, non-canonical stand-in "
        "(a different retrieval mechanism, not a reranker over plain_vector's candidates)."
    )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    integrity = payload["corpus_integrity"]
    lines = [
        "# Gate k-sweep",
        "",
        f"Command: `{payload.get('command', '')}`",
        "",
        f"Corpus content SHA256: `{integrity['content_sha256_before_run']}` "
        f"(expected `{integrity['expected_content_sha256']}`, unchanged: {integrity['unchanged']})",
        "",
        "Pre-registered k grid: " + ", ".join(str(k) for k in payload["pre_registration"]["k_grid"]),
        "Out-of-scope arms (not run): "
        + ", ".join(sorted(payload["pre_registration"]["out_of_scope_arms"])),
        "",
        "## Metrics by (arm, k), cluster-BCa 95% CI, B="
        + str(payload["pre_registration"]["bootstrap"]["n_resamples"]),
        "",
        *_metric_table(payload),
        "",
        "## Disambiguation accuracy by cluster size",
        "",
        *_cluster_size_table(payload),
        "",
        "## F15/I4 precondition: bounded, enumerable competitor set",
        "",
        *_precondition_table(payload),
        "",
        "## Headroom H = (A - A_ctrl) / (R@k - A_ctrl)",
        "",
        *_headroom_table(payload),
        "",
        *_answers(payload),
        "",
        f"Wall clock: {_fmt(payload.get('wall_clock_seconds'), 1)}s",
    ]
    return "\n".join(lines)
