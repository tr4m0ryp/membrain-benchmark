"""The markdown that travels with the result JSON.

Written so the verdict cannot be read without the design that produced it: the
pre-registration comes first, the primary cell second, and the full grid after,
including the scope and the resampling unit that were NOT primary.
"""

from __future__ import annotations

from typing import Any, Mapping

from brain.experiments.convention_external.design import CONVENTIONS, MODES, PRIMARY
from brain.experiments.convention_external.runner import METRICS

__all__ = ["render"]

_VERDICT_GLOSS = {
    "STRONG_FLIP": "the sign flips AND both intervals exclude zero",
    "SIGN_FLIPS": "the sign flips; at least one interval covers zero",
    "IMMUNE": "the three conventions coincide on almost every query; the lever cannot act",
    "NO_FLIP": "the conventions move the number but never past zero",
}


def _interval(cell: Mapping[str, Any] | None) -> str:
    if not cell:
        return "-"
    return f"{cell['mean']:+.3f} [{cell['lower']:+.3f}, {cell['upper']:+.3f}]"


def _grid(results: Mapping[str, Any], taus: list[float], metric: str) -> list[str]:
    lines = ["| tau | " + " | ".join(MODES) + " |",
             "|---|" + "---|" * len(MODES)]
    for tau in taus:
        cells = [results.get(f"{mode}@{tau:g}", {}).get(metric, {}).get("delta_vs_before")
                 for mode in MODES]
        lines.append(f"| {tau:g} | " + " | ".join(_interval(c) for c in cells) + " |")
    return lines


def render(report: Mapping[str, Any]) -> str:
    corpus = report["corpus"]
    verdict = report["verdict"]
    config = report["config"]
    taus = [float(t) for t in config["thresholds"]]
    scope, unit = PRIMARY["scope"], PRIMARY["resampling_unit"]
    primary = report["results"][scope][unit]
    out: list[str] = [
        "# Does the crediting convention decide the sign on LoCoMo?",
        "",
        "Pre-registered replication of the F23 crediting-convention finding on a public",
        "benchmark. The machinery is unchanged; only the data is new.",
        "",
        "## Pre-registration (fixed before any number was computed)",
        "",
        f"- **Benchmark**: {corpus['benchmark']}, commit `{corpus['commit']}`, "
        f"file sha256 `{corpus['content_sha256'][:16]}...`",
        f"- **Shape**: {corpus['n_conversations']} conversations, {corpus['n_episodes']} episodes "
        f"(one per dialogue turn), {corpus['n_queries']} queries, {corpus['n_clusters']} sessions",
        "- **Subset**: every conversation, every usable annotated question. No subsampling.",
        f"- **Thresholds**: {', '.join(f'{t:g}' for t in taus)}, complete linkage, one pass per cell",
        f"- **Conventions**: {', '.join(CONVENTIONS)} (`credited` generous, "
        "`expected` = 1/(1+a), `strict` = nothing once a >= 1)",
        f"- **Primary cell**: {verdict['cell']}, depth @{PRIMARY['depth']}, "
        f"{scope}-scoped distractors, resampled by {unit}",
        f"- **Bootstrap**: paired BCa, {config['bootstrap_resamples']} resamples, "
        f"cluster-level only",
        "",
        "## Verdict",
        "",
        f"**{verdict['outcome']}** -- {_VERDICT_GLOSS.get(str(verdict['outcome']), '')}",
        "",
        "| convention | delta vs before (paired cluster BCa, 95%) |",
        "|---|---|",
        f"| generous (`credited`) | {_interval(verdict['credited'])} |",
        f"| expected (`1/(1+a)`) | {_interval(verdict['expected'])} |",
        f"| strict | {_interval(verdict['strict'])} |",
        "",
        f"Fraction of queries on which all three conventions coincide: "
        f"{verdict['conventions_coincide_fraction']:.4f}.",
        "",
        "## Diagnostics at the primary cell",
        "",
        "| scope | queries | all three agree | winning record absorbed >= 1 distractor | absorbed histogram |",
        "|---|---|---|---|---|",
    ]
    for scope_name, block in report["diagnostics"].items():
        out.append(
            f"| {scope_name} | {block['n_scored']} | "
            f"{block['conventions_coincide_fraction']:.4f} | "
            f"{block['fraction_absorbing_at_least_one']:.4f} | "
            f"`{block['absorbed_histogram_at_primary']}` |"
        )
    out += ["", "## Full grid, primary scope and unit", ""]
    for metric in METRICS:
        out += [f"### delta {metric}", ""] + _grid(primary, taus, metric) + [""]
    out += ["## Sensitivity: every scope and resampling unit, primary cell only", "",
            "| scope | unit | " + " | ".join(f"{c}@5" for c in CONVENTIONS) + " |",
            "|---|---|" + "---|" * len(CONVENTIONS)]
    for scope_name, units in report["results"].items():
        for unit_name, block in units.items():
            cells = [block.get(verdict["cell"], {}).get(f"{c}@5", {}).get("delta_vs_before")
                     for c in CONVENTIONS]
            out.append(f"| {scope_name} | {unit_name} | "
                       + " | ".join(_interval(c) for c in cells) + " |")
    spend = report.get("spend", {})
    out += ["", "## Cost", "",
            f"- live model calls: {spend.get('api_calls', 0)}",
            f"- cache replays: {spend.get('cache_hits', 0)}",
            f"- spend: {float(spend.get('cost_usd', 0.0)):.4f} USD",
            f"- wall clock: {report.get('wall_clock_seconds', 0.0):.1f} s", ""]
    return "\n".join(out)
