"""The human-readable gate report: same numbers as the JSON, verdict first."""

from __future__ import annotations

from typing import Any

from brain.experiments.gate_separation.gate import GateRun
from brain.experiments.gate_separation.tables import (
    arm_table,
    curve_table,
    fmt,
    m2_block,
    sweep_table,
    verdict_block,
)

__all__ = ["render_markdown"]


def render_markdown(run: GateRun, payload: dict[str, Any]) -> str:
    """Render one gate run. Reports what was measured, interprets nothing."""
    corpus = payload["corpus"]
    selection = payload["selection"]
    validation = (run.diagnostics or {}).get("harness_validation", {})
    lines: list[str] = [
        "# Gate experiment: H-A minimal falsifying test (Task 008)",
        "",
        f"Run {payload['created_at']} | command `{payload['command']}`",
        "",
        "> A fired kill criterion is a valid, recorded outcome. Nothing was retuned after",
        "> a number was seen.",
        "",
    ]
    lines += verdict_block(run, payload)
    lines += [
        "## Provenance",
        "",
        f"- corpus content SHA-256: `{corpus.get('content_sha256')}`",
        f"- episodes / queries / clusters: {corpus.get('n_episodes')} / "
        f"{corpus.get('n_queries')} / {corpus.get('n_clusters')}",
        f"- cluster-size histogram: {corpus.get('cluster_sizes')}",
        f"- generator: {corpus.get('generator')}, generation seed "
        f"{corpus.get('generation_seed')}, similarity {corpus.get('similarity')}",
        f"- k = {run.k}, seeds = {list(run.seeds)}, resampling unit = interference cluster, "
        "intervals = BCa cluster bootstrap",
        f"- {payload['config']['tau_note']}",
        f"- aggregation = {payload['config']['primary_aggregation']} "
        f"({payload['config']['aggregation_note']})",
        f"- environment: {payload['environment']}",
        "",
        "## Per-arm disambiguation accuracy",
        "",
    ]
    lines += arm_table(run)
    if run.blocked:
        lines += ["", "### BLOCKED arms", ""]
        lines += [f"- `{name}`: {reason}" for name, reason in sorted(run.blocked.items())]
    if payload.get("unavailable_required_arms"):
        lines += [
            "",
            "Required by the criterion but unavailable in this run: "
            f"{payload['unavailable_required_arms']}.",
        ]
    lines += [
        "",
        "### Harness validation",
        "",
        f"- s = 0 (decorrelate) vs `{validation.get('control')}`: "
        f"{fmt(validation.get('accuracy'))} vs {fmt(validation.get('control_accuracy'))}, "
        f"identical = {validation.get('identical')}",
        "",
        "## Accuracy vs cluster size (M-A's central prediction)",
        "",
        "M-A predicts plain vector and full context DECAY in m while separation stays FLAT.",
        "",
    ]
    lines += curve_table(run)
    lines += ["", *m2_block(run)]
    lines += [
        "## Separation strength sweep",
        "",
        f"Selected: `{selection.get('selected_arm')}` at {selection.get('selected_config')}, "
        f"accuracy {fmt(selection.get('selected_accuracy'))} against control "
        f"{fmt(selection.get('control_accuracy'))}.",
        "",
        f"**Selection bias disclosure.** {selection.get('selection_bias')}",
        "",
    ]
    lines += sweep_table(payload)
    sweep_verdicts = payload.get("sweep_verdicts") or {}
    flat = (run.diagnostics or {}).get("flat_curve_check") or {}
    if sweep_verdicts:
        lines += [
            "",
            "### The criterion applied to every configuration, not only the winner",
            "",
            f"- {sweep_verdicts['n_configurations']} configurations evaluated: "
            f"{sweep_verdicts['n_survives']} survives, {sweep_verdicts['n_killed']} killed, "
            f"{sweep_verdicts['n_undecided']} undecided.",
            f"- survivors: {sweep_verdicts['survivors'] or 'none'}.",
        ]
    if flat:
        lines += [
            f"- M-A predicts a FLAT accuracy-vs-cluster-size curve under separation. "
            f"Flat in {flat['n_flat']} of the {flat['n_measurable']} configurations where "
            f"decay is measurable ({flat['flat_arms'] or 'none'}); "
            f"{flat['n_configurations'] - flat['n_measurable']} not measurable.",
        ]
    lines += [
        "",
        "## Cost and runtime",
        "",
        f"- reported LLM spend for this run: {payload['cost_usd_reported']} USD",
        f"- wall clock: {payload['wall_clock_seconds']} s",
        *[
            f"- `{name}`: {block}"
            for name, block in sorted((payload.get("llm_accounting") or {}).items())
        ],
        "",
        "A zero cost with a full cache-hit count means the answers were REPLAYED from",
        "`data/cache/full_context/`, not that the arm is free to produce. The cold pass is",
        "costed in `docs/reproduce-gate-experiment.md`.",
        "",
        "## Reproduce",
        "",
        "```sh",
        f"{payload['command']}",
        "```",
        "",
        "See `docs/reproduce-gate-experiment.md` for the full procedure, including the one-off cost of "
        "warming the full-context response cache.",
        "",
    ]
    return "\n".join(lines)
