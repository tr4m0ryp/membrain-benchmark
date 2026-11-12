"""Markdown tables for the gate report. Formatting only, no interpretation."""

from __future__ import annotations

import math
from typing import Any, Sequence

from brain.experiments.gate_separation.gate import GateRun

__all__ = [
    "fmt",
    "table",
    "verdict_block",
    "arm_table",
    "curve_table",
    "sweep_table",
    "m2_block",
]


def fmt(value: Any, digits: int = 4) -> str:
    """Fixed-width float, or `n/a` for anything non-finite or non-numeric."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *[_row(row) for row in rows]]


def verdict_block(run: GateRun, payload: dict[str, Any]) -> list[str]:
    """The verdict line, then one comparison table per evaluated configuration."""
    lines = [f"## {payload['verdict_line']}", ""]
    for label in ("selected", "preregistered"):
        verdict = run.verdicts.get(label)
        if verdict is None:
            continue
        lines += [
            f"### Verdict at the {label} configuration: `{verdict.candidate}` "
            f"-> **{verdict.status}**",
            "",
        ]
        lines += table(
            [
                "baseline",
                "candidate",
                "baseline acc",
                "delta (pts)",
                ">= 5 pts",
                "CIs disjoint",
                "passes",
            ],
            [
                [
                    comparison.baseline,
                    fmt(comparison.candidate_accuracy),
                    fmt(comparison.baseline_accuracy),
                    f"{comparison.delta * 100:+.2f}",
                    comparison.meets_margin,
                    comparison.intervals_disjoint,
                    comparison.passes,
                ]
                for comparison in verdict.comparisons
            ],
        )
        lines.append("")
        lines += [f"- reason: {reason}" for reason in verdict.reasons]
        lines += [f"- warning: {warning}" for warning in verdict.warnings]
        lines.append("")
    return lines


def arm_table(run: GateRun) -> list[str]:
    """Per-arm headline metrics, both intervals side by side."""
    rows = []
    for name, result in sorted(run.results.items()):
        seed_ci = result.aggregate["disambiguation"]
        rows.append(
            [
                f"`{name}`",
                fmt(result.disambiguation),
                f"[{fmt(result.bootstrap.lower)}, {fmt(result.bootstrap.upper)}]",
                f"[{fmt(seed_ci.lower)}, {fmt(seed_ci.upper)}]",
                fmt(result.aggregate["recall_at_k"].mean),
                fmt(result.aggregate["mrr"].mean),
                len(result.seeds_run),
                fmt(result.wall_clock, 1),
            ]
        )
    return table(
        [
            "arm",
            "disambiguation",
            "cluster 95% BCa CI",
            "across-seed 95% t CI",
            f"recall@{run.k}",
            "MRR",
            "seeds run",
            "seconds",
        ],
        rows,
    )


def curve_table(run: GateRun) -> list[str]:
    """Accuracy at every cluster size the corpus carries, plus the decay test."""
    sizes = sorted({size for result in run.results.values() for size in result.curve})
    rows = [
        [
            f"`{name}`",
            *[
                fmt(result.curve[size].accuracy, 3) if size in result.curve else "n/a"
                for size in sizes
            ],
            fmt(result.decay.get("drop"), 3),
            result.decay.get("decays"),
        ]
        for name, result in sorted(run.results.items())
    ]
    return table(["arm", *[f"m={size}" for size in sizes], "drop", "decays"], rows)


def sweep_table(payload: dict[str, Any]) -> list[str]:
    """The (mode, s) sweep with the stored-space geometry beside each row."""
    rows = [
        [
            row["mode"],
            f"{row['s']:g}",
            fmt(row["calibrated_tau"], 3),
            fmt(row["disambiguation"]),
            f"[{fmt(row['ci_lower'])}, {fmt(row['ci_upper'])}]",
            "n/a"
            if row["delta_vs_plain_vector_points"] is None
            else f"{row['delta_vs_plain_vector_points']:+.2f}",
            fmt(row["sibling_cosine"], 3),
            fmt(row["contrast"], 3),
            row["separating_writes"],
        ]
        for row in payload["sweep"]
    ]
    return table(
        [
            "mode",
            "s",
            "tau (stored)",
            "disambiguation",
            "cluster 95% CI",
            "delta vs plain (pts)",
            "sibling cos",
            "sib - non-sib",
            "separating writes",
        ],
        rows,
    )


def m2_block(run: GateRun) -> list[str]:
    """The F12 argmax-invariance check, against the control and within each mode."""
    check = (run.diagnostics or {}).get("m2_invariance")
    if not check:
        return []
    size = check["cluster_size"]
    lines = [f"## m = {size} argmax invariance (F12 theorem)", "", f"Claim: {check['claim']}.", ""]
    if not check["measurable"]:
        return lines + [f"Not measurable on this corpus: it holds no cluster of size {size}.", ""]
    lines += [
        f"Reference `{check['reference_arm']}` at m = {size}: "
        f"**{fmt(check['reference_accuracy'])}**.",
        "",
    ]
    lines += table(
        ["arm", f"m={size} accuracy", "delta vs control", "invariant"],
        [
            [
                f"`{row['arm']}`",
                fmt(row["accuracy"]),
                "n/a" if not math.isfinite(row["delta"]) else f"{row['delta'] * 100:+.2f} pts",
                "not measured" if row["invariant"] is None else row["invariant"],
            ]
            for row in check["arms"]
        ],
    )
    lines += [
        "",
        f"Holds for: {', '.join(check['holds_for']) or 'no arm'}.",
        f"Violated by: {', '.join(check['violated_by']) or 'no arm'}.",
        "",
    ]
    within = (run.diagnostics or {}).get("m2_within_mode") or {}
    if within:
        lines += [
            "Referencing each mode to its OWN s = 0 arm separates the two causes, a change "
            "of representation and the push itself.",
            "",
        ]
        lines += table(
            ["mode", f"s = 0 at m={size}", "arms invariant", "arms violated"],
            [
                [
                    mode,
                    fmt(block["reference_accuracy"]),
                    len(block["holds_for"]),
                    len(block["violated_by"]),
                ]
                for mode, block in sorted(within.items())
            ],
        )
        lines.append("")
    return lines
