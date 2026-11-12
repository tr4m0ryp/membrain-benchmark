"""Render the measured tables of the incumbent report from the result JSON.

    python -m brain.baselines.incumbents.report results/incumbents-<stamp>.json

The tables are generated rather than typed so a number in the write-up cannot
drift from the number in the result file. Prose around them is written by hand;
this module owns only the rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["render", "main"]


def _interval(block: dict[str, Any]) -> str:
    interval = block.get("interval") or {}
    mean, lower, upper = interval.get("mean"), interval.get("lower"), interval.get("upper")
    if mean is None:
        return "n/a"
    return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"


def _rate_rows(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| Configuration | pairs | D destroyed | E relation recorded | I indistinguishable | neighbour offered |",
        "|---|---|---|---|---|---|",
    ]
    for record in payload.get("configurations", []):
        summary = record.get("summary")
        if record.get("status") != "RAN" or not summary:
            continue
        cells = [record["configuration"], str(summary["n_pairs"])]
        for name in ("destroyed", "relation_recorded", "indistinguishable", "neighbour_offered"):
            block = summary.get(name, {}).get("pooled", {})
            cells.append(f"{block.get('hits', 0)}/{block.get('n', 0)} = {_interval(block)}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _regime_rows(payload: dict[str, Any]) -> list[str]:
    lines = ["| Configuration | regime | pairs | D destroyed | I indistinguishable |", "|---|---|---|---|---|"]
    for record in payload.get("configurations", []):
        summary = record.get("summary")
        if record.get("status") != "RAN" or not summary:
            continue
        for regime in summary.get("regimes", []):
            destroyed = summary["destroyed"].get(regime, {})
            same = summary["indistinguishable"].get(regime, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        record["configuration"],
                        regime,
                        str(destroyed.get("n", 0)),
                        _interval(destroyed),
                        _interval(same),
                    ]
                )
                + " |"
            )
    return lines


def _blocked_rows(payload: dict[str, Any]) -> list[str]:
    lines = ["| Configuration | blocker |", "|---|---|"]
    any_blocked = False
    for record in payload.get("configurations", []):
        if record.get("status") == "BLOCKED":
            any_blocked = True
            lines.append(f"| {record['configuration']} | {record.get('blocked_reason', '')[:300]} |")
    return lines if any_blocked else ["*No configuration was blocked.*"]


def _exhaustive_rows(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for record in payload.get("configurations", []):
        name = record.get("configuration", "")
        if name.startswith("graphiti") and record.get("status") == "RAN":
            lines += [
                "",
                f"**Graphiti**, {record['n_responses_executed']} executions covering the whole "
                "response space of `EdgeDuplicate` under two temporal shapes.",
                "",
                f"- response model fields: `{'`, `'.join(record['response_model_fields'])}`",
                f"- keep-both response identical to the no-candidate control: "
                f"**{record['keep_both_identical_to_control']}**",
                f"- responses identical to that control: "
                f"**{record['n_identical_to_control']} of {record['n_responses_executed']}**",
                f"- invented relation fields surviving the parser: "
                f"**{record['invented_fields_surviving']}**",
                f"- `EntityEdge` fields: `{'`, `'.join(record['entity_edge_fields'])}`",
            ]
        if name.startswith("toki") and record.get("status") == "RAN":
            lines += [
                "",
                f"**TOKI**, {record['n_executions']} executions covering "
                f"{len(record['operators'])} operators by 3 arrangements.",
                "",
                f"- contradiction predicate: {record['contradiction_predicate']}",
                f"- a same-key contemporaneous pair is destroyed under every operator: "
                f"**{record['destroyed_under_every_operator']}**",
                f"- an audit row is written on every resolution: "
                f"**{record['audit_row_written_on_every_resolution']}**",
                f"- keeping both is structurally identical to two unrelated writes: "
                f"**{record['keep_both_indistinguishable_from_unrelated']}**",
            ]
    return lines or ["*No exhaustive probe ran.*"]


def _published_rows(payload: dict[str, Any]) -> list[str]:
    table = payload.get("published_numbers", {})
    lines = [
        "| System | Benchmark | Metric | Published value | Model | Source | Caveat |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table.get("rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["system"],
                    row["benchmark"],
                    row["metric"],
                    row["value"],
                    row["model"],
                    row["source"],
                    row.get("caveat", "").replace("\n", " "),
                ]
            )
            + " |"
        )
    return lines


def render(payload: dict[str, Any]) -> str:
    corpus = payload.get("corpus", {})
    parts = [
        "## 2. What ran, and what it measured",
        "",
        f"Relational corpus `{corpus.get('relations_content_sha256', '')[:16]}`, "
        f"{corpus.get('n_interference_pairs')} INTERFERENCE pairs, each with a matched "
        "cross-cluster control.",
        "",
        *_rate_rows(payload),
        "",
        "### Per regime",
        "",
        *_regime_rows(payload),
        "",
        "### Exhaustive probes",
        *_exhaustive_rows(payload),
        "",
        "### Blocked",
        "",
        *_blocked_rows(payload),
        "",
        "## Published numbers, NOT RUN BY US",
        "",
        payload.get("published_numbers", {}).get("warning", ""),
        "",
        *_published_rows(payload),
        "",
    ]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        print("usage: python -m brain.baselines.incumbents.report <result.json>", file=sys.stderr)
        return 2
    payload = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    print(render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
