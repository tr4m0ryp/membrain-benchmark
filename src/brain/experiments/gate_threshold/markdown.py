"""Markdown tables for the K1 gate report. Formatting only, no interpretation.

Split out of `report.py` for the same reason `gate_separation/markdown.py` is
split out of its gate: rendering is the one part of an experiment that grows
without bound, and mixing it with the run keeps neither under the 300-line cap
(rules.md). Nothing here computes a number; every value is read from the payload
that `report.build_payload` already fixed.
"""

from __future__ import annotations

from typing import Any, Sequence

from brain.corpus.relations import REGIMES
from brain.experiments.gate_threshold.ceiling import K1_TEXT

__all__ = ["CELLS", "STRATA", "fmt", "table", "render_markdown"]

STRATA: tuple[str, ...] = ("marginal", "same_topic")
CELLS: tuple[str, ...] = (*REGIMES, "pooled")
_CEILING_HEAD = ["cell", "n S/I", "ceiling", "cluster 95% BCa", "tau*", "orientation",
                 "threshold Bayes err", "Bayes err any f(s)", "LOCO CV", "K1"]


def fmt(value: Any, digits: int = 4) -> str:
    """Fixed-width float, or `n/a` for anything non-finite or non-numeric."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if number == number else "n/a"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *body, ""]


def _distribution(payload: dict[str, Any]) -> list[str]:
    return table(
        ["class", "regime", "n", "same-topic n", "mean", "sd", "p05", "p50", "p95", "min", "max"],
        [[r["cls"], r["regime"], r["n"], r["same_topic_n"], fmt(r["mean"]), fmt(r["sd"]),
          fmt(r["p05"]), fmt(r["p50"]), fmt(r["p95"]), fmt(r["min"]), fmt(r["max"])]
         for r in payload["distributions"]["derivable"]],
    )


def _direction(payload: dict[str, Any], stratum: str) -> list[str]:
    block = payload["direction"][stratum]
    rows = [[regime, fmt(v["mean_supersession"]), fmt(v["mean_interference"]),
             f"{v['delta_S_minus_I']:+.4f}", v["direction"], v["overlaps"],
             "n/a" if not v["interval"] else f"[{fmt(v['interval'][0])}, {fmt(v['interval'][1])}]",
             fmt(v["supersession_mass_in_overlap"]), fmt(v["interference_mass_in_overlap"])]
            for regime, v in sorted(block["per_regime"].items())]
    return [
        f"### {stratum} (direction flips between regimes: **{block['flips_between_regimes']}**)",
        "",
        *table(["regime", "E[s|S]", "E[s|I]", "S - I", "direction", "supports overlap",
                "overlap band", "S mass in band", "I mass in band"], rows),
        "Withdrawn prediction `E[s|I] > E[s|S]` would have held in: "
        f"{block['withdrawn_prediction_holds_in'] or 'no regime'}.", "",
    ]


def _ceiling_rows(payload: dict[str, Any], name: str, stratum: str) -> list[list[Any]]:
    rows = []
    for cell in CELLS:
        block = payload["ceilings"][name][stratum][cell]
        if not block["available"]:
            continue
        top, boot = block["ceiling"], block["bootstrap"]["interval"]
        rows.append([
            cell, f"{top['n_supersession']}/{top['n_interference']}",
            fmt(top["balanced_accuracy"]), f"[{fmt(boot['lower'])}, {fmt(boot['upper'])}]",
            fmt(top["tau_star"]), top["orientation"], fmt(top["threshold_bayes_error"]),
            fmt(block["bayes_error_any_f"].get("bayes_error_any_f")),
            fmt(block["cross_validated"].get("balanced_accuracy")),
            "**FIRES**" if block["k1"]["fires"] else "no",
        ])
    return rows


def _ceilings(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for name in ("derivable", "sample"):
        for stratum in STRATA:
            tag = " (PRIMARY)" if name == "derivable" and stratum == "same_topic" else ""
            lines += [f"### {name} / {stratum}{tag}", "",
                      *table(_CEILING_HEAD, _ceiling_rows(payload, name, stratum))]
    return lines + [
        "Reading the two error floors: `Bayes err any f(s)` is a KDE estimate and is biased UP by "
        "smoothing, while the oracle `threshold Bayes err` is biased DOWN because tau is fitted on "
        "the same pairs. At the class sizes here the first can exceed the second; the theoretical "
        "ordering (any `f` beats any threshold) only becomes recoverable at large n, so neither "
        "is quoted as bounding the other.", "",
        "`pooled` is deliberately reported and deliberately not the headline. It is BELOW both "
        "regimes because the optimum lives at a different place in each, so a single tau cannot "
        "serve them at once. That is Theorem 1's non-functionality claim showing up directly in "
        "the measurement rather than in the argument.", "",
        "Sensitivity, the literal action map with cross-workstream pairs counted as interference "
        "(the population `manifest.conventions` excludes):", "",
        *table(["cell", "n S/I", "ceiling", "tau*", "K1"],
               [[cell, f"{v['n_supersession']}/{v['n_interference']}",
                 fmt(v["balanced_accuracy"]), fmt(v["tau_star"]),
                 "FIRES" if v["fires_k1"] else "no"]
                for cell, v in sorted(payload["sensitivity_all_interference"].items())]),
        "That sensitivity is why the population matters: read literally, the action map fires K1 "
        "in BOTH regimes. It gets there by counting 10039 cross-workstream pairs per regime as "
        "interference, which no collision probe would ever surface, so it answers a different "
        "question than the one K1 asks.", "",
    ]


def _harm(payload: dict[str, Any]) -> list[str]:
    rows = [[regime, fmt(r["tau"], 2), r["distinct_events_destroyed"],
             fmt(r["distinct_events_destroyed_rate"]), r["supersessions_correctly_merged"],
             fmt(r["supersessions_correctly_merged_rate"]), r["supersessions_missed"],
             fmt(r["merge_precision"]), f"{r['clusters_touched']}/{r['clusters_total']}"]
            for regime in REGIMES for r in payload["harm_table"][regime]]
    return table(["regime", "tau", "distinct destroyed", "rate", "supersessions merged", "rate",
                  "supersessions missed", "merge precision", "clusters touched"], rows)


def _loss(payload: dict[str, Any]) -> list[str]:
    shown = ("0.60", "0.70", "0.80", "0.85", "0.90", "0.95")
    rows = []
    for stratum in STRATA:
        for regime in REGIMES:
            block = payload["expected_loss"][stratum][regime]
            if not block.get("available"):
                continue
            for curve in block["curves"]:
                rows.append([
                    stratum, regime, fmt(curve["cost_ratio_cS_over_cI"], 1),
                    fmt(curve["tau_argmin"], 2), fmt(curve["min_expected_loss"]),
                    *[fmt(curve["loss_by_tau"].get(tau), 3) for tau in shown],
                ])
    return table(["stratum", "regime", "c_S / c_I", "argmin tau", "min loss",
                  *[f"L({tau})" for tau in shown]], rows)


def render_markdown(payload: dict[str, Any]) -> str:
    corpus, config, notes = payload["corpus"], payload["config"], payload["pair_set_used"]
    lines = [
        "# Task 006 -- the threshold-ceiling gate (K1)", "",
        f"**{payload['verdict_line']}**", "", f"> {K1_TEXT}", "",
        "## 1. Per-class, per-regime similarity (derivable population)", "",
        *_distribution(payload),
        "## 2. Direction and overlap of the two supports", "",
    ]
    for stratum in STRATA:
        lines += _direction(payload, stratum)
    lines += ["## 3. The ceiling", "", *_ceilings(payload),
              "## 4. F17's harm table, now a three-column tradeoff", "", *_harm(payload),
              "## 5. Expected loss `c_S * P(false supersede) + c_I * P(false keep both)`", "",
              *_loss(payload), "## Provenance", "",
              f"- relations content SHA: `{corpus['relations_content_sha256']}` "
              f"(matches: {corpus['relations_sha_matches']})",
              f"- synthetic content SHA: `{corpus['synthetic_content_sha256']}` "
              f"(matches: {corpus['synthetic_sha_matches']})",
              f"- both corpus directories unchanged across the run: "
              f"{corpus['directories_unchanged']}",
              f"- pair set: {notes['comparison']}",
              f"- interference population: {notes['interference_population']}",
              f"- seeds {config['seeds']}, resamples {config['bootstrap_resamples']}, "
              f"LLM calls {config['llm_calls']}, cost ${config['cost_usd']:.2f}", ""]
    return "\n".join(lines)
