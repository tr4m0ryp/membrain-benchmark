"""The contract's four derived expectations, scored against what was measured.

Kept separate from `tables` because these are not a rendering of the run, they are
a judgement on it: the task predicted an inverted U in gamma and in alpha and a
sharp fall in sibling cosine, and stated IN ADVANCE that a cosine drop alone is
not a win. Scoring a prediction after the fact is only worth something if the
prediction is quoted verbatim beside the number, which is what these tables do.

`headroom` answers the question this task actually exists to move: not the raw
delta over plain_vector, but how much of the distance to the PAID arm a free
method covers.
"""

from __future__ import annotations

from typing import Any

from brain.experiments.gate_ncp.tables import select

__all__ = ["headroom", "implied_beta", "expectation_table"]


def headroom(payload: dict) -> str:
    """Fraction of the free-to-paid gap the best NCP cell closed, on llm_pick's queries.

    The framing of this task is that `llm_pick` costs about $0.014 per query, so
    the quantity of interest is not the raw delta over plain_vector but how much
    of the distance to the paid arm a FREE method covers. Measured on the 180
    subsample, because that is where llm_pick answered.
    """
    baselines = payload.get("baselines") or {}
    control, paid = baselines.get("plain_vector"), baselines.get("llm_pick")
    cells = [c for c in payload["cells"] if c["weights"] == "local"]
    if control is None or paid is None or not cells:
        return "n/a (an arm did not run)"
    best = max(cells, key=lambda c: float(c["metrics"]["disambiguation"]))
    floor = float(control["subsample_metrics"]["disambiguation"])
    ceiling = float(paid["subsample_metrics"]["disambiguation"])
    reached = float(best["subsample_metrics"]["disambiguation"])
    if abs(ceiling - floor) < 1e-12:
        return "n/a (no gap to close)"
    return (f"{(reached - floor) / (ceiling - floor):.1%} "
            f"({floor:.4f} free -> {reached:.4f} NCP -> {ceiling:.4f} paid)")

_SURFACE_HEAD = ("| gamma | alpha | disambiguation | 95% cluster BCa | sibling cos | "
                 "non-sib cos | contrast | pairs reordered | recall@k | degenerate |")
_SURFACE_RULE = "|---|---|---|---|---|---|---|---|---|---|"


def implied_beta(cosine: float) -> float:
    """Invert `cos = beta^2 / (beta^2 + (1 - beta)^2)`: the shared mass a cosine implies.

    The task's derived expectation is stated in beta, so the measurement is
    reported in beta as well rather than being left as a cosine the reader has to
    convert. Out-of-range cosines return NaN instead of a complex number.
    """
    value = float(cosine)
    if not 0.0 < value < 1.0:
        return float("nan")
    ratio = (value / (1.0 - value)) ** 0.5
    return ratio / (1.0 + ratio)


def expectation_table(payload: dict[str, Any]) -> list[str]:
    """The four derived expectations from the contract, scored against measurement."""
    verdict, pre = payload["verdict"], payload["preregistration"]
    primary = sorted(select(payload, weights="local", k_neighbours=pre["primary_k"],
                             query_side="mean"), key=lambda c: (float(c["gamma"]),
                                                                float(c["alpha"])))
    if not primary:
        return []
    base = float(verdict["plain_vector_disambiguation"])
    gammas = sorted({float(c["gamma"]) for c in primary})
    alphas = sorted({float(c["alpha"]) for c in primary})
    by_gamma = {g: max(float(c["metrics"]["disambiguation"]) for c in primary
                       if float(c["gamma"]) == g) for g in gammas}
    by_alpha = {a: max(float(c["metrics"]["disambiguation"]) for c in primary
                       if float(c["alpha"]) == a) for a in alphas}
    peak_g = max(by_gamma, key=lambda g: by_gamma[g])
    peak_a = max(by_alpha, key=lambda a: by_alpha[a])
    zero = next(c for c in primary if float(c["gamma"]) == 0.0)
    lines = ["", "## Derived expectations, scored against measurement", "",
             "| # | expectation | measured | holds |", "|---|---|---|---|",
             f"| 1 | sibling cosine falls sharply in gamma (beta 0.684 -> 0.3 predicts 0.155) | "
             f"cos {float(zero['sibling_cosine_dist']):.4f} (beta "
             f"{implied_beta(float(zero['sibling_cosine_dist'])):.3f}) at gamma 0 -> "
             f"{min(float(c['sibling_cosine_dist']) for c in primary):.4f} (beta "
             f"{implied_beta(min(float(c['sibling_cosine_dist']) for c in primary)):.3f}) at "
             f"gamma {max(gammas):g} | "
             f"{'partly' if min(float(c['sibling_cosine_dist']) for c in primary) < 0.5 else 'no'} |",
             f"| 2 | inverted U in gamma, equal to plain_vector at gamma = 0 | peak at gamma "
             f"{peak_g:g}, gamma = 0 gives {float(zero['metrics']['disambiguation']):.4f} against "
             f"plain_vector {base:.4f} | "
             f"{'yes' if 0.0 < peak_g < max(gammas) else 'no'} |",
             f"| 3 | inverted U in alpha, equal to plain_vector at alpha = 1 | peak at alpha "
             f"{peak_a:g} | {'yes' if 0.0 < peak_a < max(alphas) else 'no'} |",
             f"| 4 | a cosine drop alone is NOT success | "
             f"{len(verdict.get('cells_with_cosine_drop_and_no_gain') or [])} primary cells fell "
             f"to or below C-A's 0.704 sibling cosine without beating plain_vector | reported |",
             "", "Best disambiguation per gamma (max over alpha) and per alpha (max over gamma), "
             "primary surface:", "",
             "| gamma | best disambiguation | | alpha | best disambiguation |",
             "|---|---|---|---|---|"]
    for index in range(max(len(gammas), len(alphas))):
        left = (f"| {gammas[index]:g} | {by_gamma[gammas[index]]:.4f} |"
                if index < len(gammas) else "| | |")
        right = (f" | {alphas[index]:g} | {by_alpha[alphas[index]]:.4f} |"
                 if index < len(alphas) else " | | |")
        lines.append(left + right)
    return lines
