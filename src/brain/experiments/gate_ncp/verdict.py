"""The pre-registered stop conditions and the paired deltas, computed from the numbers.

Nothing here takes a threshold from the result it judges. Two questions decide
whether this task produced anything: did varying gamma REORDER sibling pairs (the
C-A test), and does conditioning weights on the LOCAL neighbourhood beat
conditioning them on global corpus statistics once the comparison is PAIRED on the
same clusters. A fired stop condition is the deliverable, not a failure.
"""

from __future__ import annotations

from typing import Any

from brain.experiments.gate_ncp.runner import PRIMARY_K, PRIMARY_QUERY_SIDE, GridRun, paired_delta

__all__ = ["verdict", "comparisons"]


def _primary(run: GridRun) -> list[dict[str, Any]]:
    return [c for c in run.cells if c["weights"] == "local"
            and int(c["k_neighbours"]) == PRIMARY_K and c["query_side"] == PRIMARY_QUERY_SIDE]


def verdict(run: GridRun) -> dict[str, Any]:
    """The pre-registered stop conditions, computed from the numbers."""
    primary = _primary(run)
    control = run.baselines.get("plain_vector", {})
    base = float(control.get("metrics", {}).get("disambiguation", float("nan")))
    spreads = {}
    for alpha in sorted({float(c["alpha"]) for c in primary}):
        values = [float(c["metrics"]["disambiguation"]) for c in primary
                  if float(c["alpha"]) == alpha]
        spreads[f"alpha={alpha:g}"] = round(max(values) - min(values), 6)
    flat = bool(spreads) and max(spreads.values()) <= 1e-4
    reordered = max((int(c["sibling_pairs_reordered"]) for c in run.cells), default=0)
    score = lambda c: float(c["metrics"]["disambiguation"])  # noqa: E731
    local = [c for c in run.cells if c["weights"] == "local"]
    incumbent = [c for c in run.cells if c["weights"] in ("global", "sif")]
    best = max(local or run.cells, key=score)
    rival = max(incumbent, key=score) if incumbent else None
    pre = next((c for c in primary if float(c["gamma"]) == 1.0 and float(c["alpha"]) == 0.5), None)
    cosine_only = [c["arm"] for c in primary
                   if float(c["sibling_cosine"]) <= 0.704 and score(c) <= base + 1e-9]
    return {
        "flat_in_gamma": flat, "gamma_spread_by_alpha": spreads,
        "max_sibling_pairs_reordered": reordered,
        "pair_reordering_observed": reordered > 0,
        "ca_signature": bool(flat and reordered == 0),
        "plain_vector_disambiguation": base,
        "best_cell": best["arm"], "best_disambiguation": score(best),
        "best_delta_points": round((score(best) - base) * 100.0, 4),
        "beats_plain_vector_by_5_points": bool(score(best) - base >= 0.05),
        "best_incumbent_cell": None if rival is None else rival["arm"],
        "best_incumbent_disambiguation": None if rival is None else score(rival),
        # The claim NCP has to earn: LOCAL df beating the best weighted-pooling
        # incumbent (global df or SIF) at its own best cell. The point estimate is
        # a max-over-configurations on both sides and is therefore optimistic for
        # whichever family has more cells, which is NCP; the paired interval in
        # `comparisons` is what settles it.
        "local_minus_best_incumbent_points": (None if rival is None
                                              else round((score(best) - score(rival)) * 100.0, 4)),
        "beats_weighted_pooling_incumbent": None if rival is None else bool(score(best) > score(rival)),
        "preregistered_cell": None if pre is None else {
            "arm": pre["arm"], "disambiguation": score(pre),
            "delta_points": round((score(pre) - base) * 100.0, 4)},
        "cells_with_cosine_drop_and_no_gain": cosine_only,
        "status": ("ca-signature" if (flat and reordered == 0)
                   else "improves" if score(best) - base >= 0.05 else "no-material-gain"),
    }


def _best_of(run: GridRun, kind: str) -> dict[str, Any] | None:
    cells = [c for c in run.cells if c["weights"] == kind]
    return max(cells, key=lambda c: float(c["metrics"]["disambiguation"])) if cells else None


def comparisons(run: GridRun, *, n_resamples: int) -> dict[str, Any]:
    """PAIRED cluster-BCa deltas. Two overlapping marginals settle nothing.

    The claim NCP has to earn is not "weighted pooling helps", which Arora et al
    published: it is that conditioning the weights on the LOCAL neighbourhood
    beats conditioning them on global corpus statistics. That is a paired
    comparison on the same clusters, and it is computed here rather than left to
    a reader eyeballing two intervals.
    """
    delta = lambda a, b: paired_delta(a, b, n_resamples=n_resamples).to_dict()  # noqa: E731
    out: dict[str, Any] = {}
    best = _best_of(run, "local")
    if best is None:
        return out
    left = run.groups[best["arm"]]
    for name in ("plain_vector", "bm25"):
        arm = run.baselines.get(name)
        if arm is not None:
            out[f"best_local_minus_{name}"] = delta(left, arm["clusters"])
    for kind in ("global", "sif"):
        rival = _best_of(run, kind)
        if rival is not None:
            out[f"best_local_minus_best_{kind}"] = {
                "local_arm": best["arm"], "rival_arm": rival["arm"],
                **delta(left, run.groups[rival["arm"]])}
    pick = run.baselines.get("llm_pick")
    if pick is not None:
        out["best_local_minus_llm_pick_on_subsample"] = delta(
            run.subsample_groups[best["arm"]], pick["subsample_clusters"])
    matched: list[dict[str, Any]] = []
    by_key = {(c["weights"], float(c["gamma"]), float(c["alpha"]), c["query_side"]): c
              for c in run.cells if int(c["k_neighbours"]) == PRIMARY_K}
    for key, cell in sorted(by_key.items(), key=lambda kv: kv[0][1:]):
        if key[0] != "local":
            continue
        rival = by_key.get(("global", key[1], key[2], key[3]))
        if rival is None:
            continue
        # The interval goes under its own key: `Interval.to_dict` carries an
        # `alpha` field of its own (the CI level), which would silently overwrite
        # the cell's blending alpha if the two were spread into one dict.
        matched.append({
            "gamma": key[1], "alpha": key[2], "query_side": key[3],
            "local": float(cell["metrics"]["disambiguation"]),
            "global": float(rival["metrics"]["disambiguation"]),
            "delta": delta(run.groups[cell["arm"]], run.groups[rival["arm"]])})
    out["local_minus_global_matched"] = matched
    # Both directions and the average, because counting only the cells that
    # favour us would be the reporting failure this project exists to avoid.
    out["matched_cells_compared"] = len(matched)
    out["matched_cells_where_local_beats_global_conclusively"] = sum(
        1 for m in matched if float(m["delta"]["lower"]) > 0.0)
    out["matched_cells_where_global_beats_local_conclusively"] = sum(
        1 for m in matched if float(m["delta"]["upper"]) < 0.0)
    out["matched_cells_with_positive_point_estimate"] = sum(
        1 for m in matched if float(m["delta"]["mean"]) > 0.0)
    out["mean_matched_delta_points"] = (
        round(100.0 * sum(float(m["delta"]["mean"]) for m in matched) / len(matched), 4)
        if matched else None)
    return out
