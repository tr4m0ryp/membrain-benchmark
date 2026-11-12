"""Two checks the verdict does not make, but a reader of it needs.

* **The m = 2 argmax-invariance theorem (F12).** M-A's symmetric T_s sends a
  colliding pair to (m + (h+s)u)/Z and (m - (h+s)u)/Z with a SHARED normalizer Z,
  so for any cue q the score difference is 2(h+s)<q,u>/Z, whose SIGN is
  sign(<q,u>), independent of s. At m = 2 separation and plain vector therefore
  rank the pair identically BY CONSTRUCTION. If the measured m = 2 accuracies
  differ, either the theorem's premises do not hold for that arm (the sparse arm
  re-thresholds after the push, which breaks the shared normalizer) or the
  implementation has drifted from M-A. Both readings need the number, so it is
  computed rather than asserted.
* **Selectivity of the push.** A separation that rotates a vector away from
  everything, not only from its cluster, buys nothing. The sibling-minus-
  non-sibling cosine contrast in the STORED space is the direct measurement, and
  F12 reports it collapsing from 0.476 to 0.105 across the sweep.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from brain.baselines.base import as_traces
from brain.experiments.gate_separation.specs import ArmSpec

__all__ = ["m2_invariance", "m2_within_mode", "stored_geometry", "geometry_sweep"]

#: Accuracies are means of a bounded indicator over 72 queries at m = 2, so an
#: exact tie is exact. This only absorbs float representation error.
_TOL = 1e-9


def m2_invariance(
    separation: Mapping[str, Any], reference: Any, *, size: int = 2
) -> dict[str, Any]:
    """Does every separation arm reproduce the control exactly at cluster size m?

    `separation` maps arm name to `ArmResult`; `reference` is the plain-vector
    `ArmResult`. Returns the per-arm delta at `size` and whether it is zero.
    """
    point = reference.curve.get(int(size))
    baseline = float("nan") if point is None else float(point.accuracy)
    rows: list[dict[str, Any]] = []
    for name in sorted(separation):
        result = separation[name]
        arm_point = result.curve.get(int(size))
        accuracy = float("nan") if arm_point is None else float(arm_point.accuracy)
        delta = accuracy - baseline
        rows.append(
            {
                "arm": name,
                "accuracy": accuracy,
                "baseline_accuracy": baseline,
                "delta": delta,
                # None, never False, when the corpus holds no cluster of this
                # size: unmeasured is not the same as refuted.
                "invariant": None if not math.isfinite(delta) else bool(abs(delta) <= _TOL),
                "s": float(result.describe.get("params", {}).get("s", float("nan"))),
                "mode": str(result.describe.get("params", {}).get("mode", "")),
            }
        )
    measurable = bool(rows) and math.isfinite(baseline)
    return {
        "cluster_size": int(size),
        "measurable": measurable,
        "reference_arm": reference.arm,
        "reference_accuracy": baseline,
        "claim": (
            "M-A's symmetric T_s shares a normalizer across the pair, so the sign of the "
            "score difference is independent of s: at m = 2 separation and plain vector "
            "must rank identically"
        ),
        "arms": rows,
        "holds_for_all": measurable and all(row["invariant"] for row in rows),
        "holds_for": sorted(row["arm"] for row in rows if row["invariant"] is True),
        "violated_by": sorted(row["arm"] for row in rows if row["invariant"] is False),
        "not_measured": sorted(row["arm"] for row in rows if row["invariant"] is None),
    }


def m2_within_mode(
    separation: Mapping[str, Any], specs: Mapping[str, ArmSpec], *, size: int = 2
) -> dict[str, Any]:
    """The same check with each mode's own s = 0 arm as the reference.

    Against plain vector, an expanded sparse arm differs at m = 2 for two
    reasons at once: the representation changed AND the push was applied.
    Referencing each mode to itself at s = 0 removes the first, so what is left
    answers the actual question, which is whether the PUSH is argmax-invariant
    in that geometry.
    """
    by_mode: dict[str, dict[str, Any]] = {}
    for name, result in separation.items():
        mode = str(specs[name].config.get("mode", ""))
        by_mode.setdefault(mode, {})[name] = result
    out: dict[str, Any] = {}
    for mode in sorted(by_mode):
        arms = by_mode[mode]
        zero = next(
            (n for n in sorted(arms) if float(specs[n].config.get("s", -1.0)) == 0.0), None
        )
        if zero is None:
            continue
        out[mode] = m2_invariance(
            {n: r for n, r in arms.items() if n != zero}, arms[zero], size=size
        )
    return out


def stored_geometry(spec: ArmSpec, corpus: Any, *, seed: int = 0) -> dict[str, Any]:
    """Mean sibling and non-sibling cosine of what this arm actually STORED.

    Builds one arm, indexes the frozen corpus, and reads the vectors back out of
    the store, so this measures the stored representation rather than a
    re-derivation of it.
    """
    episodes = sorted(corpus.episodes, key=lambda e: str(e.trace_id))
    traces = as_traces(episodes)
    arm = spec.build(int(seed))
    try:
        arm.index(traces)
        ids = [trace.id for trace in traces]
        stored = np.asarray(arm.policy.store.get_vectors(ids), dtype=np.float64)
        stats = dict(arm.stats)
    finally:
        arm.close()
    gram = stored @ stored.T
    cluster = np.array([str(episode.cluster_id) for episode in episodes])
    same = cluster[:, None] == cluster[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    sibling = gram[same & upper]
    other = gram[(~same) & upper]
    sibling_mean = float(sibling.mean()) if sibling.size else float("nan")
    other_mean = float(other.mean()) if other.size else float("nan")
    return {
        "arm": spec.name,
        **dict(spec.config),
        "stored_dim": int(stored.shape[1]),
        "sibling_cosine": sibling_mean,
        "non_sibling_cosine": other_mean,
        # The selectivity of the mechanism: a push that moves a vector away from
        # everything shrinks this without buying any disambiguation.
        "contrast": sibling_mean - other_mean,
        "n_sibling_pairs": int(sibling.size),
        "n_non_sibling_pairs": int(other.size),
        "separating_writes": int(stats.get("separating_writes", 0)),
        "collisions": int(stats.get("collisions", 0)),
        "collisions_per_write": float(stats.get("collisions_per_write", 0.0)),
        "probe_saturated_writes": int(stats.get("probe_saturated_writes", 0)),
    }


def geometry_sweep(specs: Sequence[ArmSpec], corpus: Any, *, seed: int = 0) -> list[dict[str, Any]]:
    """`stored_geometry` over the sweep, in spec order."""
    return [stored_geometry(spec, corpus, seed=seed) for spec in specs]
