"""The accuracy-vs-cluster-size curve, which is M-A's central prediction.

M-A predicts a plain vector store decays as `Acc(m) = 1 / (1 + rho*(m - 1))`
while write-time separation stays flat, so the curve is computed for EVERY arm,
with an interval on every point. The frozen corpus is balanced for exactly this
plot: sizes 2/3/4/6/8 each contribute 24 episodes and 72 queries, so no point is
better resourced than another.

The curve also decides the second half of the H-A kill criterion ("or the
accuracy-vs-cluster-size curve is flat for the baseline too"). "Flat" is not
quantified in the pre-registration, so the strictest defensible reading is
implemented and flagged: the baseline curve DECAYS when the drop from the
smallest to the largest cluster size is positive AND its cluster-bootstrap
interval excludes zero. Anything else is flat, and a flat baseline kills H-A
whatever the mechanism scored, because it would mean the corpus never made
interference the binding constraint.

A cluster has exactly one size, so two sizes never share a cluster: the contrast
is stratified, not paired (`stratified_delta_ci`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from brain.corpus.interference import fit_collision_rho
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci, stratified_delta_ci
from brain.eval.metrics import QueryOutcome, indicators_by_cluster_size
from brain.eval.stats import Interval

__all__ = ["CurvePoint", "Strata", "as_strata", "accuracy_curve", "curve_decay", "curve_to_dict"]

#: Cluster size -> cluster id -> that cluster's per-query indicators.
Strata = Mapping[int, Mapping[str, Sequence[float]]]

#: Below this many clusters a resampled interval is coarse and is marked as such.
_MIN_CLUSTERS_FOR_CI = 5


def as_strata(source: Sequence[QueryOutcome] | Strata) -> dict[int, dict[str, list[float]]]:
    """Accept either scored outcomes or ready-made strata.

    The multi-seed runner pools indicators across seeds before the bootstrap (a
    query answered in 3 of 5 seeds contributes 0.6), which no longer fits a
    boolean `QueryOutcome`, so the strata form is the primary input and outcomes
    are converted on the way in.
    """
    if isinstance(source, Mapping):
        return {
            int(size): {str(cid): [float(v) for v in values] for cid, values in sorted(groups.items())}
            for size, groups in sorted(source.items())
        }
    return indicators_by_cluster_size(source)


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """Disambiguation accuracy at one cluster size, with its interval."""

    size: int
    accuracy: float
    ci: Interval
    n_queries: int
    n_clusters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_size": int(self.size),
            "accuracy": float(self.accuracy),
            "ci": self.ci.to_dict(),
            "n_queries": int(self.n_queries),
            "n_clusters": int(self.n_clusters),
            # The frozen corpus balances QUERIES per size, not clusters, so the
            # largest sizes rest on 3 or 4 clusters. A bootstrap over that few
            # units is coarse, and a point that says so cannot be over-read.
            "few_clusters": bool(self.n_clusters < _MIN_CLUSTERS_FOR_CI),
        }


def accuracy_curve(
    source: Sequence[QueryOutcome] | Strata,
    *,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> dict[int, CurvePoint]:
    """Accuracy per cluster size, each with a cluster bootstrap interval.

    Only clusters OF THAT SIZE are resampled for a point, which is the right
    reference set: the question at m = 8 is how an 8-episode cluster behaves, not
    how the corpus behaves.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    curve: dict[int, CurvePoint] = {}
    for size, groups in as_strata(source).items():
        interval = cluster_bootstrap_ci(
            groups, alpha=alpha, n_resamples=n_resamples, rng=generator
        )
        curve[size] = CurvePoint(
            size=size,
            accuracy=interval.mean,
            ci=interval,
            n_queries=sum(len(values) for values in groups.values()),
            n_clusters=len(groups),
        )
    return curve


def curve_decay(
    source: Sequence[QueryOutcome] | Strata,
    *,
    alpha: float = 0.05,
    min_drop: float = 0.0,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Does this arm's curve decay with cluster size, or is it flat?

    Mechanical rule (no human judgement in the loop): decays iff
    `acc(m_min) - acc(m_max) >= min_drop` and the lower endpoint of the
    stratified bootstrap interval on that drop is above zero.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    strata = as_strata(source)
    sizes = sorted(strata)
    curve = accuracy_curve(strata, alpha=alpha, n_resamples=n_resamples, rng=generator)
    report: dict[str, Any] = {
        "rule": (
            "decays iff acc(min m) - acc(max m) >= min_drop and the stratified "
            "cluster-bootstrap lower bound on that drop is > 0"
        ),
        "min_drop": float(min_drop),
        "sizes": sizes,
        "accuracy_by_cluster_size": {str(size): curve[size].accuracy for size in sizes},
        "m_a_fit": fit_collision_rho(
            {str(size): curve[size].accuracy for size in sizes}
        ),
    }
    if len(sizes) < 2:
        report.update(
            {"drop": math.nan, "drop_ci": None, "decays": False, "flat": True,
             "note": "fewer than two cluster sizes present, decay is not measurable"}
        )
        return report
    smallest, largest = sizes[0], sizes[-1]
    drop = stratified_delta_ci(
        strata[smallest], strata[largest], alpha=alpha, n_resamples=n_resamples, rng=generator
    )
    decays = bool(
        math.isfinite(drop.mean)
        and drop.mean >= float(min_drop)
        and drop.defined
        and drop.lower > 0.0
    )
    report.update(
        {
            "smallest_size": smallest,
            "largest_size": largest,
            "drop": drop.mean,
            "drop_ci": drop.to_dict(),
            "decays": decays,
            "flat": not decays,
        }
    )
    return report


def curve_to_dict(curve: dict[int, CurvePoint]) -> dict[str, Any]:
    """Serializable form, keyed by cluster size as a string (JSON has no int keys)."""
    return {str(size): curve[size].to_dict() for size in sorted(curve)}
