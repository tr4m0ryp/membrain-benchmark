"""Aggregate scored cases into the three write-path rates, with intervals.

The interval is the cluster bootstrap this project already uses for every other
number, called through `brain.eval.bootstrap` rather than reimplemented, so an
incumbent row and an arm row in the paper are comparable by construction. The
resampling unit is the interference cluster, as everywhere else.

A rate over 60 pairs is not a rate over 360 queries, so the report states the
denominator on every row. Where a rate is zero or one in every cluster the
bootstrap interval degenerates to a point, which is reported as measured rather
than widened: it means every draw of clusters gives the same answer, which is
the strongest form the result can take.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from brain.config import SEED, get_rng
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci
from brain.eval.stats import Interval

from brain.baselines.incumbents.outcomes import CaseOutcome, group_by_cluster

__all__ = ["RATES", "rate", "summarise"]

#: The pre-registered quantities, and the direction that would falsify the paper.
RATES = {
    "destroyed": "D, distinct-event destruction rate (lower is better for the system)",
    "relation_recorded": "E, third-branch expressiveness (E > 0 falsifies the paper's claim)",
    "indistinguishable": "I, residual indistinguishability (I < 1 falsifies the paper's claim)",
    "both_attributable": "both episodes still separately addressable",
    "neighbour_offered": "validity: the write path saw the neighbour before deciding",
}


def rate(outcomes: Sequence[CaseOutcome], attribute: str, *, stream: str) -> Interval:
    """Cluster-bootstrapped rate of a boolean outcome attribute."""
    groups = group_by_cluster(outcomes, attribute)
    return cluster_bootstrap_ci(
        groups,
        alpha=0.05,
        n_resamples=DEFAULT_RESAMPLES,
        rng=get_rng(SEED, stream=f"incumbents.{stream}.{attribute}"),
    )


def _slice(outcomes: Iterable[CaseOutcome], regime: str | None) -> list[CaseOutcome]:
    items = list(outcomes)
    return items if regime is None else [item for item in items if item.regime == regime]


def summarise(outcomes: Sequence[CaseOutcome], *, label: str) -> dict[str, Any]:
    """Every rate, pooled and per regime, plus the raw counts behind each."""
    regimes = sorted({outcome.regime for outcome in outcomes})
    out: dict[str, Any] = {"label": label, "n_pairs": len(outcomes), "regimes": regimes}
    for name, meaning in RATES.items():
        block: dict[str, Any] = {"meaning": meaning}
        for key, subset in [("pooled", _slice(outcomes, None))] + [
            (regime, _slice(outcomes, regime)) for regime in regimes
        ]:
            if not subset:
                continue
            hits = sum(1 for outcome in subset if getattr(outcome, name))
            interval = rate(subset, name, stream=f"{label}.{key}")
            block[key] = {
                "n": len(subset),
                "hits": hits,
                "rate": hits / len(subset),
                "interval": interval.to_dict(),
                "n_clusters": len(group_by_cluster(subset, name)),
            }
        out[name] = block
    out["mean_memories_per_pair"] = (
        sum(outcome.n_memories for outcome in outcomes) / len(outcomes) if outcomes else 0.0
    )
    return out
