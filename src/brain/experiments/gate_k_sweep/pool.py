"""Per-cluster pooling for the three metrics `eval.runner.run_arm` does not
already bootstrap at the cluster level.

`run_arm` (src/brain/eval/runner.py) returns a cluster-BCa interval for
disambiguation accuracy only (`ArmResult.bootstrap`, built from its private
`_pool_clusters`); recall@k, precision@k and MRR get a Student-t interval
*across seeds* instead (`ArmResult.aggregate`), which answers a different
question (how much the arm's own randomness moves the number, not how much
another draw of interference clusters would). The task's hard constraint is
cluster-level BCa for every metric, so this module re-derives the same
per-cluster grouping `_pool_clusters` uses -- average a query's indicator
across the 5 seeds first, then group by cluster -- for the other three
metrics, using only the public `SeedRun.outcomes` already on the `ArmResult`.

Disambiguation is deliberately NOT recomputed here: `ArmResult.clusters` is
already exactly this grouping for that metric, so reusing it (rather than a
second implementation that could silently drift) is direct.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from brain.eval.metrics import QueryOutcome

__all__ = ["POOLED_METRICS", "indicator", "pool_metric_clusters"]

#: Metrics this module knows how to pool. Disambiguation is out of scope here
#: (see module docstring): it comes from `ArmResult.clusters` directly.
POOLED_METRICS: tuple[str, ...] = ("recall_at_k", "precision_at_k", "mrr")


def indicator(outcome: QueryOutcome, metric: str, k: int) -> float:
    """The per-query {0,1} or real-valued indicator a cluster mean is built from.

    Every query in the frozen corpus has >= 1 sibling (metrics.py's own
    invariant: "on the frozen corpus every cluster has m >= 2, so all 360 are
    eligible"), so unlike disambiguation, recall/precision/MRR need no
    eligibility filter -- there is nothing to filter here.
    """
    if metric == "recall_at_k":
        return float(outcome.retrieved)
    if metric == "precision_at_k":
        width = int(k)
        if width <= 0:
            raise ValueError(f"k must be positive, got {k!r}")
        return float(outcome.retrieved) / width
    if metric == "mrr":
        return float(outcome.reciprocal_rank)
    raise ValueError(f"unknown metric {metric!r}; expected one of {POOLED_METRICS}")


def pool_metric_clusters(
    runs: Iterable[object], metric: str, k: int
) -> dict[str, list[float]]:
    """Cluster id -> per-query indicators for `metric`, averaged over seeds.

    `runs` is `ArmResult.runs` (a sequence of `SeedRun`): each carries the full
    list of `QueryOutcome` for one seed. A query answered identically across all
    5 seeds contributes its single value 5 times to the average, which is a
    no-op for the deterministic arms this sweep runs (plain_vector, bm25) and
    is written the general way regardless, matching `eval/runner.py`'s own
    `_pool_clusters` convention: average per query across seeds first, so the
    cluster remains the one and only resampling unit downstream.
    """
    per_query: dict[str, list[float]] = {}
    cluster_of: dict[str, str] = {}
    for run in runs:
        for outcome in run.outcomes:  # type: ignore[attr-defined]
            value = indicator(outcome, metric, k)
            per_query.setdefault(outcome.query_id, []).append(value)
            cluster_of[outcome.query_id] = outcome.cluster_id
    groups: dict[str, list[float]] = {}
    for query_id in sorted(per_query):  # sorted: id order is what a seed cannot pin
        groups.setdefault(cluster_of[query_id], []).append(float(np.mean(per_query[query_id])))
    return {cluster_id: groups[cluster_id] for cluster_id in sorted(groups)}
