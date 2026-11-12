"""Metrics for the gate. Disambiguation accuracy is the headline; the rest are context.

The metric that tests separation is **disambiguation accuracy**: over queries that
have a correct trace and at least one sibling distractor, the fraction where the
correct trace outranks ALL of its siblings. It is scored strictly against the
siblings named by the corpus, never against the whole store, because a query the
arm answers with an unrelated episode is not evidence about interference.

Three readings are fixed here, because each one moves the number:

1. **A sibling that was not retrieved ranks below a retrieved correct trace.**
   The arms return a truncated ranking of length k, so "outranks" is evaluated on
   positions that exist; an absent sibling lost. If the correct trace itself is
   missing from the ranking the query is a miss, whatever the siblings did.
2. **Ineligible queries leave the denominator.** A query with no sibling cannot
   test disambiguation, so it is counted and excluded rather than scored as free
   credit. On the frozen corpus every cluster has m >= 2, so all 360 are eligible.
3. **An empty input is NaN, never 0.0.** A metric with no observations behind it
   must not be reportable as a bad score.

Records are duck-typed on `query_id`, `cluster_id`, `cluster_size`,
`correct_trace_id` and `distractor_ids`, so this module stays independent of the
corpus package and can be exercised on a hand-built toy case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "QueryOutcome",
    "score_query",
    "score_queries",
    "disambiguation_accuracy",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "accuracy_by_cluster_size",
    "indicators_by_cluster",
    "indicators_by_cluster_size",
    "metric_summary",
    "METRIC_NAMES",
]

#: Metrics reported for every arm, in a fixed order so result files line up.
METRIC_NAMES: tuple[str, ...] = ("disambiguation", "recall_at_k", "precision_at_k", "mrr")


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """What one query did on one arm, in one seed.

    `disambiguated` is the disambiguation indicator and is None exactly when the
    query is ineligible, which keeps "no sibling to beat" distinct from "lost to
    a sibling" in every downstream average.
    """

    query_id: str
    cluster_id: str
    cluster_size: int
    eligible: bool
    disambiguated: bool | None
    retrieved: bool
    correct_rank: int | None  # 1-based position of the correct trace
    reciprocal_rank: float
    n_distractors: int
    distractors_ahead: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "cluster_id": self.cluster_id,
            "cluster_size": int(self.cluster_size),
            "eligible": bool(self.eligible),
            "disambiguated": self.disambiguated,
            "retrieved": bool(self.retrieved),
            "correct_rank": self.correct_rank,
            "reciprocal_rank": float(self.reciprocal_rank),
            "n_distractors": int(self.n_distractors),
            "distractors_ahead": int(self.distractors_ahead),
        }


def score_query(query: Any, ranked_ids: Sequence[str]) -> QueryOutcome:
    """Score one ranked list against one query's ground truth."""
    correct = str(query.correct_trace_id)
    # Sorted and de-duplicated: the distractor list is read from disk, and set
    # iteration order over ids is the one thing a seed cannot pin (rules.md).
    distractors = sorted({str(tid) for tid in query.distractor_ids})
    if correct in distractors:
        raise ValueError(f"query {query.query_id!r} lists its correct trace as a distractor")
    position: dict[str, int] = {}
    for index, trace_id in enumerate(ranked_ids):
        position.setdefault(str(trace_id), index)  # a repeated id keeps its best rank
    correct_position = position.get(correct)
    ahead = sum(
        1
        for tid in distractors
        if position.get(tid, math.inf) < (math.inf if correct_position is None else correct_position)
    )
    eligible = bool(distractors)
    retrieved = correct_position is not None
    return QueryOutcome(
        query_id=str(query.query_id),
        cluster_id=str(query.cluster_id),
        cluster_size=int(getattr(query, "cluster_size", len(distractors) + 1)),
        eligible=eligible,
        disambiguated=(retrieved and ahead == 0) if eligible else None,
        retrieved=retrieved,
        correct_rank=None if correct_position is None else correct_position + 1,
        reciprocal_rank=0.0 if correct_position is None else 1.0 / (correct_position + 1),
        n_distractors=len(distractors),
        distractors_ahead=ahead,
    )


def score_queries(queries: Sequence[Any], rankings: Sequence[Sequence[str]]) -> list[QueryOutcome]:
    """Score aligned queries and rankings. Lengths must match exactly."""
    if len(queries) != len(rankings):
        raise ValueError(f"{len(queries)} queries but {len(rankings)} rankings")
    return [score_query(query, ranking) for query, ranking in zip(queries, rankings)]


def _mean(values: Iterable[float]) -> float:
    """Arithmetic mean, NaN on no observations (never a silent 0.0)."""
    items = [float(value) for value in values]
    return math.fsum(items) / len(items) if items else math.nan


def _eligible(outcomes: Iterable[QueryOutcome]) -> list[QueryOutcome]:
    return [outcome for outcome in outcomes if outcome.eligible]


def disambiguation_accuracy(outcomes: Iterable[QueryOutcome]) -> float:
    """Fraction of ELIGIBLE queries whose correct trace outranks every sibling."""
    return _mean(float(bool(outcome.disambiguated)) for outcome in _eligible(outcomes))


def recall_at_k(outcomes: Iterable[QueryOutcome]) -> float:
    """Fraction of queries whose correct trace appears in the returned top-k.

    One relevant document per query, so this is also hit rate at k.
    """
    return _mean(float(outcome.retrieved) for outcome in outcomes)


def precision_at_k(outcomes: Iterable[QueryOutcome], k: int) -> float:
    """Mean precision at k. With one relevant document per query this is recall/k."""
    width = int(k)
    if width <= 0:
        raise ValueError(f"k must be positive, got {k!r}")
    return _mean(float(outcome.retrieved) / width for outcome in outcomes)


def mrr(outcomes: Iterable[QueryOutcome]) -> float:
    """Mean reciprocal rank of the correct trace, 0 for a query that missed it."""
    return _mean(outcome.reciprocal_rank for outcome in outcomes)


def accuracy_by_cluster_size(outcomes: Iterable[QueryOutcome]) -> dict[int, float]:
    """Disambiguation accuracy per cluster size: the M-A curve, ungrouped by seed.

    M-A predicts this decays in m for a plain store and stays flat under write-time
    separation, so it is reported for every arm, not only the mechanism.
    """
    buckets: dict[int, list[float]] = {}
    for outcome in _eligible(outcomes):
        buckets.setdefault(int(outcome.cluster_size), []).append(float(bool(outcome.disambiguated)))
    return {size: _mean(buckets[size]) for size in sorted(buckets)}


def indicators_by_cluster(outcomes: Iterable[QueryOutcome]) -> dict[str, list[float]]:
    """Per-cluster disambiguation indicators, the resampling unit for the bootstrap.

    Queries are grouped by the cluster they belong to because three queries share
    an episode and m episodes share a cluster: they are not independent draws, and
    resampling them individually would understate the variance (F11 caution).
    """
    groups: dict[str, list[float]] = {}
    for outcome in _eligible(outcomes):
        groups.setdefault(str(outcome.cluster_id), []).append(float(bool(outcome.disambiguated)))
    return {cluster_id: groups[cluster_id] for cluster_id in sorted(groups)}


def indicators_by_cluster_size(
    outcomes: Iterable[QueryOutcome],
) -> dict[int, dict[str, list[float]]]:
    """`indicators_by_cluster`, stratified by cluster size (for the curve's CIs)."""
    strata: dict[int, list[QueryOutcome]] = {}
    for outcome in _eligible(outcomes):
        strata.setdefault(int(outcome.cluster_size), []).append(outcome)
    return {size: indicators_by_cluster(strata[size]) for size in sorted(strata)}


def metric_summary(outcomes: Sequence[QueryOutcome], k: int) -> dict[str, float]:
    """Every headline metric for one arm-seed, plus the counts behind them."""
    eligible = _eligible(outcomes)
    return {
        "disambiguation": disambiguation_accuracy(outcomes),
        "recall_at_k": recall_at_k(outcomes),
        "precision_at_k": precision_at_k(outcomes, k),
        "mrr": mrr(outcomes),
        "n_queries": float(len(outcomes)),
        "n_eligible": float(len(eligible)),
        "n_clusters": float(len(indicators_by_cluster(outcomes))),
    }
