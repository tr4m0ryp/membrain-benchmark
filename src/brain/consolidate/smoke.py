"""Tiny-N path and cost projection: `python -m brain.consolidate.smoke`.

Runs the mandated order from rules.md. Verify the frozen corpus hash, plan the
whole threshold sweep for free (grouping calls no model), smoke N=10 merges,
then extrapolate the full pass from the measured per-call cost and stop loudly
if the projection passes the budget.

The projection is exact in its call count rather than extrapolated: a merge
depends only on its group's texts, so the sweep needs exactly one call per
planned group, shared across both destructive modes and across every seed.
"""

from __future__ import annotations

from typing import Any, Sequence

from brain.baselines.base import as_traces
from brain.baselines.vectors import shared_vector_cache
from brain.config import SEED, SMOKE_TEST
from brain.consolidate.baseline import (
    DEFAULT_CACHE_DIR,
    BaselineConsolidator,
    run_consolidation,
)
from brain.consolidate.cache import MergeCache
from brain.consolidate.grouping import plan_groups
from brain.consolidate.prefetch import merge_request
from brain.consolidate.types import (
    DEFAULT_LINKAGE,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    MAX_SPEND_USD,
    MergeGroup,
)
from brain.store import EpisodicStore

__all__ = ["FROZEN_CORPUS_SHA", "source_store", "project_pass", "main"]

#: rules.md: every result to date is stated against this hash. Asserted, never
#: assumed, and the corpus is only ever read.
FROZEN_CORPUS_SHA = "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb"


def source_store(corpus: Any, *, db_path: str = ":memory:") -> EpisodicStore:
    """A store holding the corpus, built the same way every arm builds one."""
    traces = as_traces(corpus.episodes)
    cache = shared_vector_cache()
    store = EpisodicStore(db_path, cache.dim)
    store.add_many(traces, cache.matrix([trace.text for trace in traces]))
    return store


def _digest(group: MergeGroup, by_id: dict[str, Any], model: str, seed: int) -> str:
    """The digest the real pass will look up, built from the same helpers."""
    return merge_request(group, by_id, model=model, temperature=0.0, seed=seed)["digest"]


def project_pass(
    store: EpisodicStore,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    linkage: str = DEFAULT_LINKAGE,
    model: str = "claude-haiku-4-5",
    seed: int = SEED,
    cache_dir: Any = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Exact call counts for the whole sweep, split into cached and pending."""
    plan = plan_groups(store, thresholds=thresholds, linkage=linkage)
    by_id = {trace.id: trace for trace in store.all_traces(valid_only=True)}
    cache = MergeCache(None, cache_dir)
    digests = {
        tau: [_digest(group, by_id, model, seed) for group in groups]
        for tau, groups in plan.items()
    }
    pending = {tau: sum(1 for d in items if not cache.has(d)) for tau, items in digests.items()}
    return {
        "groups_per_threshold": {tau: len(groups) for tau, groups in plan.items()},
        "episodes_per_threshold": {
            tau: sum(g.size for g in groups) for tau, groups in plan.items()
        },
        "calls_total": sum(len(groups) for groups in plan.values()),
        "calls_pending": sum(pending.values()),
        "pending_per_threshold": pending,
    }


def main(*, n_smoke: int = 10, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Verify, plan, smoke at N, extrapolate, and stop if over budget."""
    from brain.corpus.loader import load_corpus

    corpus = load_corpus()
    sha = corpus.content_hash
    assert sha == FROZEN_CORPUS_SHA, f"frozen corpus changed: {sha}"
    print(f"frozen corpus SHA {sha} (verified, read-only)")
    print(f"episodes={len(corpus.episodes)} queries={len(corpus.queries)}")

    store = source_store(corpus)
    plan = project_pass(store)
    print(f"groups per threshold : {plan['groups_per_threshold']}")
    print(f"episodes grouped     : {plan['episodes_per_threshold']}")
    print(f"sweep calls total    : {plan['calls_total']}  pending {plan['calls_pending']}")

    n = 2 if SMOKE_TEST else int(n_smoke)
    probe = BaselineConsolidator(threshold=threshold, max_groups=n)
    if not probe.is_available():
        print(f"BLOCKED: {probe.blocked_reason}")
        return
    result = run_consolidation(store, threshold=threshold, max_groups=n)
    spend = result.spend
    print(f"\nsmoke N={n} at tau={threshold}: {result.describe()['stats']}")
    print(f"spend: {spend.to_dict()}")

    per_call = spend.cost_per_call_usd
    if not per_call:
        print("no live calls (all replayed from cache); full pass is free")
        return
    projected = per_call * plan["calls_pending"]
    cold = per_call * plan["calls_total"]
    print(
        f"\nper-call {per_call:.4f} USD -> pending pass {projected:.2f} USD "
        f"(cold {cold:.2f} USD over {plan['calls_total']} calls)"
    )
    if projected > MAX_SPEND_USD:
        print(f"STOP: projected {projected:.2f} USD exceeds the {MAX_SPEND_USD} USD budget")
    else:
        print(f"within the {MAX_SPEND_USD} USD budget")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
