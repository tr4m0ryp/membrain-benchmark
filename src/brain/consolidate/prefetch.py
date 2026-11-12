"""Warm the merge cache in parallel. Optional, and results are unchanged.

A cold sweep over the frozen corpus is one sequential call per planned group at
tens of seconds each, which is an hour of wall clock for a pass that costs under
a dollar. The calls are independent and the cache is keyed by the exact request,
so running them concurrently changes the wall clock and nothing else: the merge
a consolidation run later reads from disk is byte-identical to the one it would
have made itself.

This is the same trick the full-context arm uses (`baselines.full_context
.prefetch`), for the same reason and with the same guarantee.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from brain.baselines.llm import DEFAULT_MODEL, LlmUnavailable, select_client
from brain.config import SEED
from brain.consolidate.baseline import DEFAULT_CACHE_DIR
from brain.consolidate.cache import MergeCache
from brain.consolidate.grouping import plan_groups
from brain.consolidate.prompt import (
    MAX_OUTPUT_TOKENS,
    block_for,
    merge_cache_key,
    merge_schema,
    merge_system_prompt,
    merge_user_prompt,
)
from brain.consolidate.types import (
    DEFAULT_LINKAGE,
    DEFAULT_THRESHOLDS,
    MAX_SPEND_USD,
    ConsolidationSpend,
    MergeGroup,
)
from brain.store import EpisodicStore

__all__ = ["merge_request", "warm_cache"]


def merge_request(group: MergeGroup, by_id: dict[str, Any], *, model: str,
                  temperature: float, seed: int) -> dict[str, Any]:
    """The exact request one group produces, digest included.

    Built here and in `BaselineConsolidator._request` from the same helpers, so
    a warmed entry is the one the real pass looks up rather than a near miss.
    """
    blocks = [block_for(i, by_id[i].text) for i in group.member_ids]
    system = merge_system_prompt(blocks)
    user = merge_user_prompt(len(blocks))
    return {
        "digest": merge_cache_key(
            member_ids=group.member_ids,
            member_shas=[by_id[i].content_sha for i in group.member_ids],
            model=model,
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            seed=seed,
            system=system,
            user=user,
        ),
        "system": system,
        "user": user,
        "model": model,
        "temperature": temperature,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "schema": merge_schema(),
    }


def warm_cache(
    store: EpisodicStore,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    linkage: str = DEFAULT_LINKAGE,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    seed: int = SEED,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    client: Any | None = None,
    workers: int = 6,
    budget_usd: float = MAX_SPEND_USD,
) -> dict[str, Any]:
    """Fill the cache for every group in the sweep. Returns what it cost.

    Deduplicates by digest first: two thresholds that propose the same group
    share one call. Failures are counted, not raised, because the real pass will
    retry any entry that is still missing.
    """
    backend, reason = (client, None) if client is not None else select_client()
    if backend is None:
        return {"blocked_reason": reason, "pending": 0, "warmed": 0}
    cache = MergeCache(backend, cache_dir)
    by_id = {trace.id: trace for trace in store.all_traces(valid_only=True)}
    plan = plan_groups(store, thresholds=thresholds, linkage=linkage)

    requests: dict[str, dict[str, Any]] = {}
    for groups in plan.values():
        for group in groups:
            request = merge_request(
                group, by_id, model=model, temperature=temperature, seed=seed
            )
            requests.setdefault(request["digest"], request)
    pending = [r for r in requests.values() if not cache.has(r["digest"])]

    spend = ConsolidationSpend()
    lock = Lock()
    failures = 0

    def warm(request: dict[str, Any]) -> None:
        nonlocal failures
        try:
            response = cache.complete(**request)
        except LlmUnavailable:
            with lock:
                failures += 1
            return
        with lock:
            spend.add(response, model)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            list(pool.map(warm, pending))
    return {
        "groups_planned": sum(len(groups) for groups in plan.values()),
        "unique_requests": len(requests),
        "pending": len(pending),
        "warmed": cache.calls,
        "failures": failures,
        "over_budget": spend.cost_usd > float(budget_usd),
        "spend": spend.to_dict(),
    }
