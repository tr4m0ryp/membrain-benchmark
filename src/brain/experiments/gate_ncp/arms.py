"""The comparison arms and the cost probe: everything that is not the NCP grid.

Kept apart from `runner` because these answer a different question. `run_retriever_arm`
puts any `Retriever` through the same metric and bootstrap path the grid uses, so a
baseline number and a cell number are commensurable by construction rather than by
inspection. `cost_probe` measures what NCP actually costs to write and to query, at a
given corpus size, with the LLM call count reported as the literal zero it is.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from brain.baselines.base import as_traces
from brain.config import seed_everything
from brain.corpus.types import Corpus
from brain.eval.metrics import score_query
from brain.eval.runner import DEFAULT_K
from brain.experiments.gate_ncp.runner import PRIMARY_K, evaluate

__all__ = ["run_retriever_arm", "run_baselines", "cost_probe"]


def run_retriever_arm(name: str, factory: Callable[[], Any], corpus: Any, *,
                      seeds: Sequence[int], k: int, subset: set[str], n_resamples: int,
                      alpha_level: float = 0.05, bootstrap_seed: int = 0) -> dict[str, Any]:
    """One comparison arm through the same metric and bootstrap path as the grid."""
    episodes = sorted(corpus.episodes, key=lambda e: str(e.trace_id))
    queries = sorted(corpus.queries, key=lambda q: str(q.query_id))
    per_seed, stats, describe, clock = [], {}, {}, time.perf_counter()
    for seed in sorted({int(s) for s in seeds}):
        seed_everything(seed)
        arm = factory()
        try:
            if getattr(arm, "blocked_reason", None):
                raise RuntimeError(f"{name}: BLOCKED ({arm.blocked_reason})")
            arm.index(as_traces(episodes))
            per_seed.append([score_query(q, arm.retrieve(q.text, int(k))) for q in queries])
            stats, describe = dict(getattr(arm, "stats", {}) or {}), arm.describe()
        finally:
            if hasattr(arm, "close"):
                arm.close()
    return {"arm": name, "stats": stats, "describe": describe,
            "wall_clock": time.perf_counter() - clock,
            **evaluate(per_seed, k=int(k), subset=subset, alpha=alpha_level,
                       n_resamples=n_resamples, bootstrap_seed=bootstrap_seed)}


def cost_probe(traces: Sequence[Any], queries: Sequence[str], *, gamma: float = 1.0,
               k_neighbours: int = PRIMARY_K, k: int = DEFAULT_K) -> dict[str, Any]:
    """Measured write and query cost at one corpus size. Zero LLM calls, by type.

    A PRIVATE token cache, not the shared one: by the time the probe runs the
    shared cache is warm from the grid, and a warm re-read would report the write
    cost as zero. The cache is dropped when this function returns.
    """
    from brain.baselines.ncp import NcpRetriever
    from brain.embed.pooling import TokenCache

    tokens = TokenCache()
    clock = time.perf_counter()
    encoded = tokens.ensure([t.text for t in traces])
    encode_seconds = time.perf_counter() - clock
    arm = NcpRetriever(gamma=gamma, alpha=0.5, k_neighbours=k_neighbours, tokens=tokens)
    clock = time.perf_counter()
    arm.index(traces)
    index_seconds = time.perf_counter() - clock
    arm.prefetch(list(queries), k)
    clock = time.perf_counter()
    for query in queries:
        arm.retrieve(query, k)
    query_seconds = time.perf_counter() - clock
    return {"n_episodes": len(traces), "n_queries": len(queries),
            "token_encode_seconds": round(encode_seconds, 3), "texts_encoded": int(encoded),
            "index_seconds": round(index_seconds, 4),
            "query_seconds_total": round(query_seconds, 4),
            "query_ms_per_query": round(1000.0 * query_seconds / max(1, len(queries)), 4),
            "llm_calls": 0, "cost_usd": 0.0,
            "extra_bytes_per_episode": int(arm.channels()[2].shape[1] * 4)}


def run_baselines(corpus: Corpus, *, seeds: tuple[int, ...], k: int, subset: set[str],
                  n_resamples: int, progress: Any = None) -> tuple[dict[str, Any], dict[str, str]]:
    """The five required arms. A paid arm that cannot replay is BLOCKED, never zero."""
    from brain.baselines.bm25 import Bm25Retriever
    from brain.baselines.full_context import FullContextRetriever
    from brain.baselines.plain_vector import PlainVectorRetriever
    from brain.baselines.veto.llm_pick import LlmPickRetriever

    results: dict[str, Any] = {}
    blocked: dict[str, str] = {}
    sub = Corpus(corpus.episodes, [q for q in corpus.queries if str(q.query_id) in subset],
                 corpus.manifest)
    plan = (("plain_vector", PlainVectorRetriever, corpus, seeds),
            ("bm25", Bm25Retriever, corpus, seeds),
            ("full_context", FullContextRetriever, corpus, (seeds[0],)),
            ("llm_pick", LlmPickRetriever, sub, (seeds[0],)))
    for name, factory, target, arm_seeds in plan:
        keep = subset if target is corpus else {str(q.query_id) for q in sub.queries}
        try:
            results[name] = run_retriever_arm(name, factory, target, seeds=arm_seeds, k=k,
                                              subset=keep, n_resamples=n_resamples)
            results[name]["query_set"] = "full" if target is corpus else "subsample"
            results[name]["seed_independent"] = len(arm_seeds) == 1
        except Exception as exc:  # BLOCKED is a reportable outcome, not a crash
            blocked[name] = f"{type(exc).__name__}: {exc}"
        if progress:
            progress(f"baseline {name}: {'blocked' if name in blocked else 'done'}")
    return results, blocked
