"""Tiny planted corpus behind each arm's `python -m brain.baselines.<arm>` path.

Two clusters of near-duplicates that differ in exactly one fact each, which is
the interference structure the real corpus scales up. Enough to see an arm work
end to end on CPU in seconds, never a substitute for the frozen corpus.
"""

from __future__ import annotations

from typing import Any

from brain.store import Trace

__all__ = ["SMOKE_QUERY", "smoke_traces", "run_smoke"]

_EPISODES: tuple[tuple[str, str], ...] = (
    ("c0-e0", "auth/session.py: token refresh failed after the clock skew fix, root cause a stale cache"),
    ("c0-e1", "auth/session.py: token refresh failed under load, root cause an exhausted connection pool"),
    ("c0-e2", "auth/session.py: token refresh failed for one tenant, root cause a truncated secret"),
    ("c1-e0", "infra/vpc.tf: plan drift on the subnet, root cause a manual console edit"),
    ("c1-e1", "infra/vpc.tf: plan drift on the route table, root cause a provider upgrade"),
    ("c1-e2", "infra/vpc.tf: plan drift on the security group, root cause an unmanaged rule"),
)

SMOKE_QUERY = "Which auth/session.py token refresh failure was caused by an exhausted connection pool?"


def smoke_traces(n: int | None = None) -> list[Trace]:
    """The planted corpus, or its first `n` episodes."""
    items = _EPISODES if n is None else _EPISODES[: max(1, int(n))]
    return [Trace(text=text, id=trace_id, cluster_id=trace_id.split("-")[0]) for trace_id, text in items]


def run_smoke(retriever: Any, *, k: int = 3, query: str = SMOKE_QUERY) -> list[str]:
    """Index the planted corpus, run one query, print what happened."""
    from brain.config import SMOKE_TEST

    traces = smoke_traces(3 if SMOKE_TEST else None)
    retriever.index(traces)
    hits = retriever.retrieve_scored(query, min(k, len(traces)))
    print(f"arm={retriever.name} params={retriever.params}")
    for rank, (trace_id, score) in enumerate(hits, start=1):
        print(f"  {rank}. {trace_id}  score={score:.4f}")
    print(f"stats={retriever.stats}")
    return [trace_id for trace_id, _ in hits]
