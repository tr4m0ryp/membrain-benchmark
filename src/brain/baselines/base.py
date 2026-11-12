"""The uniform arm interface every retrieval arm implements (Task 006).

Task 007's harness and Task 008's gate talk to an arm only through `Retriever`:
same `index`, same `retrieve`, same tie-breaking, same embeddings. An arm that
needed a special case downstream would make the comparison unfair by
construction, so the shared machinery lives here rather than in any one arm.

Two things are load-bearing for a fair comparison and live here:

* **One tie-break rule.** Scores tie often (BM25 zeroes, equal distances) and
  Python's sort is stable, so without an explicit key the winner would be
  whatever order an arm happened to build its candidate list in. Every arm
  ranks by `(-score, trace_id)`; higher score is better everywhere.
* **One duplicate check.** `content_sha` is UNIQUE in the store, so two
  byte-identical episodes collapse into one row and the corpus silently
  shrinks. Ids are checked here and row counts are checked by the store-backed
  arms, because a shrunken corpus changes every metric without any error.

The third, one embedding pass for all arms, lives in `vectors.py`.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from brain.store import Trace

__all__ = [
    "IndexingError",
    "Retriever",
    "BaseRetriever",
    "as_traces",
    "rank_hits",
]

_EPOCH = dt.date(1970, 1, 1)


class IndexingError(RuntimeError):
    """The corpus handed to an arm cannot be indexed, or was never indexed."""


@runtime_checkable
class Retriever(Protocol):
    """What every arm exposes. Task 007 needs nothing beyond this."""

    name: str

    def index(self, traces: Sequence[Trace]) -> None:
        """Ingest the corpus. Called once per arm per seed, on a fresh store."""

    def retrieve(self, query: str, k: int) -> list[str]:
        """Return at most k trace ids, best first, deterministically ordered."""



def _timestamp(value: Any) -> float:
    """ISO date to POSIX seconds, 0.0 when absent or unparseable."""
    try:
        return float((dt.date.fromisoformat(str(value)) - _EPOCH).days * 86400)
    except (TypeError, ValueError):
        return 0.0


def as_traces(records: Iterable[Any]) -> list[Trace]:
    """Accept store `Trace`s or corpus `Episode`s and return `Trace`s.

    The corpus record type stays out of this package (baselines must not depend
    on the generator), so episodes are matched structurally on `trace_id` and
    `text`. Nothing is mutated: a `Trace` passes through unchanged.
    """
    out: list[Trace] = []
    for index, record in enumerate(records):
        if isinstance(record, Trace):
            out.append(record)
            continue
        trace_id = getattr(record, "trace_id", None)
        text = getattr(record, "text", None)
        if trace_id is None or text is None:
            raise TypeError(
                f"record {index} is {type(record).__name__}, expected Trace or "
                "an object with .trace_id and .text"
            )
        out.append(
            Trace(
                text=str(text),
                id=str(trace_id),
                ts=_timestamp(getattr(record, "date", "")),
                cluster_id=getattr(record, "cluster_id", None),
            )
        )
    return out


def rank_hits(hits: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Top-k by descending score, ties broken by ascending trace id.

    The id tie-break is what makes an arm reproducible: equal scores are common
    and a stable sort would otherwise freeze the candidate-build order into the
    result.
    """
    ordered = sorted(hits, key=lambda hit: (-float(hit[1]), str(hit[0])))
    return ordered[: max(0, int(k))]


class BaseRetriever(ABC):
    """Shared plumbing: input coercion, guards, ranking, provenance.

    Subclasses implement `_index` and `_retrieve` only. `params` records the
    configuration a result must be attributed to, `stats` records what the arm
    actually did (token spend, degraded queries, adjusted writes).
    """

    name: str = "base"

    def __init__(self, *, name: str | None = None, params: dict[str, Any] | None = None) -> None:
        if name is not None:
            self.name = str(name)
        self.params: dict[str, Any] = dict(params or {})
        self.stats: dict[str, Any] = {"n_indexed": 0, "queries": 0}
        self._traces: list[Trace] = []
        self._indexed = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, n_indexed={len(self._traces)})"

    def __enter__(self) -> "BaseRetriever":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- uniform interface -------------------------------------------------

    def index(self, traces: Sequence[Any]) -> None:
        """Ingest the corpus. Duplicate ids are refused, not silently merged."""
        items = as_traces(traces)
        repeated = sorted(tid for tid, count in Counter(t.id for t in items).items() if count > 1)
        if repeated:
            raise IndexingError(f"{self.name}: duplicate trace ids in corpus: {repeated}")
        self._traces = items
        self._index(items)
        self._indexed = True
        self.stats["n_indexed"] = len(items)

    def retrieve(self, query: str, k: int) -> list[str]:
        """At most k trace ids, best first."""
        return [trace_id for trace_id, _ in self.retrieve_scored(query, k)]

    def prefetch(self, queries: Sequence[str], k: int, **kwargs: Any) -> int:
        """Optional warm-up hook, a no-op for every arm that computes locally.

        Uniform so the harness can call it unconditionally. Only the LLM arm
        overrides it, where warming the response cache in parallel turns a
        multi-hour sequential pass into a bounded one without changing a single
        returned id.
        """
        return 0

    def retrieve_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        """Same ranking as `retrieve`, with the score that produced it.

        Scores are comparable within an arm, not across arms (cosine similarity
        for the vector arms, BM25 weight for the lexical one, a rank for the
        LLM one).
        """
        if not isinstance(query, str):
            raise TypeError(f"query is {type(query).__name__}, expected str")
        if not self._indexed:
            raise IndexingError(f"{self.name}: retrieve() called before index()")
        want = min(int(k), len(self._traces))
        if want <= 0:
            return []
        self.stats["queries"] = int(self.stats.get("queries", 0)) + 1
        return self._retrieve(query, want)

    # -- provenance and lifecycle -----------------------------------------

    @property
    def traces(self) -> list[Trace]:
        return list(self._traces)

    @property
    def trace_ids(self) -> list[str]:
        return [trace.id for trace in self._traces]

    @property
    def blocked_reason(self) -> str | None:
        """Why this arm cannot run, or None. Task 008 reports BLOCKED, not zero."""
        return None

    def is_available(self) -> bool:
        return self.blocked_reason is None

    def describe(self) -> dict[str, Any]:
        """Everything a result file needs to attribute a number to this arm."""
        return {
            "name": self.name,
            "arm_class": type(self).__name__,
            "params": dict(self.params),
            "stats": dict(self.stats),
            "blocked_reason": self.blocked_reason,
        }

    def close(self) -> None:
        """Release owned resources. Idempotent, safe on an unindexed arm."""

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def _index(self, traces: Sequence[Trace]) -> None:
        """Ingest `traces`, already coerced and id-checked."""

    @abstractmethod
    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return exactly the ranked (trace_id, score) pairs, at most k of them."""


