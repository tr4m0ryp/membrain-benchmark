"""Plain vector top-k: the primary control arm (Evaluation Plan, s = 0).

This is the arm the separation mechanism has to beat, and the one the design
falls back to if H-A dies, so it is implemented as the strongest honest version
of itself: the same embedder, the same store, the same k-NN, no separation, no
read-time reranking, no query rewriting. Every difference between this arm and
the separation arm is then attributable to the write-time transform alone.

`VectorStoreArm` holds the store plumbing shared with the write-time-MMR arm
(Task 006) and mirrors what the separation policy (Task 005) does at write, so
that all three differ only in the vector they hand to the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brain.baselines.base import BaseRetriever, IndexingError
from brain.baselines.vectors import VectorCache, shared_vector_cache
from brain.store import EpisodicStore, Trace

__all__ = ["VectorStoreArm", "PlainVectorRetriever"]


class VectorStoreArm(BaseRetriever):
    """k-NN over an `EpisodicStore`, with the embedding pass shared by all arms.

    The store dimension is read from the embedder, never hardcoded. When no
    store is passed the arm owns a private in-memory one, which is what the
    harness wants: a fresh store per seed, no leakage between runs.
    """

    name = "vector_store"

    def __init__(
        self,
        *,
        store: EpisodicStore | None = None,
        db_path: str | Path = ":memory:",
        cache: VectorCache | None = None,
        name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        self._cache = cache if cache is not None else shared_vector_cache()
        self._owns_store = store is None
        self._store = store if store is not None else EpisodicStore(db_path, self._cache.dim)
        if self._store.dim != self._cache.dim:
            raise IndexingError(
                f"store dim {self._store.dim} does not match embedder dim {self._cache.dim}"
            )
        self.params.setdefault("store_dim", self._store.dim)

    @property
    def store(self) -> EpisodicStore:
        return self._store

    @property
    def cache(self) -> VectorCache:
        return self._cache

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    # -- write path --------------------------------------------------------

    def _index(self, traces: Sequence[Trace]) -> None:
        texts = [trace.text for trace in traces]
        vectors = self._cache.matrix(texts)
        before = self._store.count()
        self._write(traces, vectors)
        written = self._store.count() - before
        if written != len(traces):
            # content_sha is UNIQUE, so byte-identical episodes collapse into one
            # row. Silently indexing a smaller corpus would move every metric.
            raise IndexingError(
                f"{self.name}: indexed {len(traces)} traces but the store gained "
                f"{written} rows; byte-identical texts were deduped"
            )

    def _write(self, traces: Sequence[Trace], vectors: np.ndarray) -> None:
        """Store the embeddings as they came out of the embedder (no separation)."""
        self._store.add_many(traces, vectors)

    # -- read path ---------------------------------------------------------

    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Cosine k-NN. `knn` already orders by (distance, id), so ties are pinned."""
        vector = self._cache.vector(query)
        return [(trace_id, 1.0 - distance) for trace_id, distance in self._store.knn(vector, k)]


class PlainVectorRetriever(VectorStoreArm):
    """Embed, store, cosine top-k. Separation strength s = 0.

    The control arm for H-A. Its parameters are recorded so a result cannot be
    confused with a tuned variant: there is nothing to tune here on purpose.
    """

    name = "plain_vector"

    def __init__(
        self,
        *,
        store: EpisodicStore | None = None,
        db_path: str | Path = ":memory:",
        cache: VectorCache | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            store=store,
            db_path=db_path,
            cache=cache,
            name=name,
            params={"separation_s": 0.0, "read_time_rerank": None, "metric": "cosine"},
        )


def _smoke() -> None:
    """Tiny CPU path: python -m brain.baselines.plain_vector."""
    from brain.baselines.smoke import run_smoke

    with PlainVectorRetriever() as arm:
        run_smoke(arm)


if __name__ == "__main__":
    _smoke()
