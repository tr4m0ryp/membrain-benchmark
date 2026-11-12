"""NCP: the two-channel retrieval arm. Zero LLM calls, no training, no new weights.

    score(q, i)  =  alpha * cos(q_full, v_full^(i))  +  (1 - alpha) * cos(q_dist, v_dist^(i))

`v_full` is today's mean-pooled vector, unchanged, and carries topic recall.
`v_dist` is pooled with `w_t^(i) = (1 - df_t^(i)) ** gamma`, where df is measured
against episode i's own k nearest neighbours, and carries disambiguation. Two
channels because one is not enough: a query that cues shared content would fail
against distinctive-only vectors, since exactly those tokens were suppressed.

**Query side.** A query has no neighbourhood, so its distinctive channel cannot
use the same rule. Two variants are implemented and both are reported rather
than one being chosen quietly:

* `query_side="mean"`: `q_dist = q_full`, the mean-pooled query. The asymmetry is
  deliberate; the document side is what the neighbourhood is a property of.
* `query_side="idf"`: the query is pooled with GLOBAL corpus df at the same
  gamma (SIF weights when the document side is SIF), which is the closest
  well-defined analogue of the document rule for a text with no neighbours.

**Why this is not the killed C-A mechanism.** C-A was a symmetric transform on
finished vectors: a colliding pair shared a normalizer, so the sign of the score
difference did not depend on the knob and the pair could never be reordered.
NCP acts BEFORE pooling, on different token sets, with a different weight vector
per episode, so no shared normalizer exists and the premise of that theorem
fails. The experiment verifies the conclusion empirically (it counts reordered
sibling pairs) rather than resting on this paragraph.

**Storage and cost.** One extra float32 vector per episode. Write: one k-NN probe
over the store plus a token-level encode, both batched offline. Query: two dot
products against the store instead of one. `df` depends on the neighbourhood, so
adding an episode changes its neighbours' weights; that is a batch recomputation
question and is deliberately not solved online here.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from brain.baselines.base import BaseRetriever, IndexingError, rank_hits
from brain.embed.pooling.encode import TokenCache, mean_pool, pool_weighted, shared_token_cache
from brain.embed.pooling.weights import (
    SIF_DEFAULT_A,
    global_document_frequency,
    global_weights,
    local_weights,
    neighbour_indices,
    remove_first_principal_component,
    sif_weights,
    token_sets,
    unigram_probabilities,
)
from brain.store import Trace

__all__ = ["WEIGHT_SOURCES", "QUERY_SIDES", "DEFAULT_K_NEIGHBOURS", "NcpRetriever"]

#: `uniform` is the identity (mean pooling on both channels) and exists so the
#: gamma = 0 correctness check runs through the same code path as every cell.
WEIGHT_SOURCES = ("uniform", "local", "global", "sif")
QUERY_SIDES = ("mean", "idf")
#: The largest interference cluster on the frozen corpus is 8, so k = 8 is the
#: smallest neighbourhood guaranteed to be able to contain every sibling of every
#: episode. Pre-registered before any number was seen.
DEFAULT_K_NEIGHBOURS = 8


class NcpRetriever(BaseRetriever):
    """Neighbourhood-conditional pooling behind the uniform `Retriever` interface."""

    name = "ncp"

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        alpha: float = 0.5,
        k_neighbours: int = DEFAULT_K_NEIGHBOURS,
        weights: str = "local",
        query_side: str = "mean",
        sif_a: float = SIF_DEFAULT_A,
        remove_pc: bool = False,
        tokens: TokenCache | None = None,
        name: str | None = None,
    ) -> None:
        if weights not in WEIGHT_SOURCES:
            raise ValueError(f"weights must be one of {list(WEIGHT_SOURCES)}, got {weights!r}")
        if query_side not in QUERY_SIDES:
            raise ValueError(f"query_side must be one of {list(QUERY_SIDES)}, got {query_side!r}")
        if float(gamma) < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma!r}")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        super().__init__(
            name=name,
            params={
                "gamma": float(gamma),
                "alpha": float(alpha),
                "k_neighbours": int(k_neighbours),
                "weights": str(weights),
                "query_side": str(query_side),
                "sif_a": float(sif_a),
                "remove_pc": bool(remove_pc),
                "metric": "cosine",
                "llm_calls_per_query": 0,
            },
        )
        self._gamma = float(gamma)
        self._alpha = float(alpha)
        self._k = int(k_neighbours)
        self._source = str(weights)
        self._query_side = str(query_side)
        self._sif_a = float(sif_a)
        self._remove_pc = bool(remove_pc)
        self._tokens = tokens if tokens is not None else shared_token_cache()
        self._ids: list[str] = []
        self._full: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self._dist: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self._global_df: dict[int, float] = {}
        self._unigram: dict[int, float] = {}

    # -- write path --------------------------------------------------------

    def _index(self, traces: Sequence[Trace]) -> None:
        texts = [trace.text for trace in traces]
        docs = self._tokens.docs(texts)
        full = mean_pool(docs)  # identical to what plain_vector stores
        sets = token_sets(docs)
        self._global_df = global_document_frequency(sets)
        self._unigram = unigram_probabilities(docs)
        order = neighbour_indices(full, self._k)
        self.stats["neighbours_per_episode"] = int(order.shape[1]) if order.size or len(docs) else 0
        rows = self._weight_rows(docs, sets, order)
        if rows is None:
            dist, pooling = full, {"degenerate_documents": 0, "documents": len(docs)}
        else:
            dist, pooling = pool_weighted(docs, rows)
            if self._remove_pc:
                dist = remove_first_principal_component(dist)
        self._ids = [trace.id for trace in traces]
        self._full = np.asarray(full, dtype=np.float64)
        self._dist = np.asarray(dist, dtype=np.float64)
        if self._full.shape != self._dist.shape:
            raise IndexingError(f"channel shapes differ: {self._full.shape} vs {self._dist.shape}")
        self.stats.update(
            {
                "degenerate_documents": int(pooling["degenerate_documents"]),
                "mean_tokens_per_doc": float(np.mean([doc.n_tokens for doc in docs])) if docs else 0.0,
                "distinct_token_types": len(self._global_df),
                "mean_token_weight": (
                    float(np.mean([float(np.mean(row)) for row in rows])) if rows else 1.0
                ),
            }
        )

    def _weight_rows(self, docs: Sequence[Any], sets: Sequence[frozenset[int]], order: np.ndarray):
        """The document-side weight vectors, or None for the identity (uniform)."""
        if self._source == "uniform":
            return None
        if self._source == "local":
            return local_weights(docs, sets, order, self._gamma)
        if self._source == "global":
            return global_weights(docs, self._global_df, self._gamma)
        return sif_weights(docs, self._unigram, self._sif_a)

    # -- read path ---------------------------------------------------------

    def channels(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Ids and both stored channels, for the geometry diagnostics."""
        return list(self._ids), self._full.copy(), self._dist.copy()

    def query_vectors(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """`(q_full, q_dist)` for one query. Both unit norm, so cosine is a dot."""
        doc = self._tokens.doc(query)
        full = np.asarray(mean_pool([doc])[0], dtype=np.float64)
        if self._source == "uniform" or self._query_side == "mean":
            return full, full
        if self._source == "sif":
            rows = sif_weights([doc], self._unigram, self._sif_a)
        else:
            rows = global_weights([doc], self._global_df, self._gamma)
        return full, np.asarray(pool_weighted([doc], rows)[0][0], dtype=np.float64)

    def scores(self, query: str) -> np.ndarray:
        """The blended score against every indexed episode, in index order."""
        if self._full.size == 0:
            return np.zeros(0, dtype=np.float64)
        full, dist = self.query_vectors(query)
        return self._alpha * (self._full @ full) + (1.0 - self._alpha) * (self._dist @ dist)

    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        values = self.scores(query)
        return rank_hits(zip(self._ids, values.tolist()), k)

    def prefetch(self, queries: Sequence[str], k: int, **kwargs: Any) -> int:
        """Token-encode the queries in ONE batch. No network, no LLM, no cost."""
        return int(self._tokens.ensure([q for q in queries if isinstance(q, str)]))


def _smoke() -> None:
    """Tiny CPU path: python -m brain.baselines.ncp."""
    from brain.baselines.smoke import run_smoke

    with NcpRetriever(gamma=1.0, alpha=0.5, k_neighbours=2) as arm:
        run_smoke(arm, k=3)


if __name__ == "__main__":
    _smoke()
