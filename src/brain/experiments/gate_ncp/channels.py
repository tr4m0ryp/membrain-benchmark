"""The corpus-side math of one cell: both channels, and the geometry they induce.

Split out of `runner` because it answers a different question. Everything here is
a pure function of the corpus texts and one weight setting, so it is computed
once per (weights, gamma, k) and reused across every alpha, which is what keeps a
300-cell surface cheap.

The two channels:

    v_full^(i)  =  mean pooling            (today's stored vector, byte for byte)
    v_dist^(i)  =  normalize( SUM_t w_t^(i) h_t^(i) / SUM_t w_t^(i) )

and the score a cell ranks by is `alpha * cos(q_full, v_full) + (1 - alpha) *
cos(q_dist, v_dist)`. That is a dot product in the concatenated space
`[sqrt(alpha) v_full, sqrt(1 - alpha) v_dist]`, so the sibling cosine of the
geometry retrieval actually runs in is `alpha * cos_full + (1 - alpha) * cos_dist`,
which is what `blended_geometry` returns and what every cell reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from brain.embed.pooling import (
    global_document_frequency, global_weights, local_weights, mean_pool, neighbour_indices,
    pool_weighted, remove_first_principal_component, sif_weights, token_sets,
    unigram_probabilities,
)

__all__ = ["CorpusState", "corpus_state", "weight_rows", "dist_channel", "query_channel",
           "geometry", "blended_geometry"]


@dataclass(frozen=True, slots=True)
class CorpusState:
    """Everything derived once from the corpus texts, shared by every cell."""

    docs: tuple[Any, ...]
    sets: tuple[frozenset[int], ...]
    full: np.ndarray  # mean-pooled, float64, unit rows: today's stored vectors
    global_df: dict[int, float]
    unigram: dict[int, float]
    encode_seconds: float


def corpus_state(texts: Sequence[str], tokens: Any) -> CorpusState:
    """Token-encode the corpus once and derive every statistic the grid needs."""
    started = time.perf_counter()
    docs = tokens.docs(list(texts))
    sets = token_sets(docs)
    return CorpusState(
        docs=tuple(docs), sets=tuple(sets),
        full=np.asarray(mean_pool(docs), dtype=np.float64),
        global_df=global_document_frequency(sets), unigram=unigram_probabilities(docs),
        encode_seconds=time.perf_counter() - started,
    )


def weight_rows(state: CorpusState, spec: dict[str, Any]) -> list[np.ndarray] | None:
    """The document-side weight vectors for `spec`, or None for the identity."""
    kind = str(spec["weights"])
    if kind == "uniform":
        return None
    if kind == "local":
        order = neighbour_indices(state.full, int(spec["k_neighbours"]))
        return local_weights(list(state.docs), list(state.sets), order, float(spec["gamma"]))
    if kind == "global":
        return global_weights(list(state.docs), state.global_df, float(spec["gamma"]))
    return sif_weights(list(state.docs), state.unigram, float(spec["sif_a"]))


def dist_channel(state: CorpusState, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """`v_dist` for one (weights, gamma, k) setting, plus its pooling statistics."""
    started = time.perf_counter()
    rows = weight_rows(state, spec)
    if rows is None:
        return np.array(state.full, copy=True), {
            "degenerate_documents": 0, "mean_token_weight": 1.0, "index_seconds": 0.0}
    matrix, stats = pool_weighted(list(state.docs), rows)
    if bool(spec.get("remove_pc")):
        matrix = remove_first_principal_component(matrix)
    return np.asarray(matrix, dtype=np.float64), {
        "degenerate_documents": int(stats["degenerate_documents"]),
        "mean_token_weight": float(np.mean([float(np.mean(row)) for row in rows])),
        "index_seconds": time.perf_counter() - started,
    }


def query_channel(state: CorpusState, query_docs: Sequence[Any], q_full: np.ndarray,
                  spec: dict[str, Any]) -> np.ndarray:
    """`q_dist` for the query side. A query has no neighbourhood, so df is GLOBAL.

    `query_side="mean"` keeps the mean-pooled query on both channels;
    `query_side="idf"` pools it with corpus-wide df at the same gamma, or with
    SIF weights when the document side is SIF. The LOCAL rule is deliberately not
    an option here: a query is not in the store and has no neighbours, so there
    is no `N(q)` to measure df against. Which variant produced a number is
    recorded on every cell.
    """
    kind, side = str(spec["weights"]), str(spec["query_side"])
    if kind == "uniform" or side == "mean":
        return q_full
    items = list(query_docs)
    rows = (sif_weights(items, state.unigram, float(spec["sif_a"])) if kind == "sif"
            else global_weights(items, state.global_df, float(spec["gamma"])))
    return np.asarray(pool_weighted(items, rows)[0], dtype=np.float64)


def geometry(full: np.ndarray, dist: np.ndarray, cluster_of: Sequence[str]) -> dict[str, float]:
    """Mean sibling and non-sibling cosine in BOTH stored channels."""
    cluster = np.asarray([str(value) for value in cluster_of])
    same = cluster[:, None] == cluster[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    out: dict[str, float] = {"n_sibling_pairs": float(int((same & upper).sum()))}
    for name, matrix in (("full", full), ("dist", dist)):
        values = np.asarray(matrix, dtype=np.float64)
        gram = values @ values.T
        sibling, other = gram[same & upper], gram[(~same) & upper]
        out[f"sibling_cosine_{name}"] = float(sibling.mean()) if sibling.size else float("nan")
        out[f"non_sibling_cosine_{name}"] = float(other.mean()) if other.size else float("nan")
    return out


def blended_geometry(base: dict[str, float], alpha: float) -> dict[str, float]:
    """Sibling geometry in the space the blended score is a dot product in."""
    a = float(alpha)
    sibling = a * base["sibling_cosine_full"] + (1.0 - a) * base["sibling_cosine_dist"]
    other = a * base["non_sibling_cosine_full"] + (1.0 - a) * base["non_sibling_cosine_dist"]
    return {"sibling_cosine": sibling, "non_sibling_cosine": other, "contrast": sibling - other}
