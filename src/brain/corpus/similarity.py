"""Measured difficulty of the corpus, in two geometries.

Two independent views, because either one alone can lie:

- `lexical`: TF-IDF cosine. Pure string overlap, no model, no downloads. This is
  the BM25-shaped view of the corpus.
- `embedding`: the shared `brain.embed` encoder, the geometry the gate actually
  runs in. Optional so the corpus stays generatable and testable with no model.

The numbers reported here calibrate the instrument. They are diagnostics ONLY:
they are excluded from the content hash so that recomputing them on another
machine cannot invalidate a frozen corpus. That also absorbs the ~1e-7 float
drift the encoder shows across different batch compositions.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

from brain.corpus.cues import verbatim_cue_query_texts
from brain.corpus.interference import collision_rates, fit_collision_rho
from brain.corpus.types import Corpus

__all__ = [
    "tokenize",
    "lexical_matrix",
    "shared_embedder",
    "embedding_matrix",
    "token_length_stats",
    "corpus_similarity_stats",
]

_TOKEN = re.compile(r"[a-z0-9_]+")
_EPSILON = 1e-12


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def lexical_matrix(texts: Sequence[str], *, vocabulary: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """L2-normalized TF-IDF rows over a sorted vocabulary.

    Sorted vocabulary, not insertion order: set iteration over strings is the one
    ordering the seed cannot pin (rules.md determinism addendum).
    """
    docs = [tokenize(text) for text in texts]
    if vocabulary is None:
        vocabulary = sorted({token for doc in docs for token in doc})
    index = {token: position for position, token in enumerate(vocabulary)}
    counts = np.zeros((len(docs), len(vocabulary)), dtype=np.float64)
    for row, doc in enumerate(docs):
        for token in doc:
            position = index.get(token)
            if position is not None:
                counts[row, position] += 1.0
    document_frequency = (counts > 0).sum(axis=0)
    idf = np.log((1.0 + len(docs)) / (1.0 + document_frequency)) + 1.0
    weighted = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, _EPSILON)), 0.0) * idf
    return _l2_normalize(weighted), list(vocabulary)


def shared_embedder() -> Any | None:
    """The process-wide `brain.embed` encoder, or None if it cannot load.

    Imported lazily: the encoder pulls in torch, which costs seconds, and
    neither generation nor the offline tests need it.
    """
    try:
        from brain.embed import get_embedder

        embedder = get_embedder()
        _ = embedder.dim  # forces the load, so a missing model cache fails here
        return embedder
    except Exception:  # pragma: no cover - depends on the local environment
        return None


def embedding_matrix(texts: Sequence[str], *, embedder: Any | None = None) -> np.ndarray | None:
    """Unit-norm embeddings for `texts`, or None when the encoder is unavailable.

    One call, one fixed order, so batch composition is fixed for the whole
    measurement.
    """
    embedder = embedder or shared_embedder()
    if embedder is None:
        return None
    try:
        return np.asarray(embedder.encode(list(texts)), dtype=np.float64)
    except Exception:  # pragma: no cover - offline cache miss, disk, etc.
        return None


def token_length_stats(texts: Sequence[str], embedder: Any) -> dict[str, Any]:
    """Wordpiece lengths against the backbone window.

    A trace over the window is truncated silently, which would delete the very
    fact that distinguishes it from its siblings, so `over_window` must be 0.
    """
    tokenizer = embedder.model.tokenizer
    lengths = [len(tokenizer.encode(text)) for text in texts]
    window = int(embedder.max_seq_length)
    summary = _summary(lengths)
    return {
        "window": window,
        "over_window": int(sum(1 for length in lengths if length > window)),
        **summary,
        "min": int(summary["min"]),
        "max": int(summary["max"]),
    }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, _EPSILON)


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std(ddof=1)) if array.size > 1 else 0.0, 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
    }


def _pairwise_stats(vectors: np.ndarray, cluster_ids: Sequence[str]) -> dict[str, Any]:
    """Intra- vs inter-cluster cosine. Intra >> inter means clusters are real."""
    gram = vectors @ vectors.T
    labels = np.asarray(cluster_ids)
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(gram, dtype=bool), k=1)
    intra = gram[same & upper]
    inter = gram[(~same) & upper]
    per_cluster: dict[str, float] = {}
    for cluster_id in sorted(set(cluster_ids)):
        mask = labels == cluster_id
        block = gram[np.ix_(mask, mask)]
        pairs = block[np.triu(np.ones_like(block, dtype=bool), k=1)]
        per_cluster[cluster_id] = round(float(pairs.mean()), 6) if pairs.size else math.nan
    return {
        "sibling": _summary(intra),
        "non_sibling": _summary(inter),
        "separation_index": round(float(intra.mean() - inter.mean()), 6)
        if intra.size and inter.size
        else math.nan,
        "per_cluster_sibling_mean": per_cluster,
    }


def _query_stats(
    query_vectors: np.ndarray, episode_vectors: np.ndarray, corpus: Corpus
) -> dict[str, Any]:
    """How hard the corpus is for raw similarity search, before any mechanism.

    `disambiguation_accuracy` here is the untuned baseline in this geometry. At
    1.0 the corpus is at ceiling and no mechanism could show a gain; at chance it
    is unanswerable. Both are instrument failures and are reported, not hidden.
    """
    row_of = {episode.trace_id: index for index, episode in enumerate(corpus.episodes)}
    scores = query_vectors @ episode_vectors.T
    correct_hits: list[float] = []
    margins: list[float] = []
    chance: list[float] = []
    by_size: dict[int, list[float]] = {}
    for row, query in enumerate(corpus.queries):
        correct = float(scores[row, row_of[query.correct_trace_id]])
        sibling_scores = [float(scores[row, row_of[tid]]) for tid in query.distractor_ids]
        best_sibling = max(sibling_scores)
        hit = 1.0 if correct > best_sibling else 0.0
        correct_hits.append(hit)
        margins.append(correct - best_sibling)
        chance.append(1.0 / (len(query.distractor_ids) + 1))
        by_size.setdefault(query.cluster_size, []).append(hit)
    return {
        "disambiguation_accuracy": round(float(np.mean(correct_hits)), 6),
        "chance_accuracy": round(float(np.mean(chance)), 6),
        "margin": _summary(margins),
        "accuracy_by_cluster_size": {
            str(size): round(float(np.mean(by_size[size])), 6) for size in sorted(by_size)
        },
    }


def corpus_similarity_stats(corpus: Corpus, *, measure_embeddings: bool = True) -> dict[str, Any]:
    """Difficulty report for one corpus, in every geometry available."""
    episode_texts = [e.text for e in corpus.episodes]
    query_texts = [q.text for q in corpus.queries]
    verbatim_texts = verbatim_cue_query_texts(corpus)
    cluster_ids = [e.cluster_id for e in corpus.episodes]
    bounds = (len(episode_texts), len(episode_texts) + len(query_texts))
    stats: dict[str, Any] = {}

    joint, vocabulary = lexical_matrix(episode_texts + query_texts + verbatim_texts)
    stats["lexical"] = {
        "method": "tfidf-cosine",
        "vocabulary_size": len(vocabulary),
        **_geometry_stats(joint, bounds, cluster_ids, corpus),
    }

    if not measure_embeddings:
        return stats

    embedder = shared_embedder()
    if embedder is None:
        stats["embedding"] = {"method": "cosine", "status": "unavailable"}
        return stats
    vectors = embedding_matrix(episode_texts + query_texts + verbatim_texts, embedder=embedder)
    if vectors is None:  # pragma: no cover - encoder loaded but failed to encode
        stats["embedding"] = {"method": "cosine", "status": "unavailable"}
        return stats
    stats["embedding"] = {
        "method": "cosine",
        "model": embedder.model_name,
        "dim": int(embedder.dim),
        "episode_tokens": token_length_stats(episode_texts, embedder),
        **_geometry_stats(vectors, bounds, cluster_ids, corpus),
    }
    return stats


def _geometry_stats(
    vectors: np.ndarray, bounds: tuple[int, int], cluster_ids: Sequence[str], corpus: Corpus
) -> dict[str, Any]:
    """Pairwise and query statistics for one geometry, given stacked vectors."""
    n_episodes, n_queries_end = bounds
    episode_vectors = vectors[:n_episodes]
    query_vectors = vectors[n_episodes:n_queries_end]
    verbatim_vectors = vectors[n_queries_end:]
    queries = _query_stats(query_vectors, episode_vectors, corpus)
    return {
        **_pairwise_stats(episode_vectors, cluster_ids),
        "collision_rates": collision_rates(episode_vectors, cluster_ids),
        "queries": queries,
        "queries_verbatim_cue": _query_stats(verbatim_vectors, episode_vectors, corpus),
        "m_a_fit": fit_collision_rho(queries["accuracy_by_cluster_size"]),
    }
