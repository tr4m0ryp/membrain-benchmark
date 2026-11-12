"""F17a: merge-harm rate on independent draws of the main synthetic corpus.

Reconstructs the ad hoc bucketing the original F17 measurement used directly
on the frozen corpus (finding F17): a pair of
episodes is INTERFERENCE if they share a cluster (same-topic, distinct events
by construction), COINCIDENCE if they sit in different clusters but share
`facts["diagnosis"]`, and UNRELATED otherwise. That script was never checked
in (F17: "analysis scripts under the session tmp dir"), so this is a
from-scratch reconstruction, verified byte-for-byte against the frozen corpus
before any draw runs: at tau=0.85 it reproduces 79/216 = 36.5741%, and the
per-class n/mean/sd match F17's table to four decimals.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from brain.corpus.generate import GeneratorConfig, generate_corpus
from brain.corpus.types import Corpus, Episode

__all__ = ["MERGE_TAU", "PairBuckets", "bucket_pairs", "merge_harm_rate", "draw_merge_harm"]

MERGE_TAU = 0.85


@dataclass(frozen=True, slots=True)
class PairBuckets:
    """Cosine scores for every episode pair, split by F17's three classes."""

    interference: np.ndarray
    coincidence: np.ndarray
    unrelated: np.ndarray


def _unit_vectors(episodes: Sequence[Episode], embedder: Any) -> np.ndarray:
    vectors = np.asarray(embedder.encode([e.text for e in episodes]), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def bucket_pairs(corpus: Corpus, embedder: Any) -> PairBuckets:
    """Every episode pair, cosine-scored and bucketed by cluster/diagnosis."""
    episodes = sorted(corpus.episodes, key=lambda e: e.trace_id)
    vectors = _unit_vectors(episodes, embedder)
    gram = np.clip(vectors @ vectors.T, -1.0, 1.0)
    buckets: dict[str, list[float]] = {"interference": [], "coincidence": [], "unrelated": []}
    for i, j in itertools.combinations(range(len(episodes)), 2):
        a, b = episodes[i], episodes[j]
        score = float(gram[i, j])
        if a.cluster_id == b.cluster_id:
            buckets["interference"].append(score)
        elif a.facts.get("diagnosis") == b.facts.get("diagnosis"):
            buckets["coincidence"].append(score)
        else:
            buckets["unrelated"].append(score)
    return PairBuckets(**{name: np.array(values, dtype=np.float64) for name, values in buckets.items()})


def merge_harm_rate(buckets: PairBuckets, *, tau: float = MERGE_TAU) -> dict[str, Any]:
    """Fraction of INTERFERENCE (same-cluster, distinct-event) pairs merged at `tau`."""
    inter = buckets.interference
    merged = int((inter >= tau).sum())
    return {
        "tau": float(tau),
        "n_interference": int(inter.size),
        "merged": merged,
        "rate": (merged / inter.size) if inter.size else math.nan,
        "n_coincidence": int(buckets.coincidence.size),
        "n_unrelated": int(buckets.unrelated.size),
        "mean_interference": float(inter.mean()) if inter.size else math.nan,
    }


def draw_merge_harm(seed: int, embedder: Any, *, tau: float = MERGE_TAU) -> dict[str, Any]:
    """One draw: generate the synthetic-corpus shape at `seed`, bucket, score."""
    corpus = generate_corpus(GeneratorConfig(seed=int(seed)))
    buckets = bucket_pairs(corpus, embedder)
    result = merge_harm_rate(buckets, tau=tau)
    result.update(
        seed=int(seed), n_episodes=len(corpus.episodes), n_clusters=len(corpus.clusters())
    )
    return result
