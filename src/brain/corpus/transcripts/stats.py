"""Real versus synthetic, on the three quantities the paper's decomposition needs.

This is the part of the pipeline that pays off WITHOUT any labels. The paper
writes a sibling vector as a shared component plus a distinctive one, treats the
distinctive parts as roughly orthogonal, and gets

    cos(v_i, v_j) = b^2 / (b^2 + (1 - b)^2),

where `b` is the fraction of the vector's mass carried by the shared component.
Inverting it, with r = sqrt(c / (1 - c)), gives b = r / (1 + r). The measured
synthetic sibling cosine of 0.824 implies b ~ 0.684. Since the paper's own
limitation is that generated episodes may share boilerplate more or less
uniformly than genuine transcripts, and since that shared-context term is what
reverses the direction of the supersession/interference ordering between regimes,
`b` measured on real transcripts is the single most load-bearing number available
before a human has labelled anything.

Two independent estimates of the shared fraction are reported, because agreeing
by two routes is the only evidence available that either is right:

- `beta_implied`, inverted from the measured sibling cosine, which is what the
  paper's model says.
- `token_overlap`, a lexical Jaccard over sibling pairs, which is what a reader
  can verify by eye and which the encoder cannot influence.

Siblings are defined per corpus in the way each corpus defines them: `cluster_id`
in the synthetic split, the changed file path in the real one. Both are "same
topic, different event", which is the property the statistic is about.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np

__all__ = ["implied_beta", "cosine_profile", "lexical_profile", "corpus_profile", "compare"]

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_MAX_PAIRS = 200_000


def implied_beta(cosine: float) -> float:
    """Invert cos = b^2 / (b^2 + (1-b)^2). Returns NaN outside (0, 1)."""
    if not 0.0 < cosine < 1.0:
        return math.nan
    ratio = math.sqrt(cosine / (1.0 - cosine))
    return round(ratio / (1.0 + ratio), 6)


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "p10": math.nan,
                "median": math.nan, "p90": math.nan}
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 6),
        "sd": round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6),
        "p10": round(float(np.quantile(values, 0.10)), 6),
        "median": round(float(np.median(values)), 6),
        "p90": round(float(np.quantile(values, 0.90)), 6),
    }


def _pair_indices(clusters: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(clusters)
    rows, cols = np.triu_indices(n, k=1)
    labels = np.asarray(clusters)
    same = labels[rows] == labels[cols]
    return rows, cols, same


def cosine_profile(vectors: np.ndarray, clusters: Sequence[str]) -> dict[str, Any]:
    """Sibling and non-sibling cosine, plus the shared fraction each implies."""
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    rows, cols, same = _pair_indices(clusters)
    # Index the n-by-n gram rather than gathering matrix[rows]: at 1850 episodes
    # the gather is a 1.7M-by-384 float64 array, about 5 GB, while the gram is 27 MB.
    gram_values = (matrix @ matrix.T)[rows, cols]
    sibling = gram_values[same]
    non_sibling = gram_values[~same]
    profile = {
        "sibling": _summary(sibling),
        "non_sibling": _summary(non_sibling),
    }
    profile["separation_index"] = (
        round(float(sibling.mean() - non_sibling.mean()), 6)
        if sibling.size and non_sibling.size
        else math.nan
    )
    profile["beta_implied"] = implied_beta(profile["sibling"]["mean"])
    profile["beta_implied_non_sibling"] = implied_beta(profile["non_sibling"]["mean"])
    return profile


def lexical_profile(texts: Sequence[str], clusters: Sequence[str]) -> dict[str, Any]:
    """Shared-boilerplate fraction, measured on tokens rather than on vectors.

    `token_overlap` is |T_i n T_j| / |T_i u T_j| over sibling pairs: the fraction
    of the pair's combined vocabulary that both episodes carry. `containment` is
    |T_i n T_j| / min(|T_i|, |T_j|), which is the more forgiving reading and is
    reported so a difference between the two cannot be hidden by picking one.
    """
    token_sets = [frozenset(_TOKEN.findall(text.lower())) for text in texts]
    rows, cols, same = _pair_indices(clusters)
    sibling_rows, sibling_cols = rows[same], cols[same]
    if sibling_rows.size > _MAX_PAIRS:  # deterministic head, never a random draw
        sibling_rows, sibling_cols = sibling_rows[:_MAX_PAIRS], sibling_cols[:_MAX_PAIRS]
    jaccard: list[float] = []
    containment: list[float] = []
    for i, j in zip(sibling_rows, sibling_cols):
        left, right = token_sets[i], token_sets[j]
        if not left or not right:
            continue
        shared = len(left & right)
        jaccard.append(shared / len(left | right))
        containment.append(shared / min(len(left), len(right)))
    return {
        "token_overlap": _summary(np.asarray(jaccard, dtype=np.float64)),
        "containment": _summary(np.asarray(containment, dtype=np.float64)),
        "vocabulary": int(len(set().union(*token_sets))) if token_sets else 0,
    }


def corpus_profile(
    texts: Sequence[str], clusters: Sequence[str], vectors: np.ndarray | None
) -> dict[str, Any]:
    """Length, lexical overlap and, when an encoder is available, cosine."""
    words = np.asarray([len(text.split()) for text in texts], dtype=np.float64)
    unique = len(set(texts))
    profile: dict[str, Any] = {
        "n_episodes": len(texts),
        "n_distinct_texts": unique,
        "exact_duplicate_fraction": (
            round(1.0 - unique / len(texts), 6) if texts else math.nan
        ),
        "n_clusters": len(set(clusters)),
        "words": _summary(words),
        "lexical": lexical_profile(texts, clusters),
    }
    if vectors is not None:
        profile["cosine"] = cosine_profile(vectors, clusters)
    return profile


def _delta(real: Any, synthetic: Any) -> Any:
    if isinstance(real, (int, float)) and isinstance(synthetic, (int, float)):
        if isinstance(real, bool) or isinstance(synthetic, bool):
            return None
        return round(float(real) - float(synthetic), 6)
    return None


def compare(real: dict[str, Any], synthetic: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side on the quantities the decomposition depends on."""
    def pick(profile: dict[str, Any], *path: str) -> Any:
        node: Any = profile
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return math.nan
            node = node[key]
        return node

    rows: dict[str, Any] = {}
    for name, path in (
        ("episode_words_mean", ("words", "mean")),
        ("episode_words_median", ("words", "median")),
        ("episode_words_sd", ("words", "sd")),
        ("sibling_cos_mean", ("cosine", "sibling", "mean")),
        ("sibling_cos_sd", ("cosine", "sibling", "sd")),
        ("sibling_cos_p10", ("cosine", "sibling", "p10")),
        ("sibling_cos_p90", ("cosine", "sibling", "p90")),
        ("non_sibling_cos_mean", ("cosine", "non_sibling", "mean")),
        ("separation_index", ("cosine", "separation_index")),
        ("beta_implied", ("cosine", "beta_implied")),
        ("token_overlap_mean", ("lexical", "token_overlap", "mean")),
        ("containment_mean", ("lexical", "containment", "mean")),
        ("n_episodes", ("n_episodes",)),
        ("n_clusters", ("n_clusters",)),
        ("exact_duplicate_fraction", ("exact_duplicate_fraction",)),
    ):
        real_value, synth_value = pick(real, *path), pick(synthetic, *path)
        rows[name] = {
            "real": real_value,
            "synthetic": synth_value,
            "delta": _delta(real_value, synth_value),
        }
    return rows
