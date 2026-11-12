"""Token weights: the local (neighbourhood) rule, and the global incumbents it must beat.

The formal model, symbol for symbol (task 008):

    df_t^(i)  =  |{ j in N(i) : t appears in x_j }| / |N(i)|
    w_t^(i)   =  (1 - df_t^(i)) ** gamma

`N(i)` is episode i's k nearest neighbours in the store, computed on the
mean-pooled vectors, i excluded. `gamma = 0` sends every weight to exactly 1.0
(`x ** 0.0 == 1.0` for every finite x, including 0.0), which is what makes the
whole construction a strict generalisation of mean pooling rather than a new
encoder.

**Three readings are fixed here because each one moves the number.**

1. **Presence, not count.** `t appears in x_j` is set membership, so a token
   repeated five times in a neighbour counts once for that neighbour. Inside
   episode i every occurrence of t carries the same `w_t`, so a repeated token
   still contributes proportionally more mass, exactly as under mean pooling.
2. **The token is the model's token.** Weights are keyed by tokenizer id, so
   subword pieces are weighted individually. That is the faithful reading: the
   contextual embeddings being pooled are indexed by those same pieces, and
   re-aggregating to whole words would weight a vector by a key it does not have.
   `[CLS]` and `[SEP]` are in every document, so their local df is 1.0 and they
   are suppressed at any gamma > 0, which is the correct behaviour and not a
   special case.
3. **An empty neighbourhood gives df = 0.** A store with one episode has nothing
   to be distinctive against, so every weight is 1 and the encoder degrades to
   mean pooling instead of dividing by zero.

**The incumbent.** Weighted pooling is not new (Arora et al 2017, SIF and uSIF).
The claimed delta of NCP is that the weights are conditioned on the LOCAL
neighbourhood rather than on global corpus statistics, so both global forms live
here as first-class baselines: `global_weights` is the same functional form on
corpus-wide df (the exact ablation, one variable changed), and `sif_weights` is
SIF's own `a / (a + p(t))` with the optional common-component removal.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from brain.embed.pooling.encode import TokenDoc

__all__ = [
    "SIF_DEFAULT_A",
    "token_sets",
    "neighbour_indices",
    "local_document_frequency",
    "distinctiveness",
    "local_weights",
    "global_document_frequency",
    "global_weights",
    "unigram_probabilities",
    "sif_weights",
    "remove_first_principal_component",
]

#: Arora et al's reported working range is 1e-3 to 1e-4; 1e-3 is their default.
SIF_DEFAULT_A = 1e-3


def token_sets(docs: Sequence[TokenDoc]) -> list[frozenset[int]]:
    """Per document, the set of token ids it contains. Presence, not count."""
    return [doc.types for doc in docs]


def neighbour_indices(vectors: np.ndarray, k: int) -> np.ndarray:
    """Row indices of each vector's k nearest neighbours by cosine, self excluded.

    Rows are assumed unit norm (they come from `pool_weighted`), so the gram
    matrix is the cosine matrix. Ties break by ascending index through a stable
    sort, because two equally near neighbours must not be ordered by whatever
    argsort's pivot chose (rules.md: sort ids before iterating).
    """
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {values.shape}")
    n = int(values.shape[0])
    width = max(0, min(int(k), n - 1))
    if n == 0 or width == 0:
        return np.zeros((n, 0), dtype=np.int64)
    gram = values @ values.T
    np.fill_diagonal(gram, -np.inf)  # never a neighbour of itself
    order = np.argsort(-gram, axis=1, kind="stable")
    return np.asarray(order[:, :width], dtype=np.int64)


def local_document_frequency(
    doc: TokenDoc, sets: Sequence[frozenset[int]], neighbours: Sequence[int]
) -> np.ndarray:
    """`df_t^(i)` for every token POSITION of `doc`, in float64."""
    members = [int(index) for index in neighbours]
    if not members:
        return np.zeros(doc.n_tokens, dtype=np.float64)
    size = float(len(members))
    counts: dict[int, int] = {}
    for token in doc.types:
        counts[token] = sum(1 for index in members if token in sets[index])
    return np.array([counts[int(t)] / size for t in doc.ids.tolist()], dtype=np.float64)


def distinctiveness(df: np.ndarray, gamma: float) -> np.ndarray:
    """`(1 - df) ** gamma`, clipped into [0, 1] before the power.

    The clip only absorbs float error at the endpoints: df is a ratio of counts
    and cannot leave [0, 1] mathematically. At gamma = 0 this returns exactly
    1.0 everywhere, including where df is exactly 1.0.
    """
    exponent = float(gamma)
    if exponent < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma!r}")
    base = np.clip(1.0 - np.asarray(df, dtype=np.float64), 0.0, 1.0)
    if exponent == 0.0:
        return np.ones_like(base)
    return np.power(base, exponent)


def local_weights(
    docs: Sequence[TokenDoc],
    sets: Sequence[frozenset[int]],
    neighbours: np.ndarray,
    gamma: float,
) -> list[np.ndarray]:
    """`w_t^(i)` for every document: the neighbourhood-conditional rule."""
    rows: list[np.ndarray] = []
    for index, doc in enumerate(docs):
        members = neighbours[index] if index < len(neighbours) else np.zeros(0, dtype=np.int64)
        df = local_document_frequency(doc, sets, members.tolist())
        rows.append(distinctiveness(df, gamma))
    return rows


def global_document_frequency(
    sets: Sequence[frozenset[int]],
) -> dict[int, float]:
    """Corpus-wide document frequency per token id. The global-IDF ablation's input."""
    total = float(len(sets))
    if total <= 0.0:
        return {}
    counts: dict[int, int] = {}
    for members in sets:
        for token in sorted(members):  # sorted: iteration order must not matter
            counts[int(token)] = counts.get(int(token), 0) + 1
    return {token: counts[token] / total for token in sorted(counts)}


def global_weights(
    docs: Sequence[TokenDoc], frequencies: Mapping[int, float], gamma: float
) -> list[np.ndarray]:
    """`(1 - df_global) ** gamma`: NCP's functional form on corpus statistics.

    This is the ablation that isolates the claim. Only one thing differs from
    `local_weights`: where df is measured. A token never seen in the corpus gets
    df = 0, i.e. maximally distinctive, which is the right reading for a query.
    """
    rows: list[np.ndarray] = []
    for doc in docs:
        df = np.array([float(frequencies.get(int(t), 0.0)) for t in doc.ids.tolist()])
        rows.append(distinctiveness(df, gamma))
    return rows


def unigram_probabilities(docs: Sequence[TokenDoc]) -> dict[int, float]:
    """`p(t)`: token occurrence probability over the corpus. SIF's input."""
    counts: dict[int, int] = {}
    total = 0
    for doc in docs:
        for token in doc.ids.tolist():
            counts[int(token)] = counts.get(int(token), 0) + 1
            total += 1
    if total <= 0:
        return {}
    return {token: counts[token] / float(total) for token in sorted(counts)}


def sif_weights(
    docs: Sequence[TokenDoc], probabilities: Mapping[int, float], a: float = SIF_DEFAULT_A
) -> list[np.ndarray]:
    """Arora et al's smooth inverse frequency weight `a / (a + p(t))`.

    Reported in the results table as the named incumbent, not as a strawman: it
    is the published weighting, at the published default, with the published
    common-component removal available separately.
    """
    alpha = float(a)
    if not alpha > 0.0:
        raise ValueError(f"SIF a must be positive, got {a!r}")
    rows: list[np.ndarray] = []
    for doc in docs:
        p = np.array([float(probabilities.get(int(t), 0.0)) for t in doc.ids.tolist()])
        rows.append(alpha / (alpha + np.clip(p, 0.0, None)))
    return rows


def remove_first_principal_component(matrix: np.ndarray) -> np.ndarray:
    """SIF's second step: project out the top singular direction, then renormalise.

    Computed in float64 on the mean-centred matrix and cast back to float32, the
    same discipline the embedder uses for its own normalisation. Returns the
    input unchanged when there is nothing to project (fewer than two rows).
    """
    from brain.embed.vectors import l2_normalize

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {values.shape}")
    if values.shape[0] < 2:
        return np.asarray(matrix, dtype=np.float32)
    centred = values - values.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    direction = right[0]
    projected = values - np.outer(values @ direction, direction)
    norms = np.linalg.norm(projected, axis=1)
    if not np.all(norms > 0.0):
        # Removing the common component annihilated a row. Keeping the original
        # row is the only answer that stays a unit vector, and it is rare enough
        # that silently returning zeros would be the worse failure.
        projected = np.where(norms[:, None] > 0.0, projected, values)
    return l2_normalize(projected)
