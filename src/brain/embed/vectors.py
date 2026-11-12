"""Input coercion and vector hygiene shared by every embedding backend.

One question: how do raw inputs and raw model output become a valid embedding
matrix. Kept separate from the backends so the local and API paths cannot drift
apart on normalization or on what counts as a degenerate vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["as_text_list", "l2_normalize"]


def as_text_list(texts: str | Sequence[str]) -> list[str]:
    """Accept a single string or any sequence of strings, reject anything else."""
    if isinstance(texts, str):
        return [texts]
    items = list(texts)
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(f"texts[{index}] is {type(item).__name__}, expected str")
    return items


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization of a 2-D array, returned as float32.

    Accumulating the norm in float64 and casting the quotient down costs one
    temporary and buys an exact float32 unit norm, so cosine equals dot product
    with no residual scale factor. Degenerate rows raise instead of silently
    producing NaN or an unnormalized vector.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {values.shape}")
    finite_rows = np.isfinite(values).all(axis=1)
    if not finite_rows.all():
        bad = np.flatnonzero(~finite_rows)
        raise ValueError(f"non-finite embedding values at rows {bad.tolist()}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    degenerate = np.flatnonzero(norms[:, 0] <= 0.0)
    if degenerate.size:
        raise ValueError(f"zero-norm embedding at rows {degenerate.tolist()}")
    return (values / norms).astype(np.float32, copy=False)
