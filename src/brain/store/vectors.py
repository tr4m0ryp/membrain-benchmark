"""Vector marshalling for the store: validate once, hand sqlite-vec raw bytes.

Little-endian float32 is the on-disk vector format sqlite-vec expects, so a
contiguous array of `VECTOR_DTYPE` serializes with `tobytes()` and no
per-element Python packing.

Validation lives here rather than in the store so a batch is checked in one
vectorized pass instead of once per row on the write path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "VECTOR_DTYPE",
    "InvalidVectorError",
    "DimensionMismatchError",
    "prepare",
    "from_blob",
]

VECTOR_DTYPE = np.dtype("<f4")


class InvalidVectorError(ValueError):
    """Vector is unusable: wrong shape, not finite, or has no direction."""


class DimensionMismatchError(InvalidVectorError):
    """Vector length does not match the dimension the store was opened with."""


def prepare(vector: Any, dim: int, *, rows: int | None = None) -> np.ndarray:
    """Cast, check and return a contiguous float32 array.

    `rows` is set on the batch path, where the argument is an (n, d) block and
    every check runs once over the whole block.
    """
    try:
        arr = np.ascontiguousarray(np.asarray(vector, dtype=VECTOR_DTYPE))
    except (TypeError, ValueError) as exc:
        raise InvalidVectorError(f"not a numeric vector: {exc}") from exc
    want_ndim = 1 if rows is None else 2
    if arr.ndim != want_ndim:
        raise InvalidVectorError(
            f"expected a {want_ndim}-D array of width {dim}, "
            f"got shape {tuple(arr.shape)}"
        )
    if arr.shape[-1] != dim:
        raise DimensionMismatchError(f"expected dim {dim}, got {arr.shape[-1]}")
    if rows is not None and arr.shape[0] != rows:
        raise InvalidVectorError(f"got {arr.shape[0]} vectors for {rows} traces")
    # Cast-then-check also catches a float64 magnitude that overflows float32.
    if not np.isfinite(arr).all():
        raise InvalidVectorError("vector is not finite (NaN or inf)")
    # sqlite-vec divides by the float32 norm, so a squared norm that underflows
    # to zero would yield a NaN distance. Refuse the vector here instead.
    sq = np.einsum("...i,...i->...", arr, arr)
    if not np.all(np.isfinite(sq)) or not np.all(sq > 0.0):
        raise InvalidVectorError("vector has zero norm in float32, no direction")
    return arr


def from_blob(blob: bytes) -> np.ndarray:
    """Decode a stored vector into a fresh writable float32 array."""
    return np.frombuffer(blob, dtype=VECTOR_DTYPE).copy()
