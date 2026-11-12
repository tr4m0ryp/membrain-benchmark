"""One embedding pass for every arm.

Re-encoding a subset per arm shifts the last ulps of float32 inference (batch
composition, about 1e-7 on this machine), which would hand one arm a slightly
different corpus than another. So the corpus is encoded once per process, here,
and every arm reads the same rows.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from brain.baselines.base import IndexingError

__all__ = ["VectorCache", "shared_vector_cache", "reset_shared_vector_cache"]


class VectorCache:
    """Text to embedding, computed once per process and shared by every arm.

    Keyed by text rather than by trace id: two arms must see bit-identical
    vectors for the same string, and the id is the arm's business. Missing
    texts are encoded in sorted order so the batch composition is a function of
    the missing SET, not of the order the caller happened to ask in.
    """

    def __init__(self, embedder: Any | None = None) -> None:
        self._embedder = embedder
        self._rows: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"VectorCache(cached={len(self._rows)})"

    @property
    def embedder(self) -> Any:
        """The process-wide embedder, loaded on first use."""
        if self._embedder is None:
            from brain.embed import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    @property
    def dim(self) -> int:
        """Embedding width, read from the embedder and never hardcoded."""
        return int(self.embedder.dim)

    def ensure(self, texts: Sequence[str]) -> None:
        """Encode whatever is not cached yet, in one batched call."""
        missing = sorted({text for text in texts if text not in self._rows})
        if not missing:
            return
        block = np.asarray(self.embedder.encode(missing), dtype=np.float32)
        if block.shape[0] != len(missing):
            raise IndexingError(f"embedder returned {block.shape[0]} rows for {len(missing)} texts")
        for text, row in zip(missing, block):
            frozen = np.array(row, dtype=np.float32, copy=True)
            frozen.flags.writeable = False  # a shared row must not be mutated in place
            self._rows[text] = frozen

    def matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Rows for `texts`, in the given order, as a fresh writable (n, d) array."""
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        self.ensure(items)
        return np.stack([self._rows[text] for text in items])

    def vector(self, text: str) -> np.ndarray:
        """One row, as a fresh writable (d,) array."""
        return self.matrix([text])[0]


_SHARED: VectorCache | None = None


def shared_vector_cache() -> VectorCache:
    """The process-wide cache. Arms default to it so the corpus is encoded once."""
    global _SHARED
    if _SHARED is None:
        _SHARED = VectorCache()
    return _SHARED


def reset_shared_vector_cache() -> None:
    """Drop the shared cache. For tests that inject a stub embedder."""
    global _SHARED
    _SHARED = None
