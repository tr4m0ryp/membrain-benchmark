"""Optional Voyage API fallback (spec T3), off unless BRAIN_EMBEDDER=voyage.

Deliberately thin: the local model is the testbed default, and this path exists
so a run can be reproduced on a machine without the sentence-transformers stack.
It fails loudly on a missing key or client rather than degrading to a different
vector space behind the caller's back.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from brain.embed.vectors import as_text_list, l2_normalize

__all__ = ["VOYAGE_DEFAULT_MODEL", "VOYAGE_DEFAULT_BATCH_SIZE", "VoyageEmbedder"]

VOYAGE_DEFAULT_MODEL = "voyage-3.5-lite"
VOYAGE_DEFAULT_BATCH_SIZE = 128
_VOYAGE_DIMS = {"voyage-3.5-lite": 1024, "voyage-3.5": 1024, "voyage-3-large": 1024}


class VoyageEmbedder:
    """API encoder with the same surface as `Embedder`: `dim` and `encode`."""

    def __init__(
        self,
        model_name: str = VOYAGE_DEFAULT_MODEL,
        *,
        normalize: bool = True,
        api_key: str | None = None,
        input_type: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "VoyageEmbedder needs VOYAGE_API_KEY. Unset BRAIN_EMBEDDER to use "
                "the local sentence-transformers model instead."
            )
        self.model_name = str(model_name)
        self.normalize = bool(normalize)
        self.input_type = input_type
        self._api_key = key
        self._client: Any | None = None
        self._dim: int | None = _VOYAGE_DIMS.get(self.model_name)

    def __repr__(self) -> str:
        return f"VoyageEmbedder(model_name={self.model_name!r}, normalize={self.normalize})"

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import voyageai
            except ImportError as exc:
                raise RuntimeError(
                    "BRAIN_EMBEDDER=voyage requires the 'voyageai' package"
                ) from exc
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["dimension probe"]).shape[1])
        return self._dim

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int = VOYAGE_DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Embed `texts` into a float32 array of shape (n, dim)."""
        items = as_text_list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        size = max(1, int(batch_size))
        rows: list[Any] = []
        for start in range(0, len(items), size):
            response = self.client.embed(
                items[start : start + size],
                model=self.model_name,
                input_type=self.input_type,
            )
            rows.extend(response.embeddings)
        matrix = np.asarray(rows, dtype=np.float32).reshape(len(items), -1)
        self._dim = int(matrix.shape[1])
        return l2_normalize(matrix) if self.normalize else matrix

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single string into a 1-D float32 array of shape (dim,)."""
        if not isinstance(text, str):
            raise TypeError(f"text is {type(text).__name__}, expected str")
        return self.encode([text])[0]
