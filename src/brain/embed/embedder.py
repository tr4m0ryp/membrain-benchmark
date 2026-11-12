"""Deterministic, CPU-only text embedding: the one vector interface in the testbed.

Every component (separation, baselines, corpus, eval) embeds through this module
so that a reported number depends on exactly one embedding function.

Design notes that are load-bearing for reproducibility:

* The model is loaded lazily on first use, so `import brain` stays fast and does
  not touch the network or the Hugging Face cache.
* The global seeds are set via `brain.config.seed_everything` immediately before
  the model is constructed, and never again. Reseeding inside `encode` would be
  a hidden side effect on a caller's own random stream (the separation module
  draws random projections between encodes), which is a worse failure than the
  one it would guard against. Inference itself is randomness-free: the model is
  in eval mode inside `torch.inference_mode()`, so no dropout and no autograd.
* Guarantee: for a fixed input sequence, seed and process, `encode` returns
  byte-identical vectors on every call and across runs. What is NOT guaranteed
  is invariance to batch composition: a text encoded alone and the same text
  encoded next to a longer neighbour differ by about 1e-7 (float32 rounding in
  padded attention and in GEMM blocking, measured on this machine). That is
  intrinsic to batched transformer inference, not something the cache
  introduces, and it is orders of magnitude below any effect the experiments
  measure. Thread count does not change the values (verified 1 vs 2 threads).
* Normalization is a single vectorized pass done in float64 and cast back to
  float32, which lands the float32 norm on exactly 1.0 rather than 1 - 6e-8.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from brain.config import SEED, SMOKE_TEST, seed_everything
from brain.embed.vectors import as_text_list, l2_normalize
from brain.embed.voyage import VOYAGE_DEFAULT_MODEL, VoyageEmbedder

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE_SIZE",
    "Embedder",
    "get_embedder",
]

#: Testbed default. Overrides the spec's Ollama/nomic pick (T3) because Ollama
#: is not installed on this machine; local-first and CPU-only still hold.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: Fixed on purpose, and deliberately NOT reduced under SMOKE_TEST: batch size
#: shifts the last ulps of float32 inference, so making it depend on an env flag
#: would mean the smoke run and the real run produce different bytes for the
#: same input. Tiny-N is expressed as fewer texts, never as a smaller batch.
DEFAULT_BATCH_SIZE = 32
DEFAULT_CACHE_SIZE = 4096


class Embedder:
    """Local sentence-transformers encoder, CPU, lazily loaded.

    `cache_size=0` disables the in-process cache. The cache is keyed by text and
    scoped to the instance, whose identity already carries the model name and
    the normalize flag, so the effective key is (model, normalize, text).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        normalize: bool = True,
        *,
        seed: int | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.model_name = str(model_name)
        self.device = str(device)
        self.normalize = bool(normalize)
        self.seed = SEED if seed is None else int(seed)
        self.cache_size = max(0, int(cache_size))
        self._model: Any | None = None
        self._dim: int | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __repr__(self) -> str:
        return (
            f"Embedder(model_name={self.model_name!r}, device={self.device!r}, "
            f"normalize={self.normalize}, seed={self.seed})"
        )

    @property
    def model(self) -> Any:
        """The underlying SentenceTransformer, loaded on first access."""
        if self._model is None:
            self._load()
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimension (384 for the default model)."""
        if self._dim is None:
            self._load()
        return int(self._dim or 0)

    @property
    def max_seq_length(self) -> int:
        """Token budget of the backbone; longer texts are truncated (256 here).

        Exposed so the corpus and experiment tasks can check that a trace fits
        instead of silently losing its tail.
        """
        return int(self.model.max_seq_length)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        # Seed before construction so any init-time randomness is pinned too.
        seed_everything(self.seed)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=self.device)
        model.eval()
        # ST 5.x renamed the accessor; keep working under both spellings.
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:
            getter = model.get_sentence_embedding_dimension
        self._model = model
        self._dim = int(getter())

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Embed `texts` into a float32 array of shape (n, dim).

        Unit-norm rows when `normalize` is set. Duplicate texts inside one call
        are computed once. The returned array is always a fresh copy, so the
        caller may mutate it without corrupting the cache.
        """
        items = as_text_list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.cache_size == 0:
            return self._compute(items, batch_size)

        lookup: dict[str, np.ndarray] = {}
        for text in items:
            row = self._cache.get(text)
            if row is not None:
                lookup[text] = row
                self._cache.move_to_end(text)
        # dict.fromkeys keeps first-seen order, so the batch fed to the model is
        # a deterministic function of the input order (no set iteration anywhere).
        missing = [t for t in dict.fromkeys(items) if t not in lookup]
        if missing:
            fresh = self._compute(missing, batch_size)
            for text, row in zip(missing, fresh):
                lookup[text] = self._store(text, row)
        return np.stack([lookup[t] for t in items])

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single string into a 1-D float32 array of shape (dim,)."""
        if not isinstance(text, str):
            raise TypeError(f"text is {type(text).__name__}, expected str")
        return self.encode([text])[0]

    def clear_cache(self) -> None:
        self._cache.clear()

    def _store(self, text: str, row: np.ndarray) -> np.ndarray:
        cached = np.array(row, dtype=np.float32, copy=True)
        cached.flags.writeable = False
        self._cache[text] = cached
        self._cache.move_to_end(text)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def _compute(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch

        model = self.model  # load outside the inference context
        size = max(1, int(batch_size))
        # One model call for the whole list; SentenceTransformer does the
        # mini-batching internally, so this is ceil(n / size) forward passes,
        # never one per text.
        with torch.inference_mode():
            raw = model.encode(
                texts,
                batch_size=size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        matrix = np.asarray(raw, dtype=np.float32).reshape(len(texts), -1)
        if self.normalize:
            return l2_normalize(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("embedder produced non-finite values")
        return matrix


@lru_cache(maxsize=None)
def _shared_embedder(
    backend: str, model_name: str, device: str, normalize: bool, seed: int, cache_size: int
) -> Embedder | VoyageEmbedder:
    if backend == "voyage":
        return VoyageEmbedder(model_name, normalize=normalize)
    return Embedder(
        model_name, device=device, normalize=normalize, seed=seed, cache_size=cache_size
    )


def get_embedder(
    model_name: str | None = None,
    *,
    device: str = "cpu",
    normalize: bool = True,
    seed: int | None = None,
    cache_size: int = DEFAULT_CACHE_SIZE,
    backend: str | None = None,
) -> Embedder | VoyageEmbedder:
    """Return the process-wide embedder for these settings.

    Instances are shared so that separate baseline arms hit one warm cache
    instead of re-encoding the same corpus once per arm.
    """
    chosen = (backend or os.environ.get("BRAIN_EMBEDDER") or "local").strip().lower()
    if chosen not in {"local", "voyage"}:
        raise ValueError(f"unknown embedder backend {chosen!r}, expected local or voyage")
    default_name = VOYAGE_DEFAULT_MODEL if chosen == "voyage" else DEFAULT_MODEL
    return _shared_embedder(
        chosen,
        model_name or default_name,
        device,
        bool(normalize),
        SEED if seed is None else int(seed),
        int(cache_size),
    )


def _smoke() -> None:
    """Tiny CPU path: python -m brain.embed.embedder."""
    samples = [
        "the deploy failed",
        "the deploy failed",
        "the migration lock was held by a stale worker",
        "unrelated note about a person",
    ]
    texts = samples[:2] if SMOKE_TEST else samples
    embedder = Embedder()
    vectors = embedder.encode(texts)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    print(f"model={embedder.model_name} dim={embedder.dim} max_seq={embedder.max_seq_length}")
    print(f"shape={vectors.shape} dtype={vectors.dtype} norms={norms.tolist()}")
    print(f"dup rows identical: {vectors[0].tobytes() == vectors[1].tobytes()}")
    print(f"repeat identical: {embedder.encode(texts).tobytes() == vectors.tobytes()}")


if __name__ == "__main__":
    _smoke()
