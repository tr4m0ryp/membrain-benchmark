"""Token-level extraction and weighted pooling, with mean pooling as the gamma = 0 case.

The whole module exists to make one claim checkable: `pool_weighted(docs, None)`
must return the vectors the testbed already stores, byte for byte. Everything
here is arranged so that it does.

**How the token embeddings are obtained without touching the frozen embedder.**
`SentenceTransformer.encode(..., output_value="token_embeddings")` runs exactly
the same length sort, the same batching and the same forward pass as the plain
call; it only changes which tensor is read out of `out_features`. So the token
embeddings handed back here are the ones that produced today's sentence vector,
not a re-derivation of it. `brain.embed.embedder` is not modified at all.

**Why the arithmetic is written the way it is.** The pooling module of
`all-MiniLM-L6-v2` computes `sum(h_t * m_t) / clamp(sum(m_t), 1e-9)` in float32
and the model's third module then applies `F.normalize(p=2, dim=1)`. This module
reproduces that operation order literally, with the attention mask replaced by
`m_t * w_t`. `output_value="token_embeddings"` already trims the padding, so the
mask is all ones over the returned rows and `m_t * w_t` is `w_t`. At `w_t = 1`
the two expressions are the same float program and the output is bit-identical,
which is asserted in `tests/test_ncp.py` against the shared `VectorCache`.

**Numerical stability.** Weights are rescaled by their maximum before pooling.
The pooled vector is invariant to a positive rescaling of `w` by construction
(it is a ratio of two sums that both carry the factor), and the rescaling keeps
`sum(w)` away from float32 underflow when a large gamma drives every weight to
1e-30. The rescaling is skipped when the maximum is already exactly 1.0, so the
gamma = 0 path is untouched. An episode whose every token is shared has an
all-zero weight row: rather than dividing by the 1e-9 clamp and emitting a
zero vector that normalization cannot rescue, it falls back to uniform weights
and is COUNTED, so a degenerate episode is visible in the result instead of
silently becoming noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from brain.embed.vectors import l2_normalize

__all__ = [
    "TOKEN_BATCH_SIZE",
    "TokenDoc",
    "TokenCache",
    "shared_token_cache",
    "reset_shared_token_cache",
    "mean_pool",
    "pool_weighted",
]

#: Must equal `brain.embed.embedder.DEFAULT_BATCH_SIZE`. Batch composition moves
#: the last ulps of float32 inference, so a different value here would break the
#: byte-identity claim rather than merely slowing things down.
TOKEN_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class TokenDoc:
    """One text: its tokenizer ids and the contextual embedding of each token.

    `ids` and `embeddings` are aligned position by position, which is checked at
    construction time: the tokenizer is called separately from the forward pass,
    so a silent disagreement about truncation would misattribute every weight.
    """

    text: str
    ids: np.ndarray  # int64, shape (L,)
    embeddings: Any  # torch.Tensor, float32, shape (L, D)

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[0])

    @property
    def types(self) -> frozenset[int]:
        """The distinct token ids in this document (presence, not count)."""
        return frozenset(int(value) for value in self.ids.tolist())


class TokenCache:
    """Text to `TokenDoc`, computed once per process and shared by every arm.

    Mirrors `brain.baselines.vectors.VectorCache` deliberately, including the
    sorted-missing-set rule: the batch fed to the model is a function of the
    missing SET and not of the order a caller happened to ask in, which is what
    makes two arms see identical rows.
    """

    def __init__(self, embedder: Any | None = None, *, batch_size: int = TOKEN_BATCH_SIZE) -> None:
        self._embedder = embedder
        self._batch_size = int(batch_size)
        self._docs: dict[str, TokenDoc] = {}

    def __len__(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return f"TokenCache(cached={len(self._docs)}, batch_size={self._batch_size})"

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from brain.embed import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    @property
    def dim(self) -> int:
        return int(self.embedder.dim)

    def ensure(self, texts: Sequence[str]) -> int:
        """Encode whatever is not cached yet, in one batched call. Returns the count."""
        missing = sorted({text for text in texts if text not in self._docs})
        if not missing:
            return 0
        import torch

        model = self.embedder.model  # load outside the inference context
        tokenizer = model.tokenizer
        max_length = int(model.max_seq_length)
        with torch.inference_mode():
            blocks = model.encode(
                missing,
                batch_size=self._batch_size,
                output_value="token_embeddings",
                convert_to_numpy=False,
                show_progress_bar=False,
            )
        if len(blocks) != len(missing):
            raise ValueError(f"embedder returned {len(blocks)} blocks for {len(missing)} texts")
        for text, block in zip(missing, blocks):
            encoded = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
            ids = np.asarray(encoded, dtype=np.int64)
            if int(ids.shape[0]) != int(block.shape[0]):
                raise ValueError(
                    f"tokenizer produced {ids.shape[0]} ids but the forward pass returned "
                    f"{int(block.shape[0])} token embeddings for text {text[:40]!r}"
                )
            self._docs[text] = TokenDoc(text=text, ids=ids, embeddings=block)
        return len(missing)

    def doc(self, text: str) -> TokenDoc:
        if text not in self._docs:
            self.ensure([text])
        return self._docs[text]

    def docs(self, texts: Sequence[str]) -> list[TokenDoc]:
        items = list(texts)
        self.ensure(items)
        return [self._docs[text] for text in items]


_SHARED: TokenCache | None = None


def shared_token_cache() -> TokenCache:
    """The process-wide token cache. Arms default to it so a text is encoded once."""
    global _SHARED
    if _SHARED is None:
        _SHARED = TokenCache()
    return _SHARED


def reset_shared_token_cache() -> None:
    """Drop the shared cache. For tests that inject a stub embedder."""
    global _SHARED
    _SHARED = None


def pool_weighted(
    docs: Sequence[TokenDoc], weights: Sequence[np.ndarray] | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Weighted mean over token embeddings, normalised, as float32 unit rows.

    `weights[i][t]` is `w_t^(i)` from the formal model. `None` means uniform,
    which is mean pooling and therefore today's vectors exactly.
    """
    import torch
    import torch.nn.functional as F

    items = list(docs)
    if not items:
        return np.zeros((0, 0), dtype=np.float32), {"degenerate_documents": 0, "documents": 0}
    if weights is not None and len(weights) != len(items):
        raise ValueError(f"{len(items)} documents but {len(weights)} weight rows")
    rows: list[Any] = []
    degenerate = 0
    with torch.inference_mode():
        for index, doc in enumerate(items):
            embeddings = doc.embeddings
            if int(embeddings.shape[0]) == 0:
                raise ValueError(f"document {index} has no tokens")
            if weights is None:
                w = torch.ones(int(embeddings.shape[0]), dtype=embeddings.dtype)
            else:
                w = _weight_tensor(weights[index], embeddings, index)
                peak = float(w.max())
                if not peak > 0.0:
                    # Every token is shared with every neighbour. Uniform is the
                    # honest fallback (it is the gamma = 0 answer) and the count
                    # is reported so the cell cannot look clean when it is not.
                    degenerate += 1
                    w = torch.ones(int(embeddings.shape[0]), dtype=embeddings.dtype)
                elif peak != 1.0:
                    w = w / peak
            numerator = (embeddings * w.unsqueeze(-1)).sum(dim=0)
            denominator = w.sum().clamp(min=1e-9)
            rows.append(numerator / denominator)
        matrix = F.normalize(torch.stack(rows), p=2, dim=1)
    pooled = l2_normalize(np.asarray(matrix.numpy(), dtype=np.float32))
    return pooled, {"degenerate_documents": int(degenerate), "documents": len(items)}


def _weight_tensor(row: Any, embeddings: Any, index: int) -> Any:
    import torch

    values = np.asarray(row, dtype=np.float32).reshape(-1)
    if values.shape[0] != int(embeddings.shape[0]):
        raise ValueError(
            f"document {index}: {values.shape[0]} weights for {int(embeddings.shape[0])} tokens"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"document {index}: non-finite token weight")
    if float(values.min()) < 0.0:
        raise ValueError(f"document {index}: negative token weight")
    return torch.as_tensor(values, dtype=embeddings.dtype)


def mean_pool(docs: Sequence[TokenDoc]) -> np.ndarray:
    """Uniform pooling: the gamma = 0 case, and today's stored vectors."""
    return pool_weighted(docs, None)[0]
