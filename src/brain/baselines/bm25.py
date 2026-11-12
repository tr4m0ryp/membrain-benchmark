"""BM25 lexical arm (rank-bm25 0.2.2), the non-semantic control.

Kept strong on the two axes that decide whether a lexical baseline is honest:

* **Parameters at the standard defaults.** BM25Okapi k1 = 1.5, b = 0.75,
  epsilon = 0.25, exactly as shipped, recorded in `params`. Nothing is tuned
  against this corpus in either direction. Okapi is also the best of the three
  variants rank-bm25 ships, measured on the frozen corpus (disambiguation
  0.636 Okapi, 0.625 Plus, 0.503 L), so the standard choice and the strongest
  choice coincide.
* **Tokenization that suits the domain.** The corpus is engineering session
  notes, where the discriminative tokens are paths, identifiers and numbers
  (`platform/featureflags/eval.go`, `PR #9166`, `1.9`). Whitespace splitting
  leaves `eval.go,` glued to its punctuation and costs a lot: measured 0.531
  against 0.636 for alphanumeric runs. The default is therefore alphanumeric
  runs, which recovers every path component as a token. A code-search style
  variant that also keeps each compound whole is available as
  `tokenizer="compound"` and measures marginally lower (0.631), because the
  repeated tokens inflate document length; it is kept so the comparison is
  reproducible, not because it is better.

No stemming and no stop list: BM25 already discounts ubiquitous terms through
IDF, and a stemmer would add an unpinned dependency to a reproducible build.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from brain.baselines.base import BaseRetriever, rank_hits
from brain.store import Trace

__all__ = ["Bm25Retriever", "tokenize", "tokenize_with_compounds", "TOKENIZERS", "K1", "B", "EPSILON"]

#: rank-bm25 BM25Okapi defaults, restated here so a result records them.
K1 = 1.5
B = 0.75
EPSILON = 0.25

_ALNUM = re.compile(r"[a-z0-9]+")
#: A run of alphanumerics, optionally joined by the separators that hold code
#: identifiers and paths together.
_COMPOUND = re.compile(r"[a-z0-9]+(?:[._/\\-][a-z0-9]+)*")
_SPLIT = re.compile(r"[._/\\-]")

#: A document with no tokens breaks the average-length term in BM25, so an
#: empty document gets one placeholder token that no query can produce.
_EMPTY_DOC_TOKEN = "\x00empty"


def tokenize(text: str) -> list[str]:
    """Lowercase runs of alphanumerics. The default (see the module docstring).

    `platform/featureflags/eval.go` becomes `platform`, `featureflags`, `eval`,
    `go`, and `PR #9166` becomes `pr`, `9166`, so paths, identifiers and
    numbers all stay matchable.
    """
    return _ALNUM.findall(text.lower())


def tokenize_with_compounds(text: str) -> list[str]:
    """Alphanumeric runs plus each compound kept whole (`tokenizer="compound"`).

    Emits `platform/featureflags/eval.go` as well as its parts, so an exact
    path repeat scores twice. Measured marginally weaker than `tokenize`.
    """
    tokens: list[str] = []
    for compound in _COMPOUND.findall(text.lower()):
        tokens.append(compound)
        parts = [part for part in _SPLIT.split(compound) if part]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


TOKENIZERS = {"alnum": tokenize, "compound": tokenize_with_compounds}


class Bm25Retriever(BaseRetriever):
    """BM25 Okapi over trace text. No embeddings, no store.

    Deliberately independent of the vector store: this arm measures what pure
    lexical overlap achieves on the interference corpus, which is the floor the
    semantic arms have to clear.
    """

    name = "bm25"

    def __init__(
        self,
        *,
        k1: float = K1,
        b: float = B,
        epsilon: float = EPSILON,
        tokenizer: str = "alnum",
        name: str | None = None,
    ) -> None:
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {sorted(TOKENIZERS)}, got {tokenizer!r}")
        super().__init__(
            name=name,
            params={
                "variant": "BM25Okapi",
                "k1": float(k1),
                "b": float(b),
                "epsilon": float(epsilon),
                "tokenizer": tokenizer,
                "stemming": False,
                "stopwords": False,
            },
        )
        self._k1 = float(k1)
        self._b = float(b)
        self._epsilon = float(epsilon)
        self._tokenize = TOKENIZERS[tokenizer]
        self._model: Any | None = None
        self._ids: list[str] = []

    def _index(self, traces: Sequence[Trace]) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = [trace.id for trace in traces]
        documents = [self._tokenize(trace.text) or [_EMPTY_DOC_TOKEN] for trace in traces]
        empty = sum(1 for doc in documents if doc == [_EMPTY_DOC_TOKEN])
        self.stats["empty_documents"] = empty
        self.stats["mean_tokens_per_doc"] = (
            float(np.mean([len(doc) for doc in documents])) if documents else 0.0
        )
        self._model = (
            BM25Okapi(documents, k1=self._k1, b=self._b, epsilon=self._epsilon)
            if documents
            else None
        )

    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._model is None:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            # No lexical evidence at all: report zeros rather than inventing an
            # order, and let the shared tie-break (ascending id) decide.
            self.stats["empty_queries"] = int(self.stats.get("empty_queries", 0)) + 1
            return rank_hits(((trace_id, 0.0) for trace_id in self._ids), k)
        scores = np.asarray(self._model.get_scores(tokens), dtype=np.float64)
        return rank_hits(zip(self._ids, scores.tolist()), k)


def _smoke() -> None:
    """Tiny CPU path: python -m brain.baselines.bm25."""
    from brain.baselines.smoke import run_smoke

    with Bm25Retriever() as arm:
        run_smoke(arm)


if __name__ == "__main__":
    _smoke()
