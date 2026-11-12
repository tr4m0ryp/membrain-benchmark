"""Comparison arms: plain vector, BM25, write-time MMR, full context (Task 006).

Every arm satisfies the same `Retriever` interface, so Task 007's harness and
Task 008's gate call them interchangeably:

    arm.name                       # str, the label a result is filed under
    arm.index(traces)              # Sequence[Trace] or corpus Episodes, once
    arm.retrieve(query, k)         # -> list[trace_id], best first, at most k

with three extras every arm also carries, used for provenance and for the
BLOCKED path rather than for retrieval: `retrieve_scored`, `describe()` and
`blocked_reason` / `is_available()`.

`ARM_BUILDERS` maps an arm name to a factory, which is exactly the
`retriever_factory` shape Task 007's `run_arm` consumes.

Imports are lazy so that `import brain.baselines` does not pull in torch or the
LLM transport before an arm is actually built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # import-free at runtime, still visible to type checkers
    from brain.baselines.base import (
        BaseRetriever,
        IndexingError,
        Retriever,
        as_traces,
        rank_hits,
    )
    from brain.baselines.bm25 import Bm25Retriever, tokenize
    from brain.baselines.full_context import FullContextRetriever
    from brain.baselines.llm import LlmResponse, LlmUnavailable, select_client
    from brain.baselines.smoke import run_smoke, smoke_traces
    from brain.baselines.vectors import (
        VectorCache,
        reset_shared_vector_cache,
        shared_vector_cache,
    )
    from brain.baselines.plain_vector import PlainVectorRetriever, VectorStoreArm
    from brain.baselines.write_time_mmr import DEFAULT_LAMBDA, WriteTimeMmrRetriever

_ORIGIN = {
    "Retriever": "base",
    "BaseRetriever": "base",
    "IndexingError": "base",
    "VectorCache": "vectors",
    "shared_vector_cache": "vectors",
    "reset_shared_vector_cache": "vectors",
    "smoke_traces": "smoke",
    "run_smoke": "smoke",
    "as_traces": "base",
    "rank_hits": "base",
    "PlainVectorRetriever": "plain_vector",
    "VectorStoreArm": "plain_vector",
    "Bm25Retriever": "bm25",
    "tokenize": "bm25",
    "WriteTimeMmrRetriever": "write_time_mmr",
    "DEFAULT_LAMBDA": "write_time_mmr",
    "DEFAULT_MODE": "write_time_mmr",
    "FullContextRetriever": "full_context",
    "LlmUnavailable": "llm",
    "LlmResponse": "llm",
    "select_client": "llm",
}

__all__ = [*sorted(_ORIGIN), "ARM_BUILDERS", "build_arm"]


def __getattr__(name: str) -> Any:
    module_name = _ORIGIN.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value  # cache the lookup, later access is a plain global
    return value


def __dir__() -> list[str]:
    return list(__all__)


def _plain_vector(**kwargs: Any) -> Any:
    from brain.baselines.plain_vector import PlainVectorRetriever

    return PlainVectorRetriever(**kwargs)


def _bm25(**kwargs: Any) -> Any:
    from brain.baselines.bm25 import Bm25Retriever

    return Bm25Retriever(**kwargs)


def _write_time_mmr(**kwargs: Any) -> Any:
    from brain.baselines.write_time_mmr import WriteTimeMmrRetriever

    return WriteTimeMmrRetriever(**kwargs)


def _full_context(**kwargs: Any) -> Any:
    from brain.baselines.full_context import FullContextRetriever

    return FullContextRetriever(**kwargs)


#: Arm name -> factory. A fresh arm (and a fresh store) per call, which is what
#: the harness needs for a clean run per seed.
ARM_BUILDERS: dict[str, Callable[..., Any]] = {
    "plain_vector": _plain_vector,
    "bm25": _bm25,
    "write_time_mmr": _write_time_mmr,
    "full_context": _full_context,
}


def build_arm(name: str, **kwargs: Any) -> Any:
    """Build one arm by name. Unknown names fail loudly, never silently."""
    try:
        builder = ARM_BUILDERS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; known arms: {sorted(ARM_BUILDERS)}") from None
    return builder(**kwargs)
