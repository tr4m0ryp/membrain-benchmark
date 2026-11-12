"""Deterministic CPU embedding interface shared by every component.

Public surface: `Embedder` (local sentence-transformers, the default),
`VoyageEmbedder` (API fallback), `get_embedder` (shared instance per config),
and the vector hygiene helpers.

Names are re-exported lazily so that `import brain.embed` costs nothing until
something actually embeds, and so the smoke entry point
(`python -m brain.embed.embedder`) does not execute that module twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import-free at runtime, still visible to type checkers
    from brain.embed.embedder import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CACHE_SIZE,
        DEFAULT_MODEL,
        Embedder,
        get_embedder,
    )
    from brain.embed.vectors import as_text_list, l2_normalize
    from brain.embed.voyage import (
        VOYAGE_DEFAULT_BATCH_SIZE,
        VOYAGE_DEFAULT_MODEL,
        VoyageEmbedder,
    )

_ORIGIN = {
    "DEFAULT_MODEL": "embedder",
    "DEFAULT_BATCH_SIZE": "embedder",
    "DEFAULT_CACHE_SIZE": "embedder",
    "Embedder": "embedder",
    "get_embedder": "embedder",
    "VOYAGE_DEFAULT_MODEL": "voyage",
    "VOYAGE_DEFAULT_BATCH_SIZE": "voyage",
    "VoyageEmbedder": "voyage",
    "as_text_list": "vectors",
    "l2_normalize": "vectors",
}

__all__ = sorted(_ORIGIN)


def __getattr__(name: str) -> Any:
    module_name = _ORIGIN.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache the lookup, later access is a plain global
    return value


def __dir__() -> list[str]:
    return list(__all__)
