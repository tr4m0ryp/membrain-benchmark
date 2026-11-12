"""Construct a Mem0 instance out of purely local parts.

Every component that would otherwise reach a paid service is replaced by a local
one, and each replacement is recorded in the result so no number can be read as
"Mem0 as shipped":

* **LLM** -- the judge stub (`stub_llm`). Mem0's prompt building and response
  parsing are untouched; only the transport is.
* **Embedder** -- sentence-transformers `all-MiniLM-L6-v2`, the same model this
  project's own arms use, so the comparison is not confounded by embedding
  quality.
* **Vector store** -- FAISS on a local path, because it needs no server and can
  enumerate itself, which the dump requires.
* **Telemetry** -- off. A probe that phones home is not offline.

FAISS defaults to euclidean; cosine is set explicitly so the geometry matches
the rest of the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS = 384
COLLECTION = "brain_probe"


def offline_env() -> dict[str, str]:
    """Environment Mem0 must see. Applied before importing mem0."""
    return {
        "MEM0_TELEMETRY": "False",
        "ANONYMIZED_TELEMETRY": "False",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def apply_offline_env() -> None:
    for key, value in offline_env().items():
        os.environ.setdefault(key, value)


def memory_config(store_dir: Path, *, embed_model: str = EMBED_MODEL, llm_slot: str = "langchain") -> dict[str, Any]:
    """The config dict, recorded verbatim in the result file."""
    return {
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": COLLECTION,
                "path": str(store_dir / "faiss"),
                "embedding_model_dims": EMBED_DIMS,
                "distance_strategy": "cosine",
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embed_model, "embedding_dims": EMBED_DIMS},
        },
        "llm": {"provider": llm_slot, "config": {"model": "brain-stub"}},
        "history_db_path": str(store_dir / "history.db"),
    }


#: Loading a sentence-transformer per store would dominate the run, and the
#: model is stateless, so one instance is shared. Nothing in Mem0's logic
#: changes; only the constructor is memoised.
_MODELS: dict[tuple, Any] = {}


def _cache_sentence_transformer() -> None:
    import sentence_transformers

    original = sentence_transformers.SentenceTransformer
    if getattr(original, "_brain_cached", False):
        return

    def cached(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        try:
            hash(key)
        except TypeError:
            return original(*args, **kwargs)
        if key not in _MODELS:
            _MODELS[key] = original(*args, **kwargs)
        return _MODELS[key]

    cached._brain_cached = True  # type: ignore[attr-defined]
    sentence_transformers.SentenceTransformer = cached  # type: ignore[assignment]
    # Mem0's embedder module binds the class by name at import time, so patching
    # the package alone would be silently ineffective and every store would
    # reload the model from disk.
    try:
        import mem0.embeddings.huggingface as hf

        hf.SentenceTransformer = cached  # type: ignore[assignment]
    except ImportError:
        pass


def build_memory(store_dir: Path, judge: Any, *, embed_model: str = EMBED_MODEL) -> Any:
    """A fresh Mem0 over a fresh directory, driven by `judge`."""
    apply_offline_env()
    store_dir.mkdir(parents=True, exist_ok=True)
    _cache_sentence_transformer()

    import stub_llm

    slot = stub_llm.install(judge)

    from mem0 import Memory

    return Memory.from_config(memory_config(store_dir, embed_model=embed_model, llm_slot=slot))


def mem0_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mem0ai")
    except PackageNotFoundError:  # installed from a checkout
        import mem0

        return str(getattr(mem0, "__version__", "unknown"))


def component_report() -> dict[str, Any]:
    """What actually ran, for the provenance block of the result file."""
    report: dict[str, Any] = {
        "mem0ai": mem0_version(),
        "llm": "brain_stub (no model, no network); Mem0 prompts and parsing unchanged",
        "embedder": EMBED_MODEL,
        "vector_store": "faiss (local path, cosine)",
        "telemetry": "disabled",
    }
    for package in ("sentence_transformers", "faiss", "spacy", "numpy", "openai"):
        try:
            module = __import__(package)
            report[package] = str(getattr(module, "__version__", "present"))
        except ImportError:
            report[package] = None
    try:
        from mem0.utils.spacy_models import get_nlp_full

        report["spacy_pipeline_loaded"] = get_nlp_full() is not None
    except Exception:
        report["spacy_pipeline_loaded"] = False
    return report
