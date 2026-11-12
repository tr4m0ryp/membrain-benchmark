"""Frozen synthetic interference corpus and its manifest (Task 004).

Clusters of similar-but-DISTINCT coding-session episodes, plus queries that can
only be answered by telling cluster members apart. Ground truth is programmatic:
generation controls which episode answers which query, so the correct-trace label
and the sibling distractors come for free.

Typical use:

    from brain.corpus import load_corpus
    corpus = load_corpus()          # verifies the content hash, raises on drift
"""

from brain.corpus.cues import verbatim_cue_query_texts
from brain.corpus.freeze import CorpusAlreadyFrozen, build_manifest, freeze_corpus
from brain.corpus.generate import GeneratorConfig, default_config, generate_corpus
from brain.corpus.loader import (
    SYNTHETIC_DIR,
    CorpusHashMismatch,
    CorpusNotFrozen,
    content_hashes,
    load_corpus,
)
from brain.corpus.similarity import corpus_similarity_stats
from brain.corpus.types import SCHEMA_VERSION, Corpus, Episode, Query
from brain.corpus.validate import MAX_EPISODE_WORDS, CorpusValidationError, validate_corpus

__all__ = [
    "SCHEMA_VERSION",
    "SYNTHETIC_DIR",
    "MAX_EPISODE_WORDS",
    "Corpus",
    "Episode",
    "Query",
    "GeneratorConfig",
    "CorpusValidationError",
    "CorpusHashMismatch",
    "CorpusNotFrozen",
    "CorpusAlreadyFrozen",
    "default_config",
    "generate_corpus",
    "validate_corpus",
    "verbatim_cue_query_texts",
    "corpus_similarity_stats",
    "build_manifest",
    "freeze_corpus",
    "content_hashes",
    "load_corpus",
]
