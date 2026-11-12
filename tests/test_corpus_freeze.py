"""Freeze tests: the manifest, the content hash, and the tamper check.

Split from `test_corpus.py` (generation and label invariants) to keep each file
inside the project's 300 line cap.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from brain.corpus import (
    Corpus,
    CorpusAlreadyFrozen,
    CorpusHashMismatch,
    CorpusNotFrozen,
    CorpusValidationError,
    GeneratorConfig,
    content_hashes,
    freeze_corpus,
    generate_corpus,
    load_corpus,
    validate_corpus,
)
from brain.corpus import freeze as freeze_module
from brain.corpus import llm as llm_module
from brain.corpus.types import serialize_records


@pytest.fixture(scope="module")
def small_config() -> GeneratorConfig:
    return GeneratorConfig(seed=7, cluster_sizes=(2, 3), clusters_per_size=(2, 2), label="test")


@pytest.fixture(scope="module")
def small_corpus(small_config: GeneratorConfig) -> Corpus:
    return generate_corpus(small_config)


def _freeze(tmp_path, corpus: Corpus, config: GeneratorConfig, **kwargs):
    return freeze_corpus(corpus, config, tmp_path, measure_embeddings=False, **kwargs)


def test_freeze_writes_a_complete_manifest(tmp_path, small_corpus, small_config) -> None:
    manifest = _freeze(tmp_path, small_corpus, small_config)
    assert manifest["seed"] == small_config.seed
    assert manifest["generator"]["similarity"] == small_config.similarity
    assert manifest["counts"]["episodes"] == len(small_corpus.episodes)
    assert manifest["counts"]["queries"] == len(small_corpus.queries)
    assert len(manifest["clusters"]) == len(small_corpus.clusters())
    assert manifest["similarity"]["lexical"]["sibling"]["mean"] > 0
    assert len(manifest["content_sha256"]) == 64
    for name in ("episodes.jsonl", "queries.jsonl", "manifest.json"):
        assert (tmp_path / name).is_file()


def test_round_trip_through_the_loader(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    loaded = load_corpus(tmp_path)
    assert [e.trace_id for e in loaded.episodes] == [e.trace_id for e in small_corpus.episodes]
    assert [e.text for e in loaded.episodes] == [e.text for e in small_corpus.episodes]
    assert loaded.queries[0].distractor_ids == small_corpus.queries[0].distractor_ids
    assert loaded.content_hash == content_hashes(small_corpus.episodes, small_corpus.queries)[
        "content_sha256"
    ]


def test_loader_detects_a_tampered_episode(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    episodes = list(small_corpus.episodes)
    episodes[0] = dataclasses.replace(episodes[0], text=episodes[0].text + " (edited)")
    (tmp_path / "episodes.jsonl").write_bytes(serialize_records(episodes, key="trace_id"))
    with pytest.raises(CorpusHashMismatch):
        load_corpus(tmp_path)
    assert load_corpus(tmp_path, verify=False) is not None


def test_loader_detects_a_tampered_query(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    queries = list(small_corpus.queries)
    queries[0] = dataclasses.replace(queries[0], correct_trace_id=queries[0].distractor_ids[0])
    (tmp_path / "queries.jsonl").write_bytes(serialize_records(queries, key="query_id"))
    with pytest.raises(CorpusHashMismatch):
        load_corpus(tmp_path)


def test_loader_requires_a_manifest(tmp_path) -> None:
    with pytest.raises(CorpusNotFrozen):
        load_corpus(tmp_path)


def test_refreezing_identical_content_is_a_no_op(tmp_path, small_corpus, small_config) -> None:
    first = _freeze(tmp_path, small_corpus, small_config)
    second = _freeze(tmp_path, small_corpus, small_config)
    assert second["created_at"] == first["created_at"]
    assert second["content_sha256"] == first["content_sha256"]


def test_refreezing_different_content_is_refused(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    other_config = dataclasses.replace(small_config, seed=small_config.seed + 1)
    other = generate_corpus(other_config)
    with pytest.raises(CorpusAlreadyFrozen):
        _freeze(tmp_path, other, other_config)
    forced = _freeze(tmp_path, other, other_config, force=True)
    assert load_corpus(tmp_path).content_hash == forced["content_sha256"]


def test_manifest_is_valid_json_with_sorted_keys(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    raw = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert list(parsed) == sorted(parsed)


class _StubClient:
    """Deterministic stand-in for the anthropic client."""

    def __init__(self, transform) -> None:
        self.transform = transform
        self.calls = 0

    def complete(self, *, model: str, temperature: float, max_tokens: int, prompt: str) -> str:
        self.calls += 1
        return self.transform(prompt)


def test_llm_rewrite_caches_and_keeps_facts(tmp_path, small_corpus) -> None:
    client = _StubClient(lambda prompt: "Note. " + prompt.split("NOTE:\n", 1)[1])
    rewritten, stats = llm_module.rewrite_corpus(
        small_corpus, cache_dir=tmp_path / "cache", client=client
    )
    assert stats["api_calls"] == len(small_corpus.episodes)
    assert stats["rejected"] == 0
    assert all(e.text.startswith("Note.") for e in rewritten.episodes)
    again, stats2 = llm_module.rewrite_corpus(
        small_corpus, cache_dir=tmp_path / "cache", client=client
    )
    assert stats2["cache_hits"] == len(small_corpus.episodes)
    assert stats2["api_calls"] == 0
    assert [e.text for e in again.episodes] == [e.text for e in rewritten.episodes]


def test_llm_rewrite_rejects_a_lossy_response(tmp_path, small_corpus) -> None:
    client = _StubClient(lambda prompt: "we fixed a thing in the file")
    rewritten, stats = llm_module.rewrite_corpus(
        small_corpus, cache_dir=tmp_path / "cache", client=client
    )
    assert stats["rejected"] == len(small_corpus.episodes)
    assert [e.text for e in rewritten.episodes] == [e.text for e in small_corpus.episodes]


def test_frozen_corpus_on_disk_verifies() -> None:
    """The committed corpus must load and match its manifest."""
    try:
        corpus = load_corpus()
    except CorpusNotFrozen:
        pytest.skip("no frozen corpus committed at data/corpus/synthetic")
    validate_corpus(corpus)
    assert corpus.manifest["counts"]["episodes"] == len(corpus.episodes)
    assert len(corpus.clusters()) == corpus.manifest["counts"]["clusters"]


def test_freeze_refuses_a_corpus_the_encoder_would_truncate() -> None:
    """The guard fires on a measured over-window count, whatever produced it."""
    with pytest.raises(CorpusValidationError):
        freeze_module._assert_fits_encoder(
            {"embedding": {"episode_tokens": {"over_window": 2, "window": 256, "max": 300}}}
        )
    freeze_module._assert_fits_encoder(
        {"embedding": {"episode_tokens": {"over_window": 0, "window": 256, "max": 184}}}
    )
