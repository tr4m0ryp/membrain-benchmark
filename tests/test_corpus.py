"""Corpus tests: reproducible generation, unambiguous labels, measured difficulty.

Freeze, hash and loader tests live in `test_corpus_freeze.py`.
"""

from __future__ import annotations

import dataclasses

import pytest

from brain.corpus import (
    MAX_EPISODE_WORDS,
    Corpus,
    CorpusValidationError,
    GeneratorConfig,
    content_hashes,
    corpus_similarity_stats,
    default_config,
    generate_corpus,
    validate_corpus,
    verbatim_cue_query_texts,
)
from brain.corpus.interference import collision_rates, fit_collision_rho
from brain.corpus.similarity import lexical_matrix


@pytest.fixture(scope="module")
def small_config() -> GeneratorConfig:
    return GeneratorConfig(seed=7, cluster_sizes=(2, 3), clusters_per_size=(2, 2), label="test")


@pytest.fixture(scope="module")
def small_corpus(small_config: GeneratorConfig) -> Corpus:
    return generate_corpus(small_config)


def test_generation_is_reproducible(small_config: GeneratorConfig) -> None:
    first = generate_corpus(small_config)
    second = generate_corpus(small_config)
    assert content_hashes(first.episodes, first.queries) == content_hashes(
        second.episodes, second.queries
    )


def test_a_different_seed_gives_different_content(small_config: GeneratorConfig) -> None:
    other = dataclasses.replace(small_config, seed=small_config.seed + 1)
    assert (
        content_hashes(generate_corpus(small_config).episodes, generate_corpus(small_config).queries)
        != content_hashes(generate_corpus(other).episodes, generate_corpus(other).queries)
    )


def test_counts_match_the_configuration(small_corpus: Corpus, small_config: GeneratorConfig) -> None:
    assert len(small_corpus.episodes) == small_config.n_episodes == 10
    assert len(small_corpus.clusters()) == small_config.n_clusters == 4
    assert len(small_corpus.queries) == small_config.n_episodes * small_config.queries_per_episode
    assert sorted(small_corpus.cluster_sizes().values()) == [2, 2, 3, 3]


def test_smoke_config_is_two_clusters_of_three() -> None:
    corpus = generate_corpus(default_config(smoke=True))
    assert len(corpus.clusters()) == 2
    assert sorted(corpus.cluster_sizes().values()) == [3, 3]


def test_every_query_has_one_correct_trace_and_real_distractors(small_corpus: Corpus) -> None:
    by_id = small_corpus.episode_by_id()
    for query in small_corpus.queries:
        assert query.correct_trace_id in by_id
        assert len(query.distractor_ids) >= 1
        assert query.correct_trace_id not in query.distractor_ids
        siblings = {
            episode.trace_id
            for episode in small_corpus.clusters()[query.cluster_id]
            if episode.trace_id != query.correct_trace_id
        }
        assert set(query.distractor_ids) == siblings
        assert query.cluster_size == len(siblings) + 1


def test_siblings_are_distinct_events_on_every_axis(small_corpus: Corpus) -> None:
    for members in small_corpus.clusters().values():
        assert len({m.topic["path"] for m in members}) == 1
        for axis in ("trigger", "diagnosis", "outcome"):
            values = [m.facts[axis] for m in members]
            assert len(set(values)) == len(values)
        assert len({m.date for m in members}) == len(members)


def test_query_cue_text_does_not_quote_the_episode(small_corpus: Corpus) -> None:
    """The cue is a paraphrase, so a query never contains a verbatim answer span."""
    by_id = small_corpus.episode_by_id()
    for query in small_corpus.queries:
        episode = by_id[query.correct_trace_id]
        for sentence in episode.text.split(". "):
            trimmed = sentence.strip().rstrip(".")
            if len(trimmed) > 30:
                assert trimmed not in query.text


def test_validation_rejects_a_query_pointing_at_a_sibling(small_corpus: Corpus) -> None:
    query = small_corpus.queries[0]
    broken = dataclasses.replace(query, correct_trace_id=query.distractor_ids[0])
    corpus = Corpus(
        episodes=list(small_corpus.episodes),
        queries=[broken] + list(small_corpus.queries[1:]),
    )
    with pytest.raises(CorpusValidationError):
        validate_corpus(corpus)


def test_validation_rejects_duplicate_episode_text(small_corpus: Corpus) -> None:
    episodes = list(small_corpus.episodes)
    episodes[1] = dataclasses.replace(episodes[1], text=episodes[0].text)
    with pytest.raises(CorpusValidationError):
        validate_corpus(Corpus(episodes=episodes, queries=list(small_corpus.queries)))


def test_cluster_of_one_is_rejected_by_config() -> None:
    with pytest.raises(ValueError):
        GeneratorConfig(cluster_sizes=(1,), clusters_per_size=(2,))


def test_similarity_knob_raises_measured_sibling_similarity() -> None:
    base = GeneratorConfig(seed=3, cluster_sizes=(4,), clusters_per_size=(6,), label="knob")
    low = corpus_similarity_stats(
        generate_corpus(dataclasses.replace(base, similarity=0.1)), measure_embeddings=False
    )
    high = corpus_similarity_stats(
        generate_corpus(dataclasses.replace(base, similarity=1.0)), measure_embeddings=False
    )
    assert high["lexical"]["sibling"]["mean"] > low["lexical"]["sibling"]["mean"]


def test_siblings_are_more_similar_than_non_siblings(small_corpus: Corpus) -> None:
    stats = corpus_similarity_stats(small_corpus, measure_embeddings=False)["lexical"]
    assert stats["sibling"]["mean"] > stats["non_sibling"]["mean"]
    assert stats["queries"]["disambiguation_accuracy"] >= stats["queries"]["chance_accuracy"]


def test_verbatim_cue_is_the_solvability_ceiling(small_corpus: Corpus) -> None:
    """Restoring the episode's own wording must make the query easier, not harder.

    If it did not, a missed query would mean an unanswerable question rather
    than interference between siblings, and the instrument would be invalid.
    """
    stats = corpus_similarity_stats(small_corpus, measure_embeddings=False)["lexical"]
    assert len(verbatim_cue_query_texts(small_corpus)) == len(small_corpus.queries)
    assert (
        stats["queries_verbatim_cue"]["disambiguation_accuracy"]
        >= stats["queries"]["disambiguation_accuracy"]
    )


def test_episode_length_stays_under_the_encoder_window(small_corpus: Corpus) -> None:
    """all-MiniLM-L6-v2 truncates at 256 wordpieces and would drop the tail."""
    for episode in small_corpus.episodes:
        assert len(episode.text.split()) <= MAX_EPISODE_WORDS


def test_validation_rejects_an_over_long_episode(small_corpus: Corpus) -> None:
    episodes = list(small_corpus.episodes)
    episodes[0] = dataclasses.replace(episodes[0], text=" ".join(["word"] * (MAX_EPISODE_WORDS + 1)))
    with pytest.raises(CorpusValidationError):
        validate_corpus(Corpus(episodes=episodes, queries=list(small_corpus.queries)))


def test_m_a_fit_recovers_a_known_rho() -> None:
    """Acc(m) = 1 / (1 + rho (m - 1)) must invert exactly on synthetic input."""
    rho = 0.25
    curve = {str(m): 1.0 / (1.0 + rho * (m - 1)) for m in (2, 3, 4, 6, 8)}
    fit = fit_collision_rho(curve)
    assert fit["rho"] == pytest.approx(rho, abs=1e-6)
    assert fit["max_abs_error"] == pytest.approx(0.0, abs=1e-6)


def test_m_a_fit_handles_degenerate_input() -> None:
    assert fit_collision_rho({})["n_points"] == 0
    assert fit_collision_rho({"1": 1.0, "4": 0.0})["n_points"] == 0


def test_collision_rate_is_monotone_in_tau(small_corpus: Corpus) -> None:
    vectors, _ = lexical_matrix([e.text for e in small_corpus.episodes])
    rows = collision_rates(vectors, [e.cluster_id for e in small_corpus.episodes])
    rates = [row["rho_all_pairs"] for row in rows]
    assert rates == sorted(rates, reverse=True)
    assert all(row["rho_sibling"] >= row["rho_non_sibling"] for row in rows)
