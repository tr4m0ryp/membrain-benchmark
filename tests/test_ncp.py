"""NCP contract. The load-bearing test is the first one: gamma = 0 must be today's vectors.

Weighted pooling is a strict generalisation of mean pooling only if the identity
holds exactly, so `test_gamma_zero_is_byte_identical_to_stored_vectors` is not a
nicety: it is a free correctness check on the token extraction, the alignment
between tokenizer ids and token embeddings, the weight function, the pooling
arithmetic and the normalisation, all at once. If it passes, none of those five
can be silently wrong.

Everything here is offline: the local encoder, the frozen corpus, no LLM
transport is imported or reachable.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.baselines.base import rank_hits
from brain.baselines.ncp import NcpRetriever
from brain.baselines.plain_vector import PlainVectorRetriever
from brain.baselines.smoke import SMOKE_QUERY, smoke_traces
from brain.baselines.vectors import reset_shared_vector_cache, shared_vector_cache
from brain.corpus import load_corpus
from brain.corpus.relations.loader import SYNTHETIC_CONTENT_SHA256
from brain.embed.pooling import (
    TokenDoc,
    distinctiveness,
    global_document_frequency,
    global_weights,
    local_document_frequency,
    local_weights,
    mean_pool,
    neighbour_indices,
    pool_weighted,
    remove_first_principal_component,
    reset_shared_token_cache,
    shared_token_cache,
    sif_weights,
    token_sets,
    unigram_probabilities,
)
from brain.experiments.gate_ncp.runner import (
    ALPHA_GRID,
    GAMMA_GRID,
    SUBSAMPLE_N,
    cell_specs,
    rank_ids,
    reordering,
    subsample_ids,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def texts(corpus) -> list[str]:
    return [e.text for e in sorted(corpus.episodes, key=lambda e: str(e.trace_id))]


@pytest.fixture(scope="module", autouse=True)
def _fixed_batch_composition(texts):
    """Both process-wide caches, emptied and refilled from the same batch.

    The identity tests below compare the NCP pooling path against the stored
    vectors byte for byte, and byte-identity holds only within a fixed batch
    composition: cross-batch float32 rounding is ~1e-7 and is intrinsic to the
    encoder. Resetting only one of the two caches is not enough, because then
    the reference side is whatever an earlier test left behind and the candidate
    side is fresh. Reset both, then warm both from `texts` in this one order.
    """
    from brain.embed.embedder import _shared_embedder

    # The embedder too, not just the caches. Both sides of the identity are
    # recomputed here, and a model instance that has already run a different
    # batch does not reproduce the same float32 bytes on this one; the ~1e-7
    # difference is intrinsic and documented. Clearing the factory's cache
    # gives both sides a pristine instance and the same batch.
    _shared_embedder.cache_clear()
    reset_shared_token_cache()
    reset_shared_vector_cache()
    shared_vector_cache().ensure(texts)
    shared_token_cache().ensure(texts)
    yield
    _shared_embedder.cache_clear()
    reset_shared_token_cache()
    reset_shared_vector_cache()


@pytest.fixture(scope="module")
def docs(texts):
    return shared_token_cache().docs(texts)


# -- the identity --------------------------------------------------------------
def test_gamma_zero_is_byte_identical_to_stored_vectors(texts, docs):
    """Mean pooling through the NCP path reproduces the shared vector cache exactly."""
    reference = shared_vector_cache().matrix(texts)
    pooled = mean_pool(docs)
    assert pooled.dtype == reference.dtype
    assert pooled.shape == reference.shape
    assert pooled.tobytes() == reference.tobytes()


def test_gamma_zero_through_the_general_weight_path_is_also_byte_identical(texts, docs):
    """Not a special case: the weights are computed and they are all exactly 1.0."""
    reference = shared_vector_cache().matrix(texts)
    order = neighbour_indices(np.asarray(mean_pool(docs), dtype=np.float64), 8)
    rows = local_weights(docs, token_sets(docs), order, 0.0)
    assert all(float(row.min()) == 1.0 and float(row.max()) == 1.0 for row in rows)
    pooled, stats = pool_weighted(docs, rows)
    assert pooled.tobytes() == reference.tobytes()
    assert stats == {"degenerate_documents": 0, "documents": len(texts)}


def test_global_weights_at_gamma_zero_are_also_the_identity(texts, docs):
    rows = global_weights(docs, global_document_frequency(token_sets(docs)), 0.0)
    assert pool_weighted(docs, rows)[0].tobytes() == shared_vector_cache().matrix(texts).tobytes()


# -- the weight function -------------------------------------------------------
def test_distinctiveness_matches_the_formal_model():
    df = np.array([0.0, 0.5, 1.0])
    assert np.allclose(distinctiveness(df, 1.0), [1.0, 0.5, 0.0])
    assert np.allclose(distinctiveness(df, 2.0), [1.0, 0.25, 0.0])
    # gamma = 0 is exactly 1 everywhere, including where df is exactly 1.
    assert (distinctiveness(df, 0.0) == 1.0).all()
    with pytest.raises(ValueError):
        distinctiveness(df, -1.0)


def test_local_df_counts_presence_not_repetition():
    doc = TokenDoc("a", np.array([7, 7, 7, 9], dtype=np.int64), None)
    sets = [frozenset({7}), frozenset({9}), frozenset({7, 9}), frozenset({1})]
    df = local_document_frequency(doc, sets, [0, 1, 2, 3])
    # token 7 is in neighbours 0 and 2 -> 2/4; token 9 in 1 and 2 -> 2/4.
    assert df.tolist() == [0.5, 0.5, 0.5, 0.5]
    assert df.shape[0] == doc.n_tokens


def test_local_df_of_an_empty_neighbourhood_is_zero():
    doc = TokenDoc("a", np.array([1, 2], dtype=np.int64), None)
    assert local_document_frequency(doc, [], []).tolist() == [0.0, 0.0]


def test_neighbour_indices_excludes_self_and_breaks_ties_by_index():
    vectors = np.eye(4)
    order = neighbour_indices(vectors, 2)
    assert order.shape == (4, 2)
    assert all(row not in order[row].tolist() for row in range(4))
    # every off-diagonal cosine is 0, so ties must resolve to ascending index.
    assert order[1].tolist() == [0, 2]
    assert neighbour_indices(vectors, 99).shape == (4, 3)  # clamped to n - 1
    assert neighbour_indices(np.eye(1), 5).shape == (1, 0)


def test_sif_weights_decrease_with_frequency(docs):
    probabilities = unigram_probabilities(docs)
    common = max(probabilities, key=lambda t: probabilities[t])
    rare = min(probabilities, key=lambda t: probabilities[t])
    doc = TokenDoc("x", np.array([common, rare], dtype=np.int64), None)
    weights = sif_weights([doc], probabilities, 1e-3)[0]
    assert weights[0] < weights[1]


# -- pooling: stability and the degenerate case --------------------------------
def test_all_shared_document_falls_back_to_uniform_and_is_counted(docs):
    subject = docs[:2]
    rows = [np.zeros(doc.n_tokens, dtype=np.float64) for doc in subject]
    pooled, stats = pool_weighted(subject, rows)
    assert stats["degenerate_documents"] == 2
    assert np.allclose(np.linalg.norm(pooled.astype(np.float64), axis=1), 1.0)
    assert pooled.tobytes() == mean_pool(subject).tobytes()


def test_vanishing_weights_stay_finite_and_unit_norm(docs):
    subject = docs[:4]
    order = neighbour_indices(np.asarray(mean_pool(subject), dtype=np.float64), 3)
    rows = local_weights(subject, token_sets(subject), order, 200.0)
    pooled, _ = pool_weighted(subject, rows)
    assert np.isfinite(pooled).all()
    assert np.allclose(np.linalg.norm(pooled.astype(np.float64), axis=1), 1.0, atol=1e-6)


def test_pool_weighted_rejects_bad_weights(docs):
    subject = docs[:1]
    for bad in (np.full(subject[0].n_tokens, -1.0), np.full(subject[0].n_tokens, np.nan),
                np.ones(subject[0].n_tokens + 1)):
        with pytest.raises(ValueError):
            pool_weighted(subject, [bad])
    assert pool_weighted([], None)[1]["documents"] == 0


def test_remove_first_principal_component_keeps_unit_rows(docs):
    pooled = mean_pool(docs[:20])
    projected = remove_first_principal_component(pooled)
    assert np.allclose(np.linalg.norm(projected.astype(np.float64), axis=1), 1.0, atol=1e-6)


# -- the arm -------------------------------------------------------------------
def test_gamma_zero_arm_ranks_exactly_like_plain_vector():
    traces = smoke_traces()
    with NcpRetriever(gamma=0.0, alpha=0.5, k_neighbours=3) as ncp, \
            PlainVectorRetriever() as control:
        ncp.index(traces)
        control.index(traces)
        assert ncp.retrieve(SMOKE_QUERY, 4) == control.retrieve(SMOKE_QUERY, 4)


def test_alpha_one_is_plain_vector_at_every_gamma():
    traces = smoke_traces()
    with PlainVectorRetriever() as control:
        control.index(traces)
        expected = control.retrieve(SMOKE_QUERY, 4)
    for gamma in (0.5, 2.0, 8.0):
        with NcpRetriever(gamma=gamma, alpha=1.0, k_neighbours=3) as arm:
            arm.index(traces)
            assert arm.retrieve(SMOKE_QUERY, 4) == expected


def test_gamma_zero_is_alpha_invariant():
    traces = smoke_traces()
    scores = []
    for alpha in ALPHA_GRID:
        with NcpRetriever(gamma=0.0, alpha=alpha, k_neighbours=3) as arm:
            arm.index(traces)
            scores.append(arm.scores(SMOKE_QUERY))
    for row in scores[1:]:
        assert np.allclose(row, scores[0], atol=1e-12)


def test_arm_rejects_invalid_configuration():
    for kwargs in ({"gamma": -1.0}, {"alpha": 1.5}, {"weights": "nope"},
                   {"query_side": "nope"}):
        with pytest.raises(ValueError):
            NcpRetriever(**kwargs)


def test_arm_makes_no_llm_call_and_declares_it():
    arm = NcpRetriever()
    assert arm.params["llm_calls_per_query"] == 0
    assert not any("llm" in key or "cost" in key for key in arm.stats)
    assert arm.blocked_reason is None


def test_dist_channel_differs_from_full_once_gamma_is_positive():
    traces = smoke_traces()
    with NcpRetriever(gamma=2.0, alpha=0.5, k_neighbours=3) as arm:
        arm.index(traces)
        _, full, dist = arm.channels()
    assert not np.allclose(full, dist)
    assert np.allclose(np.linalg.norm(dist, axis=1), 1.0, atol=1e-6)


# -- the experiment plumbing ---------------------------------------------------
def test_rank_ids_matches_the_shared_tie_break():
    ids = ["a", "b", "c", "d"]
    scores = np.array([[0.5, 0.5, 0.1, 0.9]])
    assert rank_ids(scores, ids, 3)[0] == [t for t, _ in rank_hits(zip(ids, scores[0]), 3)]


def test_reordering_against_itself_is_zero():
    scores = np.random.default_rng(0).normal(size=(5, 6))
    members = [np.array([0, 1, 2]) for _ in range(5)]
    out = reordering(scores, scores, members)
    assert out["sibling_pairs_reordered"] == 0
    assert out["sibling_pairs_compared"] == 15


def test_subsample_matches_the_veto_experiments_rule(corpus):
    from brain.experiments.gate_veto.report import _subsample

    picked = subsample_ids(corpus.queries, SUBSAMPLE_N)
    assert len(picked) == SUBSAMPLE_N
    assert picked == [str(q.query_id) for q in _subsample(corpus, SUBSAMPLE_N).queries]


def test_preregistered_grid_is_the_documented_one():
    specs = cell_specs()
    assert GAMMA_GRID[0] == 0.0 and ALPHA_GRID[-1] == 1.0
    assert {s["weights"] for s in specs} == {"local", "global", "sif"}
    assert len(specs) == 3 * len(GAMMA_GRID) * 2 + len(GAMMA_GRID) * 2 + 2 * 2 * 2


def test_frozen_corpus_is_unchanged(corpus):
    assert corpus.content_hash == SYNTHETIC_CONTENT_SHA256
