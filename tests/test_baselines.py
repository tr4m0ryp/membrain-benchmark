"""Baseline arms: the uniform interface, each arm's fair best, and determinism.

The LLM arm has its own file (`test_baselines_full_context.py`); what is checked
here is that it satisfies the same interface as the local arms and returns the
same ids twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brain.baselines import ARM_BUILDERS, build_arm
from brain.baselines.base import IndexingError, Retriever, as_traces, rank_hits
from brain.baselines.bm25 import Bm25Retriever, tokenize, tokenize_with_compounds
from brain.baselines.plain_vector import PlainVectorRetriever
from brain.baselines.vectors import VectorCache
from brain.baselines.write_time_mmr import OBJECTIVE, PROJECTION, WriteTimeMmrRetriever
from brain.store import Trace
from tests.baseline_arms import QUERY, all_arms, full_context, traces  # noqa: F401

# -- the uniform interface -------------------------------------------------


def test_every_arm_satisfies_the_retriever_protocol(tmp_path: Path, traces: list[Trace]) -> None:
    for arm in all_arms(tmp_path):
        with arm:
            assert isinstance(arm, Retriever)
            assert isinstance(arm.name, str) and arm.name
            arm.index(traces)
            assert arm.describe()["params"] is not None


@pytest.mark.parametrize("k", [1, 3, 5])
def test_each_arm_returns_k_ids(tmp_path: Path, traces: list[Trace], k: int) -> None:
    known = {trace.id for trace in traces}
    for arm in all_arms(tmp_path):
        with arm:
            arm.index(traces)
            hits = arm.retrieve(QUERY, k)
            assert len(hits) == k, arm.name
            assert len(set(hits)) == k, arm.name  # no duplicates
            assert set(hits) <= known, arm.name


def test_arm_registry_builds_every_arm(tmp_path: Path) -> None:
    assert set(ARM_BUILDERS) == {"plain_vector", "bm25", "write_time_mmr", "full_context"}
    arm = build_arm("bm25")
    assert arm.name == "bm25"
    with pytest.raises(KeyError):
        build_arm("does_not_exist")


def test_guards(tmp_path: Path, traces: list[Trace]) -> None:
    arm = Bm25Retriever()
    with pytest.raises(IndexingError):
        arm.retrieve(QUERY, 3)  # before index
    arm.index(traces)
    assert arm.retrieve(QUERY, 0) == []
    assert arm.retrieve(QUERY, -1) == []
    assert len(arm.retrieve(QUERY, 999)) == len(traces)  # capped at corpus size
    with pytest.raises(TypeError):
        arm.retrieve(None, 3)  # type: ignore[arg-type]


def test_duplicate_ids_are_refused(traces: list[Trace]) -> None:
    arm = Bm25Retriever()
    with pytest.raises(IndexingError, match="duplicate trace ids"):
        arm.index([traces[0], traces[0]])


def test_byte_identical_texts_would_shrink_the_corpus_and_raise(traces: list[Trace]) -> None:
    # content_sha is UNIQUE in the store, so a repeated text collapses. The arm
    # must fail loudly instead of silently scoring a smaller corpus.
    twin = Trace(text=traces[0].text, id="twin", cluster_id="c0")
    with PlainVectorRetriever() as arm:
        with pytest.raises(IndexingError, match="deduped"):
            arm.index([traces[0], twin])


def test_as_traces_accepts_episode_like_records() -> None:
    class Episode:
        trace_id, text, cluster_id, date = "e9", "a note", "c9", "2026-01-21"

    converted = as_traces([Episode()])
    assert (converted[0].id, converted[0].cluster_id) == ("e9", "c9")
    assert converted[0].ts == pytest.approx(1768953600.0)


def test_rank_hits_breaks_ties_by_ascending_id() -> None:
    assert rank_hits([("b", 1.0), ("a", 1.0), ("c", 2.0)], 3) == [("c", 2.0), ("a", 1.0), ("b", 1.0)]


# -- plain vector ----------------------------------------------------------


def test_plain_vector_finds_the_exact_match_trace(traces: list[Trace]) -> None:
    with PlainVectorRetriever() as arm:
        arm.index(traces)
        target = traces[2]
        hits = arm.retrieve_scored(target.text, 3)
        assert hits[0][0] == target.id
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)  # cosine of a vector with itself


def test_plain_vector_stores_the_embedding_unchanged(traces: list[Trace]) -> None:
    with PlainVectorRetriever() as arm:
        arm.index(traces)
        stored = arm.store.get_vector(traces[0].id)
        assert stored is not None
        expected = arm.cache.vector(traces[0].text)
        assert stored.tobytes() == expected.tobytes()  # s = 0 is bit-exact


# -- BM25 ------------------------------------------------------------------


def test_bm25_tokenizers() -> None:
    text = "Fixed platform/featureflags/eval.go in PR #9166"
    tokens = tokenize(text)
    assert {"platform", "featureflags", "eval", "go", "pr", "9166"} <= set(tokens)
    assert "platform/featureflags/eval.go" not in tokens  # the default splits paths
    assert "platform/featureflags/eval.go" in tokenize_with_compounds(text)
    assert tokenize("") == []
    with pytest.raises(ValueError):
        Bm25Retriever(tokenizer="nope")


def test_bm25_ranks_the_rare_lexical_match_first(traces: list[Trace]) -> None:
    with Bm25Retriever() as arm:
        arm.index(traces)
        assert arm.retrieve("exhausted connection pool", 1) == ["c0-e1"]
        assert arm.params["k1"] == 1.5 and arm.params["b"] == 0.75  # stock defaults


def test_bm25_with_no_query_tokens_is_deterministic(traces: list[Trace]) -> None:
    with Bm25Retriever() as arm:
        arm.index(traces)
        assert arm.retrieve("...", 3) == sorted(trace.id for trace in traces)[:3]
        assert arm.stats["empty_queries"] == 1


# -- write-time MMR --------------------------------------------------------


def test_mmr_changes_stored_vectors_and_the_returned_order(traces: list[Trace]) -> None:
    with PlainVectorRetriever() as plain, WriteTimeMmrRetriever() as mmr:
        plain.index(traces)
        mmr.index(traces)
        plain_vectors = plain.store.get_vectors([t.id for t in traces])
        mmr_vectors = mmr.store.get_vectors([t.id for t in traces])
        assert not np.allclose(plain_vectors, mmr_vectors)
        assert mmr.stats["adjusted_writes"] >= len(traces) - 1
        assert mmr.stats["degenerate_writes"] == 0
        assert mmr.stats["mean_self_similarity"] < 1.0
        n = len(traces)
        probes = [QUERY, "token refresh failure", "plan drift caused by a provider upgrade"]
        assert any(mmr.retrieve(q, n) != plain.retrieve(q, n) for q in probes)
        norms = np.linalg.norm(mmr_vectors.astype(np.float64), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)  # unit norm in, unit norm out
        assert np.isfinite(mmr_vectors).all()


def test_mmr_lambda_one_is_a_no_op(traces: list[Trace]) -> None:
    with PlainVectorRetriever() as plain, WriteTimeMmrRetriever(lambda_=1.0) as mmr:
        plain.index(traces)
        mmr.index(traces)
        ids = [trace.id for trace in traces]
        assert np.allclose(
            plain.store.get_vectors(ids), mmr.store.get_vectors(ids), atol=1e-6
        )
        assert mmr.retrieve(QUERY, 3) == plain.retrieve(QUERY, 3)


def test_mmr_rejects_out_of_range_parameters() -> None:
    with pytest.raises(ValueError):
        WriteTimeMmrRetriever(lambda_=1.5)
    with pytest.raises(ValueError):
        WriteTimeMmrRetriever(mode="nope")


def test_mmr_objective_mode_also_runs(traces: list[Trace]) -> None:
    for mode in (PROJECTION, OBJECTIVE):
        with WriteTimeMmrRetriever(mode=mode) as arm:
            arm.index(traces)
            assert len(arm.retrieve(QUERY, 2)) == 2
            assert arm.params["mode"] == mode


# -- determinism -----------------------------------------------------------


def test_every_arm_is_deterministic_across_instances(tmp_path: Path, traces: list[Trace]) -> None:
    queries = [QUERY, "plan drift caused by a provider upgrade", "truncated secret"]
    first = {}
    for arm in all_arms(tmp_path):
        with arm:
            arm.index(traces)
            first[arm.name] = [arm.retrieve(query, 3) for query in queries]
    for arm in all_arms(tmp_path):  # fresh stores, fresh clients, cached LLM answers
        with arm:
            arm.index(traces)
            assert [arm.retrieve(query, 3) for query in queries] == first[arm.name], arm.name


def test_shared_embedding_cache_encodes_each_text_once() -> None:
    class CountingEmbedder:
        dim = 4

        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def encode(self, texts: list[str]) -> np.ndarray:
            self.batches.append(list(texts))
            return np.eye(len(texts), 4, dtype=np.float32)

    embedder = CountingEmbedder()
    cache = VectorCache(embedder)
    cache.ensure(["b", "a"])
    cache.matrix(["a", "b", "c"])
    assert embedder.batches == [["a", "b"], ["c"]]  # sorted, and only the missing text
