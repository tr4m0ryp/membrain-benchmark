"""The baseline consolidation pass: grouping, both modes, cache, determinism.

Offline throughout: a deterministic stub embedder over a tiny planted corpus and
a stub LLM that echoes the ids it was shown, so every assertion here is about
this package rather than about a model. The live path was validated separately
with `python -m brain.consolidate.smoke`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from brain.baselines.llm import LlmResponse, LlmUnavailable
from brain.baselines.vectors import VectorCache
from brain.consolidate import (
    DEFAULT_THRESHOLDS,
    BaselineConsolidator,
    ConsolidationBudgetExceeded,
    consolidate_baseline,
    group_traces,
    merge_cache_key,
    parse_merged,
    plan_groups,
    price_usd,
    run_consolidation,
    sweep_thresholds,
    warm_cache,
)
from brain.consolidate.smoke import FROZEN_CORPUS_SHA
from brain.store import EpisodicStore, Trace

# Two tight pairs plus one loner. c0-e0/c0-e1 are near-identical, c0-e2 sits at
# a middle distance from both, so single and complete linkage disagree on it:
# that disagreement is what the linkage tests turn on.
_PLANTED: tuple[tuple[str, str, list[float]], ...] = (
    ("c0-e0", "auth/session.py: token refresh failed, stale cache", [1.0, 0.02, 0.0]),
    ("c0-e1", "auth/session.py: token refresh failed, exhausted pool", [1.0, 0.06, 0.0]),
    ("c0-e2", "auth/session.py: token refresh failed, truncated secret", [1.0, 0.5, 0.0]),
    ("c1-e0", "infra/vpc.tf: plan drift on the subnet, console edit", [0.0, 0.0, 1.0]),
)


class StubEmbedder:
    """Planted unit vectors.

    A merged text (`"MERGED a + b"`, from `StubClient`) lands on the mean of its
    members, which is how a real summary behaves: it sits near the episodes it
    was written from. Anything else unseen gets a deterministic random vector.
    """

    dim = 3

    def encode(self, texts: Any) -> np.ndarray:
        by_text = {text: vector for _, text, vector in _PLANTED}
        by_id = {trace_id: vector for trace_id, _, vector in _PLANTED}
        rows = []
        for text in texts:
            if text in by_text:
                vector = by_text[text]
            elif text.startswith("MERGED "):
                members = [by_id[i] for i in text[7:].split(" + ") if i in by_id]
                vector = list(np.mean(np.asarray(members, dtype=np.float64), axis=0))
            else:
                seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
                vector = list(np.random.default_rng(seed).normal(size=3))
            rows.append(vector)
        block = np.asarray(rows, dtype=np.float32)
        return block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)


class StubClient:
    """Answers by listing the ids it was shown. Records every call."""

    name = "stub"

    def __init__(self, replies: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies = list(replies or [])

    def complete(self, *, system: str, user: str, model: str, temperature: float,
                 max_tokens: int, schema: dict[str, Any] | None) -> LlmResponse:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        ids = [line[2:-2] for line in system.splitlines() if line.startswith("[[")]
        text = self.replies.pop(0) if self.replies else json.dumps(
            {"consolidated_memory": "MERGED " + " + ".join(ids)}
        )
        return LlmResponse(text=text, input_tokens=len(system) // 4, output_tokens=40,
                           cost_usd=0.001, source="api")


@pytest.fixture()
def cache() -> VectorCache:
    return VectorCache(StubEmbedder())


@pytest.fixture()
def store(cache: VectorCache) -> EpisodicStore:
    traces = [
        Trace(text=text, id=trace_id, ts=float(i), cluster_id=trace_id.split("-")[0])
        for i, (trace_id, text, _) in enumerate(_PLANTED)
    ]
    db = EpisodicStore(":memory:", cache.dim)
    db.add_many(traces, cache.matrix([t.text for t in traces]))
    return db


def consolidator(tmp_path: Path, cache: VectorCache, **kwargs: Any) -> BaselineConsolidator:
    kwargs.setdefault("client", StubClient())
    kwargs.setdefault("cache_dir", tmp_path / "merges")
    return BaselineConsolidator(vectors=cache, **kwargs)


# -- grouping -------------------------------------------------------------

def test_grouping_is_a_swept_parameter_not_a_constant(store: EpisodicStore) -> None:
    plan = plan_groups(store, thresholds=DEFAULT_THRESHOLDS)
    assert sorted(plan) == sorted(float(t) for t in DEFAULT_THRESHOLDS)
    counts = [sum(g.size for g in plan[tau]) for tau in sorted(plan)]
    assert counts == sorted(counts, reverse=True)  # a higher tau merges no more
    assert plan[0.95] == [] or all(g.score >= 0.95 for g in plan[0.95])


def test_complete_linkage_admits_no_pair_below_tau(store: EpisodicStore) -> None:
    ids = [t.id for t in store.all_traces()]
    vectors = store.get_vectors(ids)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sim = {(a, b): float(unit[i] @ unit[j])
           for i, a in enumerate(ids) for j, b in enumerate(ids)}
    for members, score in group_traces(ids, vectors, threshold=0.9, linkage="complete"):
        pairs = [sim[(a, b)] for a in members for b in members if a < b]
        assert min(pairs) >= 0.9
        assert score == pytest.approx(min(pairs))


def test_single_linkage_chains_where_complete_does_not(store: EpisodicStore) -> None:
    ids = [t.id for t in store.all_traces()]
    vectors = store.get_vectors(ids)
    # tau sits between sim(e1, e2) and sim(e0, e2): single linkage pulls e2 in
    # through e1, complete linkage refuses because e0 and e2 are below tau.
    chained = group_traces(ids, vectors, threshold=0.91, linkage="single")
    strict = group_traces(ids, vectors, threshold=0.91, linkage="complete")
    average = group_traces(ids, vectors, threshold=0.91, linkage="average")
    assert max(len(m) for m, _ in chained) == 3
    assert max(len(m) for m, _ in strict) == 2
    assert max(len(m) for m, _ in average) == 3  # sits between the two


def test_grouping_is_order_independent_and_id_sorted(store: EpisodicStore) -> None:
    ids = [t.id for t in store.all_traces()]
    vectors = store.get_vectors(ids)
    shuffled = list(reversed(ids))
    assert group_traces(ids, vectors, threshold=0.9) == group_traces(
        shuffled, store.get_vectors(shuffled), threshold=0.9
    )


def test_grouping_handles_degenerate_input() -> None:
    assert group_traces([], np.zeros((0, 3)), threshold=0.5) == []
    assert group_traces(["a"], np.ones((1, 3)), threshold=0.5) == []
    zeros = group_traces(["a", "b"], np.zeros((2, 3)), threshold=0.5)
    assert zeros == []  # zero vectors give similarity 0, never NaN
    with pytest.raises(ValueError):
        group_traces(["a", "b"], np.ones((2, 3)), threshold=0.5, linkage="ward")


# -- the two modes --------------------------------------------------------

def test_destructive_replaces_the_group(tmp_path: Path, cache: VectorCache,
                                        store: EpisodicStore) -> None:
    result = consolidator(tmp_path, cache, threshold=0.9, destructive=True).run(store)
    assert result.n_groups >= 1
    grouped = {i for group in result.groups for i in group.member_ids}
    assert grouped.isdisjoint(result.trace_ids)  # originals no longer readable
    assert set(result.merged_ids) <= set(result.trace_ids)
    assert len(result.traces) == len(store.all_traces()) - len(grouped) + result.n_groups
    # invalidation hides, never deletes: the evidence survives in the copy
    assert all(result.store.get(i) is not None for i in grouped)
    assert result.store.knn(cache.vector(_PLANTED[0][1]), 4) != []
    assert all(hit[0] not in grouped for hit in result.store.knn(cache.vector(_PLANTED[0][1]), 4))


def test_non_destructive_keeps_originals_and_adds_the_summary(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    result = consolidator(tmp_path, cache, threshold=0.9, destructive=False).run(store)
    assert result.n_groups >= 1
    assert set(t.id for t in store.all_traces()) <= set(result.trace_ids)
    assert len(result.traces) == len(store.all_traces()) + result.n_groups
    assert result.retired_ids == ()


def test_both_modes_share_the_same_merges(tmp_path: Path, cache: VectorCache,
                                          store: EpisodicStore) -> None:
    shared = tmp_path / "merges"
    client = StubClient()
    hard = BaselineConsolidator(threshold=0.9, destructive=True, client=client,
                                cache_dir=shared, vectors=cache).run(store)
    soft = BaselineConsolidator(threshold=0.9, destructive=False, client=client,
                                cache_dir=shared, vectors=cache).run(store)
    assert soft.spend.api_calls == 0  # the second mode is free
    assert [g.member_ids for g in hard.groups] == [g.member_ids for g in soft.groups]
    assert set(hard.merged_ids) == set(soft.merged_ids)


def test_the_source_store_is_never_written_to(tmp_path: Path, cache: VectorCache,
                                              store: EpisodicStore) -> None:
    before = [(t.id, t.text, t.invalid_at) for t in store.all_traces(valid_only=False)]
    consolidate_baseline(store, 0, threshold=0.9, destructive=True,
                         client=StubClient(), cache_dir=tmp_path / "m", vectors=cache)
    after = [(t.id, t.text, t.invalid_at) for t in store.all_traces(valid_only=False)]
    assert before == after


def test_contract_returns_a_store_with_the_unchanged_read_interface(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    new_store = consolidate_baseline(store, 0, threshold=0.9, client=StubClient(),
                                     cache_dir=tmp_path / "m", vectors=cache)
    assert isinstance(new_store, EpisodicStore)
    assert new_store.dim == store.dim
    assert new_store is not store
    hits = new_store.knn(cache.vector(_PLANTED[3][1]), 2)
    assert hits and hits[0][0] == "c1-e0"  # the untouched loner still retrieves


# -- determinism and cost -------------------------------------------------

def test_same_seed_and_store_give_byte_identical_output(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    def run() -> list[tuple[str, str]]:
        result = run_consolidation(store, 0, threshold=0.9, client=StubClient(),
                                   cache_dir=tmp_path / "m", vectors=cache)
        return [(t.id, t.text) for t in result.traces]

    first, second = run(), run()
    assert first == second
    assert hashlib.sha256(repr(first).encode()).hexdigest() == \
        hashlib.sha256(repr(second).encode()).hexdigest()


def test_a_second_identical_run_makes_zero_api_calls(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    client = StubClient()
    first = consolidator(tmp_path, cache, threshold=0.9, client=client).run(store)
    assert first.spend.api_calls == len(client.calls) > 0
    replay = consolidator(tmp_path, cache, threshold=0.9,
                          client=StubClient(replies=["broken"])).run(store)
    assert replay.spend.api_calls == 0
    assert replay.spend.cache_hits == first.spend.api_calls
    assert [t.text for t in replay.traces] == [t.text for t in first.traces]


def test_the_cache_key_covers_group_prompt_version_and_temperature() -> None:
    base = dict(member_ids=("a", "b"), member_shas=("sha_a", "sha_b"),
                model="m", temperature=0.0, max_tokens=16, seed=0,
                system="SYS", user="USR")
    key = merge_cache_key(**base)
    assert key != merge_cache_key(**{**base, "member_ids": ("a", "c")})
    assert key != merge_cache_key(**{**base, "member_shas": ("sha_a", "sha_c")})
    assert key != merge_cache_key(**{**base, "system": "SYS2"})
    assert key != merge_cache_key(**{**base, "model": "n"})
    assert key != merge_cache_key(**{**base, "temperature": 0.7})
    # At temperature 0 the request cannot depend on the seed, so the seed is
    # deliberately out of the key: five seeds must not cost five times.
    assert key == merge_cache_key(**{**base, "seed": 99})
    hot = merge_cache_key(**{**base, "temperature": 0.7})
    assert hot != merge_cache_key(**{**base, "temperature": 0.7, "seed": 99})


def test_temperature_zero_is_requested(tmp_path: Path, cache: VectorCache,
                                       store: EpisodicStore) -> None:
    client = StubClient()
    arm = consolidator(tmp_path, cache, threshold=0.9, client=client)
    arm.run(store)
    assert all(call["temperature"] == 0.0 for call in client.calls)
    assert arm.params["seed_in_cache_key"] is False


def test_token_accounting_sums_all_three_prompt_buckets() -> None:
    class Cached:
        source, name = "api", "x"
        input_tokens, output_tokens = 10, 40
        cache_creation_input_tokens, cached_input_tokens = 20953, 7
        cost_usd, temperature_enforced = 0.0, True
        prompt_tokens = 20970

    from brain.consolidate import ConsolidationSpend

    spend = ConsolidationSpend()
    spend.add(Cached(), "claude-haiku-4-5")
    assert spend.prompt_tokens == 20970
    # cost falls back to list price when the backend reports none
    assert spend.cost_usd == pytest.approx(
        price_usd("claude-haiku-4-5", input_tokens=10, cache_creation_tokens=20953,
                  cache_read_tokens=7, output_tokens=40)
    )
    assert spend.cost_usd > 0.0


def test_a_replay_still_discloses_that_temperature_was_not_enforced() -> None:
    """The CLI cannot pin temperature. A cached run must not hide that."""
    from brain.consolidate import ConsolidationSpend

    replayed = LlmResponse(text="{}", input_tokens=5, source="cache",
                           temperature_enforced=False)
    spend = ConsolidationSpend()
    spend.add(replayed, "claude-haiku-4-5")
    assert spend.cache_hits == 1 and spend.api_calls == 0
    assert spend.cost_usd == 0.0
    assert spend.temperature_enforced is False


def test_budget_guard_stops_the_pass(tmp_path: Path, cache: VectorCache,
                                     store: EpisodicStore) -> None:
    with pytest.raises(ConsolidationBudgetExceeded):
        consolidator(tmp_path, cache, threshold=0.5, budget_usd=1e-9).run(store)


def test_blocked_instead_of_guessing(tmp_path: Path, cache: VectorCache,
                                     store: EpisodicStore) -> None:
    arm = BaselineConsolidator(client=None, cache_dir=tmp_path / "empty", vectors=cache)
    arm._client = type(arm._client)(None, tmp_path / "empty")
    assert arm.blocked_reason is not None and arm.is_available() is False
    with pytest.raises(LlmUnavailable):
        arm.run(store)


# -- merge handling -------------------------------------------------------

def test_sweep_returns_one_result_per_threshold(tmp_path: Path, cache: VectorCache,
                                                store: EpisodicStore) -> None:
    results = sweep_thresholds(store, 0, thresholds=(0.9, 0.99), client=StubClient(),
                               cache_dir=tmp_path / "m", vectors=cache)
    assert sorted(results) == [0.9, 0.99]
    assert results[0.99].n_groups <= results[0.9].n_groups


def test_prompt_carries_every_note_and_asks_for_detail(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    client = StubClient()
    consolidator(tmp_path, cache, threshold=0.9, client=client).run(store)
    system = client.calls[0]["system"]
    assert system.startswith("PROMPT VERSION: consolidate-baseline-v1")
    assert "distinguishing detail" in system
    assert "SEVERAL DIFFERENT occasions" in system
    for trace_id, text, _ in _PLANTED:
        if f"[[{trace_id}]]" in system:
            assert text in system  # full text, never truncated


def test_an_unparseable_answer_is_repaired_then_counted(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    client = StubClient(replies=["", ""])
    result = consolidator(tmp_path, cache, threshold=0.99, client=client).run(store)
    assert result.stats["parse_failures"] == 1
    assert result.stats["empty_merges"] == 1
    assert len(client.calls) == 2  # one repair retry
    assert result.n_groups == 1 and result.retired_ids == ()


def test_prose_answers_are_accepted_and_recorded() -> None:
    assert parse_merged('{"consolidated_memory": " x "}') == "x"
    assert parse_merged("just prose") == "just prose"
    assert parse_merged("") == ""
    assert parse_merged('{"other": 1}') == ""


def test_the_pass_is_not_idempotent_and_that_is_recorded(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    """A second pass finds new groups. Measured with `plan_groups`, no spend.

    Recorded rather than repaired: a baseline that stopped re-merging its own
    output would no longer be the shipped default being measured. The frozen
    corpus numbers are in the package docstring.
    """
    first = consolidator(tmp_path, cache, threshold=0.9, destructive=False).run(store)
    assert first.stats["n_source_valid"] == len(store.all_traces())
    second = plan_groups(first.store, thresholds=(0.9,))[0.9]
    # each summary sits beside the originals it was written from
    assert len(second) >= first.n_groups
    grouped = {i for group in second for i in group.member_ids}
    assert grouped & set(first.merged_ids)


def test_warming_fills_exactly_what_the_pass_looks_up(tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    """The prefetch and the pass must build the same digest, or warming is a lie."""
    shared = tmp_path / "merges"
    report = warm_cache(store, thresholds=(0.9, 0.99), client=StubClient(),
                        cache_dir=shared, workers=2)
    assert report["pending"] == report["warmed"] > 0
    assert report["failures"] == 0
    # two thresholds proposing the same group share one call
    assert report["unique_requests"] <= report["groups_planned"]
    results = sweep_thresholds(store, 0, thresholds=(0.9, 0.99),
                               client=StubClient(replies=["broken", "broken"]),
                               cache_dir=shared, vectors=cache)
    assert all(result.spend.api_calls == 0 for result in results.values())
    assert all(result.spend.cache_hits > 0 for result in results.values()
               if result.n_groups)


def test_frozen_corpus_sha_is_pinned() -> None:
    assert FROZEN_CORPUS_SHA == (
        "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb"
    )


def test_replacements_map_every_member_to_its_consolidated_episode(
    tmp_path: Path, cache: VectorCache, store: EpisodicStore
) -> None:
    """Without this map a downstream metric scores an id change, not a loss."""
    for destructive in (True, False):
        result = consolidator(tmp_path, cache, threshold=0.9,
                              destructive=destructive).run(store)
        grouped = {i for group in result.groups for i in group.member_ids}
        assert set(result.replacements) == grouped
        assert set(result.replacements.values()) <= set(result.trace_ids)
        assert set(result.replacements.values()) == set(result.merged_ids)
