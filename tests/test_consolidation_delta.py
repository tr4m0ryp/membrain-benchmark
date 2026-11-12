"""The consolidation delta metric: crediting, pairing, and the two controls.

Offline throughout: a deterministic stub embedder over a tiny planted corpus and
a stub LLM that echoes the ids it was shown, so every assertion is about this
module rather than about a model. The live measurement ran separately with
`python -m brain.eval.consolidation_delta` on the frozen corpus.

Load-bearing, and therefore tested: merges are credited through
`result.replacements` so a merged-away answer is not a miss by definition (task
003 found that failure with an end-to-end probe); the readings bracket each
other and collapse onto `score_query` without merges; `@full` is invariant to
store size where `@5` is not, the store-size control; and the placebo rewrites to
the merged length without merging content, the length control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from brain.baselines.llm import LlmResponse
from brain.baselines.vectors import VectorCache
from brain.consolidate import run_consolidation
from brain.consolidate.smoke import FROZEN_CORPUS_SHA
from brain.corpus.types import Corpus, Episode, Query
from brain.eval.bootstrap import paired_delta_ci
from brain.eval.metrics import score_query
from brain.eval.consolidation_delta import (
    FILLER, METRICS, MODES, Condition, _pad, answer_tokens, condition_scores,
    conditions_for, measure, placebo_traces, query_indicators)
from brain.store import EpisodicStore, Trace

# Two clusters of three. Inside c0, e0 and e1 sit on top of each other and e2 is
# further out, so tau decides whether the group is a pair or a triple. Each text
# carries its trace id: the stub embedder reads ids out of a merged or padded
# string to place it.
_PLANTED: tuple[tuple[str, str, list[float]], ...] = (
    ("c0-e0", "c0-e0 auth/session.py refresh failed, stale cache", [1.0, 0.02, 0.0, 0.0]),
    ("c0-e1", "c0-e1 auth/session.py refresh failed, exhausted pool", [1.0, 0.06, 0.0, 0.0]),
    ("c0-e2", "c0-e2 auth/session.py refresh failed, truncated secret", [1.0, 0.50, 0.0, 0.0]),
    ("c1-e0", "c1-e0 infra/vpc.tf plan drift, console edit", [0.0, 0.0, 1.0, 0.02]),
    ("c1-e1", "c1-e1 infra/vpc.tf plan drift, manual import", [0.0, 0.0, 1.0, 0.06]),
    ("c1-e2", "c1-e2 infra/vpc.tf plan drift, state rollback", [0.0, 0.0, 1.0, 0.50]),
)
_QUERY_VECTORS = {"c0-e0": [1.0, 0.021, 0.0, 0.0], "c0-e1": [1.0, 0.059, 0.0, 0.0],
                  "c0-e2": [1.0, 0.499, 0.0, 0.0], "c1-e0": [0.0, 0.0, 1.0, 0.021],
                  "c1-e1": [0.0, 0.0, 1.0, 0.059], "c1-e2": [0.0, 0.0, 1.0, 0.499]}
_ANSWERS = {"c0-e0": "cause: stale cache", "c0-e1": "cause: exhausted pool",
            "c0-e2": "cause: truncated secret", "c1-e0": "cause: console edit",
            "c1-e1": "cause: manual import", "c1-e2": "cause: state rollback"}


class StubEmbedder:
    """Planted unit vectors; anything else is placed by the ids it mentions."""

    dim = 4

    def encode(self, texts: Any) -> np.ndarray:
        by_text = {text: vector for _, text, vector in _PLANTED}
        by_id = {tid: vector for tid, _, vector in _PLANTED}
        rows = []
        for text in texts:
            named = [by_id[i] for i in sorted(by_id) if i in text]
            if text in by_text:
                vector = by_text[text]
            elif text.startswith("probe"):
                vector = _QUERY_VECTORS["c0-e0"]  # nearer the query than any episode
            elif text in _QUERY_VECTORS:
                vector = _QUERY_VECTORS[text]
            elif named:  # a merged or padded string sits on the mean of the
                # episodes it mentions, which is how a real summary behaves
                vector = list(np.mean(np.asarray(named, dtype=np.float64), axis=0))
            else:
                seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
                vector = list(np.random.default_rng(seed).normal(size=4))
            rows.append(vector)
        block = np.asarray(rows, dtype=np.float32)
        return block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)


class StubClient:
    """Answers by listing the ids it was shown, padded so a merge is longer."""

    name = "stub"

    def complete(self, *, system: str, user: str, model: str, temperature: float,
                 max_tokens: int, schema: dict[str, Any] | None) -> LlmResponse:
        ids = [line[2:-2] for line in system.splitlines() if line.startswith("[[")]
        body = "MERGED " + " + ".join(ids) + ". " + "consolidated detail. " * 12
        return LlmResponse(text=json.dumps({"consolidated_memory": body}), cost_usd=0.001,
                           input_tokens=len(system) // 4, output_tokens=40, source="api")


@pytest.fixture()
def cache() -> VectorCache:
    return VectorCache(StubEmbedder())


@pytest.fixture()
def corpus() -> Corpus:
    episodes = [Episode(trace_id=tid, cluster_id=tid.split("-")[0], cluster_size=3,
                        date="2026-01-0" + tid[-1], text=text)
                for tid, text, _ in _PLANTED]
    queries = [_query(tid, sorted(o for o, _, _ in _PLANTED if o != tid and o[:2] == tid[:2]),
                      answer=_ANSWERS[tid], cluster=tid[:2], qid=f"{tid}-q0")
               for tid, _, _ in _PLANTED]
    return Corpus(episodes=episodes, queries=queries, manifest={})


@pytest.fixture()
def store(cache: VectorCache, corpus: Corpus) -> EpisodicStore:
    traces = [Trace(text=e.text, id=e.trace_id, ts=float(i), cluster_id=e.cluster_id)
              for i, e in enumerate(corpus.episodes)]
    db = EpisodicStore(":memory:", cache.dim)
    db.add_many(traces, cache.matrix([t.text for t in traces]))
    return db


def _kwargs(tmp_path: Path, cache: VectorCache) -> dict[str, Any]:
    return {"client": StubClient(), "cache_dir": tmp_path / "merges", "vectors": cache}


def _condition(name: str, traces: list[Trace], replacements: dict[str, str]) -> Condition:
    return Condition(name, traces, replacements, {"n_after": len(traces)})


def _query(correct: str, distractors: list[str], *, answer: str = "cause: x",
           cluster: str = "c0", qid: str = "q") -> Query:
    return Query(query_id=qid, cluster_id=cluster, cluster_size=len(distractors) + 1,
                 text=correct, correct_trace_id=correct, distractor_ids=list(distractors),
                 cue_axis="a", cue_value_id="v", answer_axes=["cause"], answer=answer)


# -- crediting: the error task 003 added `replacements` to prevent -------------

def test_a_merged_away_answer_is_credited_not_scored_as_a_miss() -> None:
    query = _query("c0-e0", ["c0-e1"])
    ranking = ["m_0", "c0-e2"]
    blind = query_indicators(query, ranking, {}, 2)
    credited = query_indicators(query, ranking, {"c0-e0": "m_0", "c0-e1": "m_0"}, 2)
    assert blind["credited"] == 0.0  # without the map the answer is a miss by id
    assert credited["credited"] == 1.0 and credited["expected"] == pytest.approx(0.5)


def test_the_three_readings_bracket_each_other_and_agree_without_a_merge() -> None:
    query = _query("c0-e0", ["c0-e1", "c0-e2"])
    clean = query_indicators(query, ["c0-e0", "c0-e1", "c0-e2"], {}, 3)
    assert clean["strict"] == clean["expected"] == clean["credited"] == 1.0
    merged = query_indicators(query, ["m"], {"c0-e0": "m", "c0-e1": "m", "c0-e2": "m"}, 1)
    assert merged["strict"] == 0.0 and merged["credited"] == 1.0
    assert merged["expected"] == pytest.approx(1.0 / 3.0)  # one of three events
    # a surviving original outranking the conflated record keeps full credit,
    # which is the whole difference between the two modes
    kept = query_indicators(query, ["c0-e0", "m"], {"c0-e0": "m", "c0-e1": "m"}, 2)
    assert kept == {"credited": 1.0, "expected": 1.0, "strict": 1.0}


def test_without_merges_it_is_exactly_the_project_disambiguation_metric(
        corpus: Corpus) -> None:
    """The three readings collapse onto `score_query` when nothing was merged.

    That equivalence is what lets a delta be read against the project's existing
    accuracy numbers instead of against a metric invented for this task.
    """
    ids = [tid for tid, _, _ in _PLANTED]
    for query in corpus.queries:
        for ranking in (ids, ids[::-1], ids[2:], ["nothing", "relevant"], []):
            expected = float(bool(score_query(query, ranking[:5]).disambiguated))
            assert query_indicators(query, ranking, {}, 5) == {
                "credited": expected, "expected": expected, "strict": expected}


def test_answer_tokens_drop_the_axis_labels() -> None:
    query = _query("a", ["b"], answer="cause: stale cache; fix: invalidate")
    assert answer_tokens(query) == {"stale", "cache", "invalidate"}


# -- the store-size control ---------------------------------------------------

def test_full_depth_is_store_size_free_where_the_shallow_reading_is_not(
        cache: VectorCache, corpus: Corpus) -> None:
    """Adding episodes changes `@5` by displacement and cannot change `@full`."""
    base = [Trace(text=t, id=i, cluster_id=i.split("-")[0]) for i, t, _ in _PLANTED]
    crowd = base + [Trace(text=f"probe {n}", id=f"a{n}") for n in range(8)]
    queries = sorted(corpus.queries, key=lambda q: q.query_id)
    answers = {q.query_id: answer_tokens(q) for q in queries}
    small = condition_scores(_condition("small", base, {}), queries, answers, cache=cache)
    big = condition_scores(_condition("big", crowd, {}), queries, answers, cache=cache)
    assert big["expected@full"] == small["expected@full"]  # store size cannot act
    assert big["expected@5"] != small["expected@5"]  # pure top-k displacement


# -- the length control -------------------------------------------------------

def test_pad_matches_the_target_length_and_keeps_the_original_text() -> None:
    assert _pad("short text", 5, FILLER) == "short text"  # never truncates content
    assert _pad("abc", 40, None).startswith("abc\nabc")  # self-repetition
    assert _pad("short text", 400, FILLER).startswith("short text")
    assert 380 <= len(_pad("short text", 400, FILLER)) <= 400


def test_placebo_lengthens_the_grouped_episodes_and_leaves_the_store_size_alone(
        tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    before = store.all_traces(valid_only=True)
    result = run_consolidation(store, 0, threshold=0.9, **_kwargs(tmp_path, cache))
    assert result.replacements
    original = {t.id: t.text for t in before}
    for filler in (FILLER, None):
        padded = placebo_traces(before, result, filler=filler)
        assert [t.id for t in padded] == [t.id for t in before]  # size and ids fixed
        for trace in padded:
            if trace.id not in result.replacements:
                assert trace.text == original[trace.id]
                continue
            target = len(result.store.get(result.replacements[trace.id]).text)
            assert trace.text.startswith(original[trace.id])
            assert 0.9 * target <= len(trace.text) <= target
            # nothing merged: no other member's text is inside this one
            assert not any(o in trace.text for o in result.replacements if o != trace.id)


# -- the pass, the pairing, and the report ------------------------------------

def test_exactly_one_pass_per_condition_and_four_arms_per_threshold(
        tmp_path: Path, cache: VectorCache, store: EpisodicStore) -> None:
    conditions = conditions_for(store, thresholds=(0.9, 0.99), **_kwargs(tmp_path, cache))
    assert [c.name for c in conditions][0] == "before"
    assert {c.name for c in conditions} == {"before"} | {
        f"{m}@{tau}" for m in MODES for tau in ("0.9", "0.99")}
    single = run_consolidation(store, 0, threshold=0.9, **_kwargs(tmp_path, cache))
    swept = next(c for c in conditions if c.name == "destructive@0.9")
    # identical to ONE pass: the output is never fed back in for a second round
    assert [(t.id, t.text) for t in swept.traces] == [(t.id, t.text) for t in single.traces]


def test_the_comparison_is_paired_at_cluster_level(
        tmp_path: Path, cache: VectorCache, store: EpisodicStore, corpus: Corpus) -> None:
    """Eligibility is pinned to the before store, so the clusters still align."""
    queries = sorted(corpus.queries, key=lambda q: q.query_id)
    answers = {q.query_id: answer_tokens(q) for q in queries}
    cluster_of = {q.query_id: q.cluster_id for q in queries}
    conditions = conditions_for(store, thresholds=(0.9,), **_kwargs(tmp_path, cache))
    scored = {c.name: condition_scores(c, queries, answers, cache=cache) for c in conditions}
    for name, values in scored.items():
        for metric in METRICS:
            assert sorted(values[metric]) == sorted(scored["before"][metric])
            groups: dict[str, list[float]] = {}
            base: dict[str, list[float]] = {}
            for qid in sorted(values[metric]):
                groups.setdefault(cluster_of[qid], []).append(values[metric][qid])
                base.setdefault(cluster_of[qid], []).append(scored["before"][metric][qid])
            assert paired_delta_ci(groups, base, n_resamples=200).n == 2, name  # or raises


def test_no_groups_gives_exactly_zero_delta(
        tmp_path: Path, cache: VectorCache, store: EpisodicStore, corpus: Corpus) -> None:
    report = measure(corpus=corpus, store=store, thresholds=(0.9995,), seeds=(0, 1),
                     n_resamples=200, cache=cache, **_kwargs(tmp_path, cache))
    for mode in MODES:
        entry = report["conditions"][f"{mode}@0.9995"]
        assert entry["stats"]["n_after"] == len(corpus.episodes)
        assert all(entry["metrics"][m]["delta_vs_before"]["mean"] == 0.0 for m in METRICS)


def test_measure_reports_every_arm_and_records_how_it_was_measured(
        tmp_path: Path, cache: VectorCache, store: EpisodicStore, corpus: Corpus) -> None:
    report = measure(corpus=corpus, store=store, thresholds=(0.9,), seeds=(0, 1, 2, 3, 4),
                     n_resamples=200, cache=cache, **_kwargs(tmp_path, cache))
    config = report["config"]
    assert config["passes_per_condition"] == 1 and config["seeds"] == [0, 1, 2, 3, 4]
    assert config["resampling_unit"] == "interference cluster"
    assert config["seeds_identical"] is True  # deterministic end to end
    assert set(report["conditions"]) == {"before"} | {f"{m}@0.9" for m in MODES}
    for name, entry in report["conditions"].items():
        for metric in METRICS:
            cell = entry["metrics"][metric]
            assert cell["accuracy"]["lower"] <= cell["accuracy"]["mean"] <= cell["accuracy"]["upper"]
            assert cell["delta_vs_before"]["n"] == 2
            assert name != "before" or cell["delta_vs_before"]["mean"] == 0.0
    merged = report["conditions"]["destructive@0.9"]["stats"]
    assert merged["merged_chars_median"] > 0 and merged["n_groups"] >= 1


def test_the_frozen_corpus_is_read_only_and_its_hash_is_asserted(
        tmp_path: Path, cache: VectorCache, store: EpisodicStore, corpus: Corpus) -> None:
    before = [(t.id, t.text, t.invalid_at) for t in store.all_traces(valid_only=False)]
    measure(corpus=corpus, store=store, thresholds=(0.9,), seeds=(0,), n_resamples=100,
            cache=cache, **_kwargs(tmp_path, cache))
    assert [(t.id, t.text, t.invalid_at) for t in store.all_traces(valid_only=False)] == before
    drifted = Corpus(episodes=corpus.episodes, queries=corpus.queries,
                     manifest={"content_sha256": "deadbeef"})
    with pytest.raises(RuntimeError):
        measure(corpus=drifted, store=store, thresholds=(0.9,), seeds=(0,), cache=cache)
    assert FROZEN_CORPUS_SHA == \
        "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb"

