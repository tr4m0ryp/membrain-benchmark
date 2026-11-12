"""The LoCoMo adapter and the external convention experiment.

Offline throughout. The adapter is tested against a hand-built sample in
LoCoMo's own on-disk shape rather than against the download, so the assertions
hold on a machine that has never fetched the benchmark; two tests that need the
real file skip when it is absent (`data/external/` is gitignored on purpose).

Load-bearing, and therefore tested:

* the exclusions are the pre-registered ones and nothing else silently drops;
* `correct_trace_id` is the CHRONOLOGICALLY earliest evidence turn, not the
  order the annotation happened to list;
* distractors come from the annotation and the two scopes nest, `session`
  inside `conversation`, which is what makes them a sensitivity pair rather
  than two unrelated definitions;
* byte-identical turns are collapsed and the evidence follows the survivor,
  because `content_sha` is UNIQUE and a silently shrunken store moves every
  metric;
* with nothing merged the three conventions coincide, so any spread the
  experiment reports is caused by consolidation and not by the adapter;
* the pre-registered decision rule is a function of the numbers alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from brain.baselines.llm import LlmResponse
from brain.baselines.vectors import VectorCache
from brain.corpus.external import DISTRACTOR_SCOPES, scoped_queries
from brain.corpus.external.locomo import (
    LOCOMO_JSON, LOCOMO_JSON_SHA256, load_locomo, locomo_provenance, render_turn,
)
from brain.eval.consolidation_delta import query_indicators
from brain.experiments.convention_external.design import PRE_REGISTRATION, classify_outcome
from brain.experiments.convention_external.report import render
from brain.experiments.convention_external.runner import METRICS, measure


# -- a two-session sample in LoCoMo's on-disk shape ---------------------------

def _sample() -> dict[str, Any]:
    return {
        "sample_id": "conv-t",
        "conversation": {
            "speaker_a": "Ann", "speaker_b": "Bo",
            "session_1_date_time": "1:00 pm on 1 May, 2023",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": "I adopted a cat named Pixel."},
                {"speaker": "Bo", "dia_id": "D1:2", "text": "I adopted a cat named Byte."},
                {"speaker": "Ann", "dia_id": "D1:3", "text": "I adopted a cat named Pixel."},
                {"speaker": "Bo", "dia_id": "D1:4", "text": "Nice photo.",
                 "blip_caption": "a grey cat"},
            ],
            "session_2_date_time": "2:00 pm on 8 May, 2023",
            "session_2": [
                {"speaker": "Ann", "dia_id": "D2:1", "text": "I started a pottery class."},
                {"speaker": "Bo", "dia_id": "D2:2", "text": "I started a welding class."},
            ],
        },
        "qa": [
            {"question": "Which cat did Ann adopt?", "answer": "Pixel",
             "evidence": ["D1:1"], "category": 4},
            {"question": "Which cat did Bo adopt?", "answer": "Byte",
             "evidence": ["D1:2"], "category": 4},
            {"question": "What class did Ann start?", "answer": "pottery",
             "evidence": ["D2:1"], "category": 4},
            # listed out of chronological order on purpose
            {"question": "What did they both do?", "answer": "adopted cats and took classes",
             "evidence": ["D2:2", "D1:2"], "category": 1},
            {"question": "Unanswerable", "adversarial_answer": "x",
             "evidence": ["D1:1"], "category": 5},
            {"question": "No evidence", "answer": "y", "evidence": [], "category": 3},
            {"question": "Dangling", "answer": "z", "evidence": ["D9:9"], "category": 4},
        ],
    }


@pytest.fixture()
def conversation(tmp_path: Path) -> Any:
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps([_sample()]), encoding="utf-8")
    return load_locomo(path, verify=False)[0]


# -- the mapping --------------------------------------------------------------

def test_turns_are_rendered_the_way_the_benchmark_renders_them() -> None:
    turn = {"speaker": "Bo", "text": "Nice photo.", "blip_caption": "a grey cat"}
    assert render_turn(turn) == 'Bo said, "Nice photo." and shared a grey cat.'


def test_identical_turns_collapse_and_the_evidence_follows_the_survivor(conversation: Any) -> None:
    ids = [e.trace_id for e in conversation.episodes]
    assert "conv-t/D1:3" not in ids, "the duplicate of D1:1 must not become a second row"
    assert len(ids) == len(set(ids)) == 5
    assert conversation.stats["n_collapsed_turns"] == 1
    assert conversation.stats["n_turns_raw"] == 6


def test_only_the_pre_registered_exclusions_drop_questions(conversation: Any) -> None:
    assert conversation.stats["dropped"] == {
        "adversarial": 1, "no_evidence": 1, "unresolved_evidence": 1
    }
    assert conversation.stats["n_qa_raw"] == 7
    assert len(conversation.queries) == 4


def test_the_correct_trace_is_the_chronologically_earliest_evidence(conversation: Any) -> None:
    multi = [q for q in conversation.queries if q.cue_axis == "category:1"]
    assert len(multi) == 1
    # the annotation lists D2:2 first; session 1 comes first in time
    assert multi[0].correct_trace_id == "conv-t/D1:2"


def test_clusters_are_sessions_and_episodes_carry_their_date(conversation: Any) -> None:
    assert {e.cluster_id for e in conversation.episodes} == {
        "conv-t/session_1", "conv-t/session_2"
    }
    assert all(e.date for e in conversation.episodes)


# -- distractors --------------------------------------------------------------

def test_session_distractors_nest_inside_conversation_distractors(conversation: Any) -> None:
    session = {q.query_id: set(q.distractor_ids) for q in scoped_queries(conversation, "session")}
    whole = {q.query_id: set(q.distractor_ids)
             for q in scoped_queries(conversation, "conversation")}
    assert set(session) == set(whole)
    assert any(session[qid] < whole[qid] for qid in session), "the scopes must differ somewhere"
    for qid in session:
        assert session[qid] <= whole[qid]


def test_a_questions_own_evidence_is_never_its_own_distractor(conversation: Any) -> None:
    for scope in DISTRACTOR_SCOPES:
        for query in scoped_queries(conversation, scope):
            assert query.correct_trace_id not in query.distractor_ids
    multi = [q for q in scoped_queries(conversation, "conversation")
             if q.cue_axis == "category:1"][0]
    assert "conv-t/D2:2" not in multi.distractor_ids, "the other evidence turn is not a distractor"


def test_an_unknown_scope_is_refused(conversation: Any) -> None:
    with pytest.raises(ValueError):
        scoped_queries(conversation, "corpus")


# -- the conventions ----------------------------------------------------------

def test_without_merges_the_three_conventions_coincide(conversation: Any) -> None:
    for query in scoped_queries(conversation, "conversation"):
        ranked = [query.correct_trace_id] + list(query.distractor_ids)
        values = query_indicators(query, ranked, {}, len(ranked))
        assert values["credited"] == values["expected"] == values["strict"] == 1.0


def test_a_merge_that_absorbs_a_distractor_splits_the_conventions(conversation: Any) -> None:
    query = scoped_queries(conversation, "session")[0]
    absorbed = query.distractor_ids[0]
    merged = "conv-t/m0"
    replacements = {query.correct_trace_id: merged, absorbed: merged}
    values = query_indicators(query, [merged], replacements, 5)
    assert values["credited"] == 1.0
    assert values["expected"] == pytest.approx(0.5)
    assert values["strict"] == 0.0


# -- the pre-registered decision rule ----------------------------------------

@pytest.mark.parametrize(
    "credited,strict,lower_c,upper_c,lower_s,upper_s,coincide,expected",
    [
        (0.06, -0.30, 0.01, 0.12, -0.39, -0.22, 0.4, "STRONG_FLIP"),
        (0.06, -0.30, -0.01, 0.12, -0.39, -0.22, 0.4, "SIGN_FLIPS"),
        (-0.05, -0.30, -0.09, -0.01, -0.39, -0.22, 0.4, "NO_FLIP"),
        (0.06, -0.30, 0.01, 0.12, -0.39, -0.22, 0.99, "IMMUNE"),
    ],
)
def test_the_decision_rule_is_a_function_of_the_numbers(
    credited: float, strict: float, lower_c: float, upper_c: float,
    lower_s: float, upper_s: float, coincide: float, expected: str
) -> None:
    assert classify_outcome(
        credited, strict,
        credited_ci={"lower": lower_c, "upper": upper_c},
        strict_ci={"lower": lower_s, "upper": upper_s},
        zero_absorbed_fraction=coincide,
    ) == expected


def test_the_pre_registration_names_every_load_bearing_choice() -> None:
    for key in ("dataset", "mapping", "consolidation", "conventions", "metrics",
                "statistics", "seeds", "primary_cell", "decision_rule"):
        assert key in PRE_REGISTRATION
    assert PRE_REGISTRATION["statistics"]["query_level_bootstrap"].startswith("FORBIDDEN")
    assert PRE_REGISTRATION["consolidation"]["linkage"].startswith("complete")


# -- end to end, offline ------------------------------------------------------

_TOPIC_VECTORS = {"cat": np.array([1.0, 0.0, 0.0, 0.0]),
                  "class": np.array([0.0, 1.0, 0.0, 0.0]),
                  "other": np.array([0.0, 0.0, 1.0, 0.0])}
_ID_TOPIC = {"conv-t/D1:1": "cat", "conv-t/D1:2": "cat", "conv-t/D1:4": "other",
             "conv-t/D2:1": "class", "conv-t/D2:2": "class"}
_TEXT_TOPIC = (("adopted a cat", "cat"), ("cat", "cat"), ("class", "class"))


class StubEmbedder:
    """Topic vectors with a tiny per-text jitter, so siblings sit above any tau.

    The two cat turns land within cosine 0.999 of each other and therefore merge,
    which is the only way the end-to-end test exercises the crediting path at
    all. A merged summary sits on the mean of the topics whose ids it names, the
    way a real summary behaves.
    """

    dim = 4

    def _row(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        jitter = np.random.default_rng(seed).normal(size=self.dim) * 0.02
        named = [_TOPIC_VECTORS[t] for i, t in sorted(_ID_TOPIC.items()) if i in text]
        if named:
            return np.mean(np.asarray(named), axis=0) + jitter
        for keyword, topic in _TEXT_TOPIC:
            if keyword in text:
                return _TOPIC_VECTORS[topic] + jitter
        return np.random.default_rng(seed).normal(size=self.dim)

    def encode(self, texts: Any) -> np.ndarray:
        block = np.asarray([self._row(str(t)) for t in texts], dtype=np.float32)
        return block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)


class StubClient:
    name = "stub"

    def complete(self, *, system: str, user: str, model: str, temperature: float,
                 max_tokens: int, schema: dict[str, Any] | None) -> LlmResponse:
        ids = [line[2:-2] for line in system.splitlines() if line.startswith("[[")]
        body = "MERGED " + " + ".join(ids) + ". " + "detail. " * 8
        return LlmResponse(text=json.dumps({"consolidated_memory": body}), cost_usd=0.0,
                           input_tokens=8, output_tokens=8, source="api")


def test_the_pipeline_runs_end_to_end_and_reports_every_metric(
    conversation: Any, tmp_path: Path
) -> None:
    report = measure(
        thresholds=(0.85,), n_resamples=200, conversations=[conversation],
        cache=VectorCache(StubEmbedder()), client=StubClient(),
        cache_dir=tmp_path / "merges",
    )
    assert report["schema"] == "convention-external-v1"
    assert report["pre_registration"]["registered_before_any_result"] is True
    for scope in DISTRACTOR_SCOPES:
        for unit in ("conversation", "session"):
            cells = report["results"][scope][unit]["before"]
            assert set(cells) == set(METRICS)
    assert report["verdict"]["outcome"] in {"STRONG_FLIP", "SIGN_FLIPS", "IMMUNE", "NO_FLIP"}
    assert report["spend"]["api_calls"] >= 1 and report["spend"]["temperature_enforced"] is True
    assert "Pre-registration" in render(report)
    # the run has to have actually merged something, or nothing was exercised
    assert report["conditions"]["destructive@0.85"]["stats"]["n_merged"] >= 1
    histogram = report["diagnostics"]["session"]["absorbed_histogram_at_primary"]
    assert set(histogram) - {"0", "lost"}, f"no query saw a >= 1: {histogram}"


# -- the real download, when it is present -----------------------------------

@pytest.mark.skipif(not LOCOMO_JSON.is_file(), reason="LoCoMo not downloaded (gitignored)")
def test_the_downloaded_benchmark_matches_the_recorded_hash() -> None:
    digest = hashlib.sha256(LOCOMO_JSON.read_bytes()).hexdigest()
    assert digest == LOCOMO_JSON_SHA256


@pytest.mark.skipif(not LOCOMO_JSON.is_file(), reason="LoCoMo not downloaded (gitignored)")
def test_the_downloaded_benchmark_has_the_pre_registered_shape() -> None:
    provenance = locomo_provenance(load_locomo())
    recorded = PRE_REGISTRATION["dataset"]["verified_shape"]
    assert provenance["n_conversations"] == recorded["conversations"]
    assert provenance["n_episodes"] == recorded["episodes_after_collapsing_identical_turns"]
    assert provenance["n_queries"] == recorded["usable_queries"]
    assert provenance["n_clusters"] == recorded["sessions"]
