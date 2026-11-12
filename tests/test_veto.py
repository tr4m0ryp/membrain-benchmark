"""Task 004: the two veto arms, both axes, and the K-I4 verdict.

Offline throughout: two stub clients stand in for the model, one per schema. The
load-bearing tests are the ones that stop arm B from silently becoming arm A: it
must see ONE candidate per call, emit a boolean, not re-score survivors, and
refuse a veto that carries no recalled evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brain.baselines.llm import LlmResponse
from brain.baselines.plain_vector import PlainVectorRetriever
from brain.baselines.veto import LlmPickRetriever, VerifyVetoRetriever, VetoDecision
from brain.corpus.types import Corpus, Episode, Query
from brain.eval.stats import Interval
from brain.experiments.gate_veto import (
    ARM_A, ARM_B, CONTROL, FROZEN_SHA, RECALL_AT_5_CEILING, evaluate_k_i4, headroom, run_veto_gate)
from brain.experiments.gate_veto.report import build_payload, directory_sha, render_markdown
from brain.store import Trace

_TEXTS = {
    "c0-e0": "auth/session.py token refresh failed. Trigger: only for tenants in a negative UTC "
             "offset. Cause: a stale cache entry. Fix: invalidated in the write transaction.",
    "c0-e1": "auth/session.py token refresh failed. Trigger: only under sustained load. "
             "Cause: an exhausted connection pool. Fix: raised the pool ceiling.",
    "c1-e0": "infra/vpc.tf plan drift on the subnet. Trigger: a manual console edit. "
             "Cause: unmanaged state. Fix: imported the resource.",
    "c1-e1": "infra/vpc.tf plan drift on the route table. Trigger: a provider upgrade. "
             "Cause: a renamed attribute. Fix: pinned the provider version."}


def _ids(system: str) -> list[str]:
    """The note ids a prompt actually carried, read back out of it."""
    return [line[2:-2] for line in system.splitlines() if line.startswith("[[")]


def traces() -> list[Trace]:
    return [Trace(text=t, id=i, cluster_id=i.split("-")[0]) for i, t in sorted(_TEXTS.items())]


def corpus() -> Corpus:
    # Distinct query text per query: the arms are only ever handed the text, so
    # two queries sharing one string would be one query to them.
    return Corpus(
        episodes=[Episode(trace_id=i, cluster_id=i[:2], cluster_size=2, date="2026-01-01", text=t)
                  for i, t in sorted(_TEXTS.items())],
        queries=[Query(query_id=f"{i}-q0", cluster_id=i[:2], cluster_size=2, correct_trace_id=i,
                       text=f"Which {i[:2]} event was it, the one recorded as {i}?",
                       distractor_ids=[o for o in sorted(_TEXTS) if o != i and o[:2] == i[:2]],
                       cue_axis="trigger", cue_value_id="g0", answer_axes=["cause"], answer="x")
                 for i in sorted(_TEXTS)],
        manifest={"content_sha256": "test"})


class PickStub:
    """Arm A's client: returns the prompt ids, best-first, optionally reordered."""
    name = "pick-stub"

    def __init__(self, order: list[str] | None = None) -> None:
        self.calls, self.order = [], order

    def complete(self, *, system: str, user: str, model: str, temperature: float,
                 max_tokens: int, schema: dict[str, Any] | None) -> LlmResponse:
        self.calls.append({"system": system, "user": user})
        ids = _ids(system)
        ranked = [i for i in (self.order or []) if i in ids] or ids
        ranked += [i for i in ids if i not in ranked]
        return LlmResponse(text=json.dumps({"ranked_ids": ranked}), output_tokens=9,
                           input_tokens=len(system) // 4, cost_usd=0.002, source="api")


class VetoStub:
    """Arm B's client: one verdict per (query, candidate), keyed by trace id."""
    name = "veto-stub"

    def __init__(self, verdicts: dict[str, tuple[str, str]] | None = None) -> None:
        self.calls, self.verdicts = [], dict(verdicts or {})

    def complete(self, *, system: str, user: str, model: str, temperature: float,
                 max_tokens: int, schema: dict[str, Any] | None) -> LlmResponse:
        self.calls.append({"system": system, "user": user})
        ids = _ids(system)
        verdict, recalled = self.verdicts.get(ids[0] if ids else "", ("ABSENT", ""))
        payload = {"attribute": "trigger", "cue": "the trigger named", "recalled": recalled,
                   "verdict": verdict}
        return LlmResponse(text=json.dumps(payload), input_tokens=len(system) // 4,
                           output_tokens=7, cost_usd=0.001, source="api")


def pick(tmp_path: Path, client: Any) -> LlmPickRetriever:
    return LlmPickRetriever(client=client, cache_dir=tmp_path / "pick")


def veto(tmp_path: Path, client: Any) -> VerifyVetoRetriever:
    return VerifyVetoRetriever(client=client, cache_dir=tmp_path / "veto")


def _base_order(query: str) -> list[str]:
    with PlainVectorRetriever() as base:
        base.index(traces())
        return base.retrieve(query, 4)


# -- arm B is a real veto, not a reranker ---------------------------------
def test_veto_judges_one_candidate_per_call_in_isolation(tmp_path: Path) -> None:
    client = VetoStub()
    with veto(tmp_path, client) as arm:
        arm.index(traces())
        arm.retrieve("which auth event", 4)
    assert len(client.calls) == 4  # one call per candidate, never one for the set
    for call in client.calls:
        assert len(_ids(call["system"])) == 1, "arm B saw >1 candidate and could rank them"


def test_only_a_grounded_mismatch_vetoes(tmp_path: Path) -> None:
    with veto(tmp_path, VetoStub({"c0-e1": ("MISMATCH", "only under sustained load")})) as arm:
        arm.index(traces())
        arm.retrieve("q", 4)
        decisions = arm.decisions["q"]
    assert decisions["c0-e1"].veto is True and arm.stats["vetoes"] == 1
    assert all(d.veto is False for tid, d in decisions.items() if tid != "c0-e1")


def test_match_and_absent_never_veto(tmp_path: Path) -> None:
    verdicts = {"c0-e0": ("MATCH", "a negative UTC offset"), "c0-e1": ("ABSENT", "")}
    with veto(tmp_path, VetoStub(verdicts)) as arm:
        arm.index(traces())
        arm.retrieve("q", 4)
        assert arm.stats["vetoes"] == 0  # failure to recall is not evidence


def test_an_ungrounded_mismatch_is_downgraded_not_trusted(tmp_path: Path) -> None:
    # The failure mode this guard exists for: "it just looks unrelated" dressed
    # up as a MISMATCH, with nothing actually recalled out of the note.
    with veto(tmp_path, VetoStub({"c1-e0": ("MISMATCH", "quantum wombat telemetry")})) as arm:
        arm.index(traces())
        arm.retrieve("q", 4)
        assert arm.stats["vetoes"] == 0
        assert arm.stats["ungrounded_vetoes_downgraded"] == 1
        assert arm.decisions["q"]["c1-e0"].grounded is False


def test_the_veto_output_is_a_boolean_not_a_score() -> None:
    decision = VetoDecision("c0-e0", "MISMATCH", "trigger", "cue", "sustained load", True)
    assert isinstance(decision.veto, bool)  # no score, no confidence, no margin
    assert set(decision.to_dict()) >= {"veto", "verdict", "recalled", "grounded"}


def test_survivors_keep_base_order_and_vetoed_go_below(tmp_path: Path) -> None:
    order = _base_order("auth token refresh under load")
    with veto(tmp_path, VetoStub({order[0]: ("MISMATCH", _TEXTS[order[0]].split(".")[1])})) as arm:
        arm.index(traces())
        ranked = arm.retrieve("auth token refresh under load", 4)
    assert ranked == order[1:] + [order[0]]  # a filter, never a re-score


def test_vetoing_everything_leaves_the_control_order_untouched(tmp_path: Path) -> None:
    verdicts = {tid: ("MISMATCH", _TEXTS[tid].split(".")[1]) for tid in _TEXTS}
    order = _base_order("auth")
    with veto(tmp_path, VetoStub(verdicts)) as arm:
        arm.index(traces())
        assert arm.retrieve("auth", 4) == order
        assert arm.stats["all_vetoed_queries"] == 1


# -- arm A is the strong baseline -----------------------------------------
def test_llm_pick_sees_every_candidate_in_one_call(tmp_path: Path) -> None:
    client = PickStub()
    with pick(tmp_path, client) as arm:
        arm.index(traces())
        ranked = arm.retrieve("auth", 3)
    assert len(client.calls) == 1  # comparative evidence, unlike arm B
    ids = _ids(client.calls[0]["system"])
    assert len(ids) == 3 and ranked[0] in ids


def test_llm_pick_ranking_is_honoured_and_short_answers_pad_from_base(tmp_path: Path) -> None:
    with pick(tmp_path, PickStub(order=["c1-e1"])) as arm:
        arm.index(traces())
        ranked = arm.retrieve("auth", 4)
    assert ranked[0] == "c1-e1" and sorted(ranked) == sorted(_TEXTS)


def test_both_arms_replay_from_the_cache_for_free(tmp_path: Path) -> None:
    with veto(tmp_path, VetoStub()) as arm:
        arm.index(traces())
        first, calls = arm.retrieve("q", 4), arm.stats["api_calls"]
    with veto(tmp_path, VetoStub()) as replay:  # same cache dir
        replay.index(traces())
        assert calls == 4 and replay.retrieve("q", 4) == first
        assert replay.stats["api_calls"] == 0 and replay.stats["cache_hits"] == 4


# -- the experiment: both axes, paired, and the verdict ----------------------
def _run(tmp_path: Path, pick_order: list[str] | None, verdicts: dict[str, tuple[str, str]],
         *, tag: str = "run") -> Any:
    # A fresh cache dir per run: the response cache is keyed by the request, so
    # two runs sharing a directory would replay the first run's verdicts.
    root = tmp_path / tag
    return run_veto_gate(
        corpus(),
        factories={
            CONTROL: PlainVectorRetriever,
            ARM_A: lambda: pick(root, PickStub(order=pick_order)),
            ARM_B: lambda: veto(root, VetoStub(verdicts)),
        },
        seeds=(0, 1),
        k=4,
        n_resamples=200,
        prefetch=False,
    )


def test_arms_get_identical_candidates_and_both_axes_are_reported(tmp_path: Path) -> None:
    run = _run(tmp_path, None, {})
    assert run.pairing["identical"] is True
    assert run.pairing["n_queries_compared"] == len(corpus().queries)
    axes = {"disambiguation", "rejection_rate", "demotion_rate", "harm_rate"}
    for name in (CONTROL, ARM_A, ARM_B):
        assert axes <= set(run.intervals[name])
        assert isinstance(run.intervals[name]["rejection_rate"], Interval)


def test_arm_b_rejection_rate_reads_the_boolean_off_the_mechanism(tmp_path: Path) -> None:
    # Veto every correct trace: the rejection axis goes to 1.0 even though
    # nothing about the ranking says so.
    verdicts = {tid: ("MISMATCH", _TEXTS[tid].split(".")[1]) for tid in _TEXTS}
    run = _run(tmp_path, None, verdicts, tag="all-vetoed")
    assert run.intervals[ARM_B]["rejection_rate"].mean == pytest.approx(1.0)
    run_clean = _run(tmp_path, None, {}, tag="none-vetoed")
    assert run_clean.intervals[ARM_B]["rejection_rate"].mean == pytest.approx(0.0)


def _force(run: Any, axis: str, b: float, a: float) -> None:
    run.intervals[ARM_B][axis] = Interval(b, b - 0.1, b + 0.1, 2, "cluster-bootstrap")
    run.intervals[ARM_A][axis] = Interval(a, a - 0.1, a + 0.1, 2, "cluster-bootstrap")


def test_k_i4_fires_when_arm_b_does_not_beat_arm_a(tmp_path: Path) -> None:
    run = _run(tmp_path, None, {})
    _force(run, "disambiguation", 0.40, 0.60)
    verdict = evaluate_k_i4(run, n_resamples=200)
    assert verdict["status"] == "killed" and verdict["killed"] is True
    assert "does not beat" in verdict["reasons"][0]


def test_k_i4_fires_when_the_gain_is_bought_with_rejections(tmp_path: Path) -> None:
    run = _run(tmp_path, None, {})
    _force(run, "disambiguation", 0.80, 0.60)
    _force(run, "rejection_rate", 0.30, 0.10)
    verdict = evaluate_k_i4(run, n_resamples=200)
    assert verdict["status"] == "killed" and "higher rate" in verdict["reasons"][0]


def test_k_i4_survives_only_when_both_axes_are_favourable(tmp_path: Path) -> None:
    run = _run(tmp_path, None, {})
    _force(run, "disambiguation", 0.80, 0.60)
    _force(run, "rejection_rate", 0.05, 0.10)
    verdict = evaluate_k_i4(run, n_resamples=200)
    assert verdict["status"] == "survives" and verdict["survives"] is True


def test_k_i4_is_undecided_without_both_arms(tmp_path: Path) -> None:
    run = _run(tmp_path, None, {})
    run.intervals.pop(ARM_A)
    assert evaluate_k_i4(run, n_resamples=200)["status"] == "undecided"


def test_headroom_and_payload_carry_the_ceiling_and_the_verdict(tmp_path: Path) -> None:
    run = _run(tmp_path, None, {})
    report = headroom(run)
    control = report["control_disambiguation"]
    assert report["ceiling_recall_at_5"] == RECALL_AT_5_CEILING
    assert report["available_headroom_points"] == pytest.approx(
        (RECALL_AT_5_CEILING - control) * 100.0)
    assert set(report["arms"]) == set(run.intervals)
    payload = build_payload(run, command="pytest", sha_before="a", sha_after="a", n_resamples=200)
    assert payload["corpus"]["directory_unchanged"] is True
    assert set(payload["axis_definitions"]) == {
        "disambiguation", "rejection_rate", "demotion_rate", "harm_rate"}
    text = render_markdown(payload)
    assert "correct-item rejection" in text and "VERDICT: K-I4" in text


# -- the frozen corpus is immutable ---------------------------------------
def test_the_frozen_corpus_still_carries_the_published_hash() -> None:
    from brain.config import CORPUS_DIR
    from brain.corpus import load_corpus
    directory = CORPUS_DIR / "synthetic"
    if not (directory / "manifest.json").is_file():
        pytest.skip("frozen corpus not present")
    before = directory_sha(directory)
    assert load_corpus(directory).content_hash == FROZEN_SHA
    assert directory_sha(directory) == before  # reading it changed nothing
