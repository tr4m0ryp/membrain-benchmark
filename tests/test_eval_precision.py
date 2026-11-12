"""Memory precision (I5a): the metric, the judge's plumbing, and the control set.

Offline throughout. Every judge here is a stub, so the tests pin the definition
and the accounting rather than the model's opinions. The live judge was
exercised separately with `python -m brain.eval.precision`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from brain.baselines.llm import CachedClient, LlmResponse, LlmUnavailable
from brain.eval.precision import (
    AGREEMENT_BAR,
    CONTROL_SET,
    CONTROL_TRANSCRIPTS,
    ENTAILED,
    NOT_ENTAILED,
    PROMPT_VERSION,
    UNPARSED,
    UNSUPPORTED_BUT_PLAUSIBLE,
    EntailmentJudge,
    JudgeStats,
    control_by_transcript,
    extrapolate_cost,
    memory_precision,
    precision_report,
    validate_judge,
)

TRANSCRIPT = "Fixed the flapping flag: a cache entry was never invalidated on write. PR #9166."


class StubJudge:
    """Answers from a fixed verdict list, cycling. Records what it was asked."""

    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[tuple[str, str]] = []

    def judge(self, transcript: str, assertion: str) -> tuple[str, str]:
        self.calls.append((transcript, assertion))
        verdict = self._verdicts[(len(self.calls) - 1) % len(self._verdicts)]
        return verdict, "stub"


class StubClient:
    """Returns a canned tool-call payload and records the exact request."""

    name = "stub"

    def __init__(self, replies: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies = list(replies or [])

    def complete(self, *, system: str, user: str, model: str, temperature: float,
                 max_tokens: int, schema: dict[str, Any] | None) -> LlmResponse:
        self.calls.append({"system": system, "user": user, "model": model,
                           "temperature": temperature, "schema": schema})
        text = self.replies.pop(0) if self.replies else json.dumps(
            {"verdict": ENTAILED, "reason": "stated outright"}
        )
        return LlmResponse(text=text, input_tokens=11, cache_creation_input_tokens=900,
                           cached_input_tokens=7, output_tokens=40, cost_usd=0.003, source="cli",
                           temperature_enforced=False)


# -- the metric --------------------------------------------------------------


def test_all_entailed_scores_one() -> None:
    score, items = memory_precision(["a", "b", "c"], TRANSCRIPT, judge=StubJudge([ENTAILED]))
    assert score == 1.0
    assert [i.verdict for i in items] == [ENTAILED] * 3
    assert [i.assertion_id for i in items] == ["t0#00", "t0#01", "t0#02"]


def test_all_invented_scores_zero() -> None:
    score, _ = memory_precision(["a", "b"], TRANSCRIPT, judge=StubJudge([NOT_ENTAILED]))
    assert score == 0.0


def test_plausible_inventions_count_as_invented() -> None:
    # The whole point of the third label: a good guess is still a fabrication.
    judge = StubJudge([ENTAILED, UNSUPPORTED_BUT_PLAUSIBLE, ENTAILED, NOT_ENTAILED])
    score, items = memory_precision(["a", "b", "c", "d"], TRANSCRIPT, judge=judge)
    assert score == 0.5
    assert sum(i.invented for i in items) == 2
    assert sum(i.supported for i in items) == 2


def test_single_assertion_is_zero_or_one() -> None:
    assert memory_precision(["a"], TRANSCRIPT, judge=StubJudge([ENTAILED]))[0] == 1.0
    assert memory_precision(["a"], TRANSCRIPT, judge=StubJudge([NOT_ENTAILED]))[0] == 0.0


def test_empty_assertion_set_is_nan_not_one() -> None:
    # 1 - 0/0 is undefined, and scoring it 1.0 would reward storing nothing.
    score, items = memory_precision([], TRANSCRIPT, judge=StubJudge([ENTAILED]))
    assert math.isnan(score)
    assert items == []


def test_unparsed_verdicts_leave_the_denominator() -> None:
    judge = StubJudge([ENTAILED, UNPARSED, NOT_ENTAILED])
    score, items = memory_precision(["a", "b", "c"], TRANSCRIPT, judge=judge)
    assert score == 0.5  # judged on two items, not three
    assert [i.judged for i in items] == [True, False, True]
    assert math.isnan(memory_precision(["a"], TRANSCRIPT, judge=StubJudge([UNPARSED]))[0])


# -- statistics --------------------------------------------------------------


def _four_transcripts() -> dict[str, tuple[str, list[str]]]:
    return {f"k{i}": (f"transcript {i}", [f"a{i}{j}" for j in range(5)]) for i in range(4)}


def test_bootstrap_resamples_transcripts_not_assertions() -> None:
    # One transcript wholly invented, three wholly supported: 15 of 20 assertions
    # supported, but only 4 independent units. The interval must be built on 4.
    verdicts = [ENTAILED] * 15 + [NOT_ENTAILED] * 5
    report = precision_report(_four_transcripts(), judge=StubJudge(verdicts), n_resamples=2000)
    assert report.score == pytest.approx(0.75)
    assert report.n_assertions == 20
    assert report.n_transcripts == 4
    assert sorted(report.groups) == ["k0", "k1", "k2", "k3"]
    for interval in report.seed_intervals:
        assert interval.n == 4  # clusters, not the 20 assertions
        assert interval.width > 0.0


def test_report_uses_at_least_five_seeds_and_a_stable_point_estimate() -> None:
    report = precision_report(_four_transcripts(), judge=StubJudge([ENTAILED] * 3 + [NOT_ENTAILED]),
                              n_resamples=1000)
    assert len(report.seeds) >= 5
    assert len(report.seed_intervals) == len(report.seeds)
    # The point estimate is a function of the verdicts, so it cannot move with
    # the bootstrap seed; only the endpoints may.
    assert {round(i.mean, 12) for i in report.seed_intervals} == {round(report.score, 12)}
    assert report.lower.n == len(report.seeds)
    assert report.invented_rate == pytest.approx(1.0 - report.score)


def test_report_counts_and_ambiguity_rate() -> None:
    verdicts = [ENTAILED, NOT_ENTAILED, UNSUPPORTED_BUT_PLAUSIBLE, UNPARSED]
    report = precision_report(_four_transcripts(), judge=StubJudge(verdicts), n_resamples=500)
    assert report.counts[ENTAILED] == 5
    assert report.counts[UNPARSED] == 5
    assert report.ambiguity_rate == pytest.approx(0.25)
    assert sum(report.counts.values()) == 20
    assert report.n_assertions == 15  # unparsed items are not in the denominator
    assert set(json.loads(json.dumps(report.to_dict()))["counts"]) == set(report.counts)


def test_the_stricter_reading_of_invented_is_reported_too() -> None:
    # "Invented" folds outright fabrication together with plausible guesses. The
    # choice is a choice, so the strict reading must be recoverable without
    # re-judging anything.
    verdicts = [ENTAILED, ENTAILED, NOT_ENTAILED, UNSUPPORTED_BUT_PLAUSIBLE]
    report = precision_report(_four_transcripts(), judge=StubJudge(verdicts), n_resamples=500)
    assert report.invented_rate == pytest.approx(0.5)
    assert report.fabrication_rate == pytest.approx(0.25)
    assert report.to_dict()["fabrication_rate"] == pytest.approx(0.25)


def test_a_judge_that_loses_verdicts_is_an_error_not_a_shorter_denominator() -> None:
    class Lossy:
        def judge_many(self, pairs, workers=6):  # type: ignore[no-untyped-def]
            return [(ENTAILED, "")] * (len(pairs) - 1)

    with pytest.raises(ValueError, match="verdicts for"):
        memory_precision(["a", "b"], TRANSCRIPT, judge=Lossy())


def test_report_on_no_transcripts_is_nan() -> None:
    report = precision_report({}, judge=StubJudge([ENTAILED]))
    assert math.isnan(report.score) and report.n_transcripts == 0


# -- the control set and judge validation ------------------------------------


def test_control_set_is_large_enough_and_half_invented() -> None:
    assert len(CONTROL_SET) >= 30
    invented = [item for item in CONTROL_SET if not item.supported]
    assert len(invented) * 2 == len(CONTROL_SET)
    assert {i.verdict for i in invented} == {NOT_ENTAILED, UNSUPPORTED_BUT_PLAUSIBLE}
    assert all(item.transcript_id in CONTROL_TRANSCRIPTS for item in CONTROL_SET)
    assert len({item.item_id for item in CONTROL_SET}) == len(CONTROL_SET)
    # The validity probe has to be present or the report cannot answer it.
    assert sum(item.kind == "inference" for item in CONTROL_SET) >= 4
    assert len(control_by_transcript()) == len(CONTROL_TRANSCRIPTS)


def test_validation_of_a_perfect_judge() -> None:
    class Oracle:
        def judge(self, transcript: str, assertion: str) -> tuple[str, str]:
            gold = next(i for i in CONTROL_SET if i.assertion == assertion)
            return gold.verdict, "oracle"

    report = validate_judge(judge=Oracle())
    assert report.n == len(CONTROL_SET)
    assert report.strict_agreement == 1.0
    assert report.binary_agreement == 1.0
    assert report.calibrated is True
    assert report.disagreements == ()


def test_binary_agreement_forgives_the_soft_boundary_strict_does_not() -> None:
    # A judge that confuses the two invention labels is still usable for the
    # metric, which is defined on their union; the report must show both.
    class Confused:
        def judge(self, transcript: str, assertion: str) -> tuple[str, str]:
            gold = next(i for i in CONTROL_SET if i.assertion == assertion)
            if gold.verdict == NOT_ENTAILED:
                return UNSUPPORTED_BUT_PLAUSIBLE, "swapped"
            return gold.verdict, "ok"

    report = validate_judge(judge=Confused())
    assert report.binary_agreement == 1.0
    assert report.strict_agreement < 1.0
    assert report.calibrated is True


def test_an_uncalibrated_judge_is_reported_as_such() -> None:
    report = validate_judge(judge=StubJudge([ENTAILED]))  # says yes to everything
    assert report.binary_agreement == pytest.approx(0.5)
    assert report.binary_agreement < AGREEMENT_BAR
    assert report.calibrated is False
    assert len(report.disagreements) == 18
    assert report.to_dict()["bar"] == AGREEMENT_BAR


# -- the judge: prompts, cache, accounting -----------------------------------


def test_judge_prompt_carries_the_version_and_both_texts(tmp_path: Path) -> None:
    client = StubClient()
    judge = EntailmentJudge(client=client, cache_dir=tmp_path)
    verdict, reason = judge.judge(TRANSCRIPT, "A cache entry was stale.")
    assert verdict == ENTAILED and reason
    request = client.calls[0]
    assert PROMPT_VERSION in request["system"]  # so a prompt edit changes the cache key
    assert TRANSCRIPT in request["user"] and "A cache entry was stale." in request["user"]
    assert request["temperature"] == 0.0
    assert request["schema"]["properties"]["verdict"]["enum"] == list(
        (ENTAILED, NOT_ENTAILED, UNSUPPORTED_BUT_PLAUSIBLE)
    )


def test_second_identical_run_makes_zero_api_calls(tmp_path: Path) -> None:
    first = EntailmentJudge(client=StubClient(), cache_dir=tmp_path)
    expected = first.judge(TRANSCRIPT, "A cache entry was stale.")
    assert first.stats.api_calls == 1

    replay = EntailmentJudge(client=StubClient(replies=["different answer"]), cache_dir=tmp_path)
    assert replay.judge(TRANSCRIPT, "A cache entry was stale.") == expected
    assert replay.stats.api_calls == 0
    assert replay.stats.cache_hits == 1


def test_judge_reports_blocked_instead_of_guessing(tmp_path: Path) -> None:
    judge = EntailmentJudge(client=StubClient(), cache_dir=tmp_path / "empty")
    judge._client = CachedClient(None, tmp_path / "empty")  # no backend, no cache
    assert judge.blocked_reason is not None
    with pytest.raises(LlmUnavailable):
        judge.judge(TRANSCRIPT, "anything")


def test_a_loose_answer_is_parsed_and_a_hopeless_one_is_unparsed(tmp_path: Path) -> None:
    loose = EntailmentJudge(client=StubClient(replies=["Verdict: NOT_ENTAILED, nothing says that"]),
                            cache_dir=tmp_path / "a")
    assert loose.judge(TRANSCRIPT, "x")[0] == NOT_ENTAILED

    broken = StubClient(replies=["I cannot answer that.", "still not a label"])
    judge = EntailmentJudge(client=broken, cache_dir=tmp_path / "b")
    assert judge.judge(TRANSCRIPT, "x")[0] == UNPARSED
    assert len(broken.calls) == 2  # one repair retry, then it gives up honestly
    assert judge.stats.parse_failures == 1


def test_prompt_tokens_sum_every_bucket() -> None:
    # A large prompt reports under cache_creation, not input_tokens; reading only
    # the latter is how this project once reported a near-zero cost.
    stats = JudgeStats()
    stats.record(LlmResponse(text="", input_tokens=10, cache_creation_input_tokens=20953,
                             cached_input_tokens=7, output_tokens=100, cost_usd=0.0, source="api"))
    assert stats.prompt_tokens == 20970
    assert stats.reported_cost_usd == 0.0
    assert stats.estimated_cost_usd > 0.0
    assert stats.cost_usd == stats.estimated_cost_usd  # falls back, never reports zero
    stats.record(LlmResponse(text="", input_tokens=1, output_tokens=1, cost_usd=0.05, source="cli"))
    assert stats.cost_usd == pytest.approx(0.05)
    assert stats.api_calls == 2 and stats.to_dict()["prompt_tokens"] == 20971


def test_a_fully_cached_smoke_still_projects_a_cost() -> None:
    # Otherwise the budget gate is silently disabled on exactly the runs where a
    # cold pass could follow a warm smoke.
    stats = JudgeStats()
    prior = extrapolate_cost(stats, 500)
    assert prior["source"] == "prior"
    assert prior["projected_usd"] > 0.0
    stats.record(LlmResponse(text="", input_tokens=900, output_tokens=100, cost_usd=0.003,
                             source="cli"))
    projection = extrapolate_cost(stats, 100)
    assert projection["source"] == "measured"
    assert projection["per_call_usd"] == pytest.approx(0.003)
    assert projection["projected_usd"] == pytest.approx(0.3)
    assert projection["projected_prompt_tokens"] == 90000


def test_extraction_returns_standalone_assertions(tmp_path: Path) -> None:
    reply = json.dumps({"assertions": ["a stale cache entry caused it", "the fix is PR #9166"]})
    client = StubClient(replies=[reply])
    judge = EntailmentJudge(client=client, cache_dir=tmp_path)
    assert judge.extract(TRANSCRIPT, max_assertions=8) == [
        "a stale cache entry caused it",
        "the fix is PR #9166",
    ]
    assert client.calls[0]["schema"]["properties"]["assertions"]["maxItems"] == 8
