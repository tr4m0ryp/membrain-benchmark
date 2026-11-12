"""Metrics, intervals and the harness: does the instrument measure what it claims.

The toy case is hand-built so every expected value is a known rational, including
the three readings that move disambiguation accuracy: an unretrieved sibling
loses, an unretrieved correct trace is a miss, and a non-sibling ahead of the
correct trace is irrelevant. The last test reproduces the arm accuracies measured
in Tasks 004 to 006 on the frozen corpus, which is the harness's correctness
anchor: if the instrument drifts, that number moves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from brain.corpus.types import Corpus, Episode, Query
from brain.eval import (
    ArmBlocked,
    Interval,
    accuracy_by_cluster_size,
    cohens_d,
    cohens_h,
    disambiguation_accuracy,
    indicators_by_cluster,
    mean_ci,
    metric_summary,
    mrr,
    non_overlapping,
    precision_at_k,
    recall_at_k,
    run_arm,
    score_query,
)

# -- the toy case ----------------------------------------------------------

#: query id -> (correct, siblings, ranking the arm returns)
TOY: dict[str, tuple[str, list[str], list[str]]] = {
    "q1": ("a0", ["a1", "a2"], ["a0", "a1", "b0"]),  # wins outright
    "q2": ("a1", ["a0", "a2"], ["a0", "a1", "a2"]),  # a sibling is ahead: lost
    "q3": ("a2", ["a0", "a1"], ["b0", "b1"]),        # correct trace not retrieved: lost
    "q4": ("b0", ["b1"], ["b0", "a0"]),              # wins outright
    "q5": ("b1", ["b0"], ["a0", "b1"]),              # non-sibling ahead, sibling absent: wins
    "q6": ("b0", [], ["b0"]),                        # no sibling: ineligible, not free credit
}
_CLUSTER_OF = {"a": "cA", "b": "cB"}
_SIZE_OF = {"cA": 3, "cB": 2}


def toy_queries() -> list[Query]:
    out = []
    for query_id, (correct, siblings, _) in sorted(TOY.items()):
        cluster = _CLUSTER_OF[correct[0]]
        out.append(
            Query(
                query_id=query_id,
                cluster_id=cluster,
                cluster_size=_SIZE_OF[cluster],
                text=f"text of {query_id}",
                correct_trace_id=correct,
                distractor_ids=list(siblings),
                cue_axis="axis",
                cue_value_id="v",
                answer_axes=["axis"],
                answer="a",
            )
        )
    return out


def toy_corpus() -> Corpus:
    episodes = [
        Episode(trace_id=tid, cluster_id=_CLUSTER_OF[tid[0]], cluster_size=_SIZE_OF[_CLUSTER_OF[tid[0]]],
                date="2026-01-01", text=f"episode {tid}")
        for tid in ("a0", "a1", "a2", "b0", "b1")
    ]
    return Corpus(
        episodes=episodes,
        queries=toy_queries(),
        manifest={"content_sha256": "toy", "counts": {"episodes": 5, "queries": 6}, "seed": 0},
    )


def toy_outcomes() -> list[Any]:
    return [score_query(query, TOY[query.query_id][2]) for query in toy_queries()]


class ToyArm:
    """Fixed answers, no embeddings. Optional seed dependence and prefetch counting."""

    def __init__(self, *, seed_dependent: bool = False, name: str = "toy") -> None:
        self.name = name
        self.blocked_reason: str | None = None
        self.prefetch_calls = 0
        self.stats: dict[str, Any] = {"n_indexed": 0}
        self._seed_dependent = seed_dependent

    def index(self, traces: Sequence[Any]) -> None:
        self.stats["n_indexed"] = len(traces)

    def prefetch(self, queries: Sequence[str], k: int, **kwargs: Any) -> int:
        self.prefetch_calls += 1
        return len(queries)

    def retrieve(self, query: str, k: int) -> list[str]:
        query_id = query.rsplit(" ", 1)[-1]
        ranking = list(TOY[query_id][2])
        if self._seed_dependent and np.random.rand() < 0.5:
            ranking = list(reversed(ranking))  # reads the GLOBAL rng on purpose
        return ranking[:k]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "arm_class": "ToyArm", "params": {"toy": True}, "stats": self.stats}

    def close(self) -> None:
        self.stats["closed"] = True


# -- metrics ---------------------------------------------------------------


def test_score_query_reads_ranks_exactly() -> None:
    outcomes = {outcome.query_id: outcome for outcome in toy_outcomes()}
    assert (outcomes["q1"].disambiguated, outcomes["q1"].correct_rank) == (True, 1)
    assert (outcomes["q2"].disambiguated, outcomes["q2"].distractors_ahead) == (False, 1)
    assert (outcomes["q3"].retrieved, outcomes["q3"].reciprocal_rank) == (False, 0.0)
    # A non-sibling ahead of the correct trace is irrelevant, and a sibling that
    # was never retrieved ranks below one that was.
    assert (outcomes["q5"].disambiguated, outcomes["q5"].correct_rank) == (True, 2)
    assert (outcomes["q6"].eligible, outcomes["q6"].disambiguated) == (False, None)


def test_metrics_match_the_hand_computed_values() -> None:
    outcomes = toy_outcomes()
    assert disambiguation_accuracy(outcomes) == pytest.approx(3 / 5)  # eligible only
    assert recall_at_k(outcomes) == pytest.approx(5 / 6)
    assert precision_at_k(outcomes, 3) == pytest.approx(5 / 18)
    assert mrr(outcomes) == pytest.approx((1.0 + 0.5 + 0.0 + 1.0 + 0.5 + 1.0) / 6)
    assert accuracy_by_cluster_size(outcomes) == {3: pytest.approx(1 / 3), 2: pytest.approx(1.0)}
    assert indicators_by_cluster(outcomes) == {"cA": [1.0, 0.0, 0.0], "cB": [1.0, 1.0]}
    summary = metric_summary(outcomes, 3)
    assert (summary["n_queries"], summary["n_eligible"], summary["n_clusters"]) == (6.0, 5.0, 2.0)


def test_empty_input_is_nan_not_zero() -> None:
    assert math.isnan(disambiguation_accuracy([]))
    assert math.isnan(mrr([]))
    with pytest.raises(ValueError):
        precision_at_k(toy_outcomes(), 0)


def test_a_corrupt_label_is_refused() -> None:
    bad = toy_queries()[0]
    broken = Query(**{**bad.to_dict(), "distractor_ids": [bad.correct_trace_id]})
    with pytest.raises(ValueError, match="correct trace as a distractor"):
        score_query(broken, ["a0"])


# -- intervals and effect size --------------------------------------------


def test_mean_ci_matches_the_scipy_reference() -> None:
    from scipy import stats

    values = [0.61, 0.63, 0.58, 0.66, 0.60]
    interval = mean_ci(values)
    sample = np.asarray(values)
    expected = stats.t.interval(
        0.95, len(values) - 1, loc=sample.mean(), scale=stats.sem(sample, ddof=1)
    )
    assert interval.lower == pytest.approx(expected[0])
    assert interval.upper == pytest.approx(expected[1])
    assert (interval.n, interval.method) == (5, "t")
    # t, not z: at n = 5 the z interval would be about 30 percent narrower.
    assert interval.width > 2 * 1.96 * float(stats.sem(sample, ddof=1))
    assert tuple(interval) == (interval.mean, interval.lower, interval.upper)


def test_mean_ci_degenerate_cases() -> None:
    single = mean_ci([0.5])
    assert single.mean == 0.5 and not single.defined  # n = 1 is NaN, not zero-width
    assert not mean_ci([]).defined
    flat = mean_ci([0.62] * 5)  # a deterministic arm: real zero variance, flagged
    assert flat.degenerate and flat.width == 0.0
    with pytest.raises(ValueError):
        mean_ci([1.0, 2.0], alpha=0.0)


def test_non_overlapping_at_the_boundary() -> None:
    assert non_overlapping(Interval(0.7, 0.65, 0.75), Interval(0.5, 0.45, 0.55))
    assert non_overlapping(Interval(0.5, 0.45, 0.55), Interval(0.7, 0.65, 0.75))  # symmetric
    # Touching endpoints are NOT separated, and neither is a shared point.
    assert not non_overlapping(Interval(0.7, 0.6, 0.75), Interval(0.5, 0.45, 0.6))
    assert not non_overlapping(Interval(0.7, 0.6, 0.75), Interval(0.5, 0.45, 0.6000001))
    assert not non_overlapping(Interval(0.5, math.nan, math.nan), Interval(0.9, 0.85, 0.95))
    assert non_overlapping((0.7, 0.65, 0.75), (0.45, 0.55))  # triple and pair forms


def test_effect_sizes() -> None:
    assert cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(-3.0)
    assert math.isnan(cohens_d([0.62] * 5, [0.55] * 5))  # deterministic arms: undefined
    assert cohens_h(0.62, 0.62) == 0.0
    assert cohens_h(1.0, 0.5) == pytest.approx(math.pi - 2 * math.asin(math.sqrt(0.5)))
    with pytest.raises(ValueError):
        cohens_h(1.2, 0.5)


# -- the harness -----------------------------------------------------------


def test_run_arm_writes_every_provenance_field(tmp_path: Path) -> None:
    result = run_arm(
        lambda: ToyArm(), toy_corpus(), seeds=(0, 1, 2, 3, 4), k=3, name="toy",
        n_resamples=200, results_dir=tmp_path,
    )
    assert result.path is not None and result.path.parent == tmp_path
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    for field in ("schema", "arm", "k", "seeds_declared", "seeds_run", "per_seed",
                  "aggregate_across_seeds", "disambiguation_cluster_bootstrap",
                  "accuracy_by_cluster_size", "cluster_size_decay", "arm_describe",
                  "corpus", "config", "wall_clock_seconds", "environment"):
        assert field in payload, field
    assert payload["corpus"]["content_sha256"] == "toy"
    assert payload["config"]["resampling_unit"] == "interference cluster"
    assert len(payload["per_seed"]) == 5 and payload["seeds_run"] == [0, 1, 2, 3, 4]
    assert payload["disambiguation_cluster_bootstrap"]["method"].startswith("cluster-bootstrap")
    assert result.disambiguation == pytest.approx(3 / 5)
    assert "NaN" not in result.path.read_text(encoding="utf-8")  # JSON, not almost-JSON


def test_the_same_seeds_give_identical_numbers(tmp_path: Path) -> None:
    def once() -> dict[str, Any]:
        result = run_arm(
            lambda: ToyArm(seed_dependent=True), toy_corpus(), seeds=(0, 1, 2, 3, 4), k=3,
            name="toy", n_resamples=200, results_dir=tmp_path, write=False,
        )
        return {
            "per_seed": [run.metrics["disambiguation"] for run in result.runs],
            "ci": tuple(result.bootstrap),
            "seed_ci": tuple(result.aggregate["disambiguation"]),
        }

    first, second = once(), once()
    assert first == second
    assert len(set(first["per_seed"])) > 1  # the arm really is seed dependent
    assert not math.isnan(first["seed_ci"][1])  # so the across-seed interval is real
    # Pooling identity: the bootstrap point estimate is exactly the mean of the
    # per-seed accuracies, so the two intervals describe the same number.
    assert first["ci"][0] == pytest.approx(float(np.mean(first["per_seed"])))


def test_prefetch_is_called_once_before_the_first_query_loop(tmp_path: Path) -> None:
    built: list[ToyArm] = []

    def factory() -> ToyArm:
        arm = ToyArm()
        built.append(arm)
        return arm

    run_arm(factory, toy_corpus(), seeds=(0, 1, 2), k=3, name="toy", n_resamples=100,
            results_dir=tmp_path, write=False)
    assert [arm.prefetch_calls for arm in built] == [1, 0, 0]
    assert all(arm.stats.get("closed") for arm in built)  # a fresh arm per seed, all closed


def test_seed_independent_arm_runs_once(tmp_path: Path) -> None:
    result = run_arm(
        lambda: ToyArm(), toy_corpus(), seeds=(0, 1, 2, 3, 4), k=3, name="toy",
        n_resamples=100, results_dir=tmp_path, write=False, seed_independent=True,
    )
    assert (result.seeds, result.seeds_run) == ((0, 1, 2, 3, 4), (0,))
    assert not result.aggregate["disambiguation"].defined  # n = 1: NaN, not zero-width
    assert result.bootstrap.defined  # the cluster interval still exists


def test_a_blocked_arm_raises_instead_of_scoring_zero(tmp_path: Path) -> None:
    def factory() -> ToyArm:
        arm = ToyArm()
        arm.blocked_reason = "no LLM backend configured"
        return arm

    with pytest.raises(ArmBlocked, match="no LLM backend"):
        run_arm(factory, toy_corpus(), seeds=(0,), k=3, results_dir=tmp_path, write=False)


def test_a_query_naming_an_absent_trace_is_refused(tmp_path: Path) -> None:
    corpus = toy_corpus()
    broken = Corpus(episodes=corpus.episodes[:2], queries=corpus.queries, manifest=corpus.manifest)
    with pytest.raises(ValueError, match="absent from the corpus"):
        run_arm(lambda: ToyArm(), broken, seeds=(0,), k=3, results_dir=tmp_path, write=False)
