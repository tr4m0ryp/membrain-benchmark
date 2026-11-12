"""Tests for the gate_k_sweep experiment.

Three layers, matching the project's existing test shape (`tests/test_eval.py`):
hand-built toy cases with known-rational expected values for the pure
functions (`pool`, `coverage`, `headroom`), duck-typed fakes for the
`SweepResult` answer methods (so a test does not have to pay for a real
`ArmResult`), and one small real integration pass on the smoke corpus that
exercises `evaluate_point`/`coverage_point` end to end on actual arms.

Nothing here runs the full pre-registered grid: that is 14 cells x 5 seeds x
B=10000 and belongs to `results/`, not to `pytest -q`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.config import CORPUS_DIR
from brain.corpus import load_corpus
from brain.corpus.types import Query
from brain.eval.metrics import score_query
from brain.eval.results import SeedRun
from brain.eval.stats import Interval

from brain.experiments.gate_k_sweep import registration as reg
from brain.experiments.gate_k_sweep.coverage import coverage_point, score_coverage
from brain.experiments.gate_k_sweep.headroom import headroom_row
from brain.experiments.gate_k_sweep.point import evaluate_point
from brain.experiments.gate_k_sweep.pool import indicator, pool_metric_clusters
from brain.experiments.gate_k_sweep.sweep import SweepResult

# -- registration ------------------------------------------------------


def test_pre_registration_grid_matches_task_spec():
    reg_block = reg.pre_registration()
    assert reg_block["k_grid"] == [1, 2, 3, 5, 10, 20, 50]
    assert set(reg_block["free_arms"]) == {"plain_vector", "bm25"}
    assert set(reg_block["out_of_scope_arms"]) == {"llm_pick", "verify_veto", "full_context"}
    assert reg_block["seeds"] == [0, 1, 2, 3, 4]
    assert reg_block["bootstrap"]["n_resamples"] == 10_000
    assert reg_block["bootstrap"]["unit"].startswith("interference cluster")


def test_expected_corpus_sha_matches_the_actual_frozen_manifest():
    """The pre-registered hash must be the one the frozen corpus actually carries,
    not a value that drifted out of sync with `data/corpus/synthetic/manifest.json`."""
    manifest = json.loads((CORPUS_DIR / "synthetic" / "manifest.json").read_text())
    assert manifest["content_sha256"] == reg.EXPECTED_CONTENT_SHA256
    assert manifest["counts"]["clusters"] == 33
    assert manifest["counts"]["episodes"] == 120
    assert manifest["counts"]["queries"] == 360


# -- pool.py -------------------------------------------------------------


def _toy_query(correct: str, siblings: list[str], cluster_id: str = "c1") -> Query:
    return Query(
        query_id=f"q-{correct}",
        cluster_id=cluster_id,
        cluster_size=len(siblings) + 1,
        text="text",
        correct_trace_id=correct,
        distractor_ids=siblings,
        cue_axis="axis",
        cue_value_id="v",
        answer_axes=["axis"],
        answer="ans",
    )


def test_indicator_recall_precision_mrr_on_a_hit_at_rank_2():
    query = _toy_query("a0", ["a1", "a2"])
    outcome = score_query(query, ["a2", "a0", "a1"])  # correct at rank 2 (1 distractor ahead)
    assert indicator(outcome, "recall_at_k", 5) == 1.0
    assert indicator(outcome, "precision_at_k", 5) == pytest.approx(1.0 / 5.0)
    assert indicator(outcome, "mrr", 5) == pytest.approx(0.5)  # 1/rank = 1/2


def test_indicator_on_a_miss_is_zero_everywhere():
    query = _toy_query("a0", ["a1"])
    outcome = score_query(query, ["b0", "b1"])  # correct trace never retrieved
    assert indicator(outcome, "recall_at_k", 3) == 0.0
    assert indicator(outcome, "precision_at_k", 3) == 0.0
    assert indicator(outcome, "mrr", 3) == 0.0


def test_indicator_rejects_unknown_metric_and_nonpositive_k():
    query = _toy_query("a0", ["a1"])
    outcome = score_query(query, ["a0"])
    with pytest.raises(ValueError):
        indicator(outcome, "disambiguation", 5)
    with pytest.raises(ValueError):
        indicator(outcome, "precision_at_k", 0)


def test_pool_metric_clusters_averages_across_seeds_then_groups_by_cluster():
    q_hit = _toy_query("a0", ["a1"], cluster_id="cA")   # retrieved
    q_miss = _toy_query("b0", ["b1"], cluster_id="cB")  # not retrieved

    seed0 = SeedRun(
        seed=0,
        metrics={},
        outcomes=[score_query(q_hit, ["a0"]), score_query(q_miss, ["x"])],
        index_seconds=0.0,
        retrieve_seconds=0.0,
    )
    # A second seed where the miss query now hits: the pooled value must be 0.5.
    seed1 = SeedRun(
        seed=1,
        metrics={},
        outcomes=[score_query(q_hit, ["a0"]), score_query(q_miss, ["b0"])],
        index_seconds=0.0,
        retrieve_seconds=0.0,
    )
    # Averaged per query across seeds FIRST (query q-b0 saw 0.0 then 1.0 -> 0.5),
    # then grouped by cluster -- matching `eval/runner.py`'s own `_pool_clusters`.
    groups = pool_metric_clusters([seed0, seed1], "recall_at_k", k=3)
    assert groups == {"cA": [1.0], "cB": [0.5]}


# -- coverage.py -----------------------------------------------------------


def test_score_coverage_full_and_partial_and_purity():
    query = _toy_query("a0", ["a1", "a2", "a3"])  # 3 siblings

    # All siblings plus the target in the window: full coverage, pure window.
    cluster, covered, purity = score_coverage(query, ["a0", "a1", "a2", "a3"])
    assert cluster == "c1"
    assert covered is True
    assert purity == pytest.approx(1.0)

    # Only one sibling recovered, plus one noise id: partial coverage, half-pure.
    cluster, covered, purity = score_coverage(query, ["a0", "a1", "zzz"])
    assert covered is False
    assert purity == pytest.approx(0.5)

    # Target not even retrieved, window is pure noise: purity 0, no coverage.
    cluster, covered, purity = score_coverage(query, ["zzz", "yyy"])
    assert covered is False
    assert purity == pytest.approx(0.0)

    # k = 1 and the target IS the only slot: no non-target slots, purity is NaN.
    cluster, covered, purity = score_coverage(query, ["a0"])
    assert covered is False  # siblings not present
    assert math.isnan(purity)


def test_the_shared_corpus_invariant_still_holds_for_coverage_inputs():
    """`score_coverage` trusts its `query.distractor_ids` input and does not
    re-validate it; the invariant that the answer cannot be its own sibling is
    enforced once, upstream, by `metrics.score_query`. This pins that down so a
    future refactor cannot silently drop the check from both places at once."""
    query = _toy_query("a0", ["a0"])
    with pytest.raises(ValueError):
        score_query(query, ["a0"])


# -- headroom.py -------------------------------------------------------


def _fake_kpoint(k: int, disambiguation: float, recall: float, arm: str = "plain_vector"):
    return SimpleNamespace(
        k=k,
        arm=arm,
        disambiguation=Interval(disambiguation, disambiguation, disambiguation, 33, "fake", 0.05),
        recall_at_k=Interval(recall, recall, recall, 33, "fake", 0.05),
    )


def test_headroom_row_matches_the_papers_worked_example_at_k5():
    """`main.tex`: A_ctrl = 0.6194, R@5 = 0.958 (rounded), H(llm_pick) = 96.2%.
    Reproduced here with the paper's own numbers as the illustrative arm to
    check the formula, not to claim llm_pick was rerun (it was not, by design).
    """
    ctrl = _fake_kpoint(5, 0.6194, 0.958)
    llm_pick_shaped = _fake_kpoint(5, 0.9444, 0.958, arm="illustrative")
    row = headroom_row(ctrl, llm_pick_shaped, coverage=None)
    assert row["headroom_available_gap"] == pytest.approx(0.958 - 0.6194)
    assert row["H_illustrative_non_canonical"] == pytest.approx(
        (0.9444 - 0.6194) / (0.958 - 0.6194), abs=1e-3
    )


def test_headroom_row_is_nan_not_a_crash_when_the_gap_is_zero():
    ctrl = _fake_kpoint(1, 0.5, 0.5)
    arm = _fake_kpoint(1, 0.6, 0.5, arm="other")
    row = headroom_row(ctrl, arm, coverage=None)
    assert math.isnan(row["H_illustrative_non_canonical"])


# -- sweep.py answer methods, on fakes (no corpus, no bootstrap) -----------


def _sweep_with_fake_points(disambiguation_by_k: dict[int, float], recall_by_k: dict[int, float]) -> SweepResult:
    points = {
        ("plain_vector", k): _fake_kpoint(k, disambiguation_by_k[k], recall_by_k[k])
        for k in disambiguation_by_k
    }
    return SweepResult(points=points, coverage={}, corpus_sha_before="x", corpus_sha_after="x", wall_clock=0.0)


def test_monotonicity_check_passes_on_a_non_increasing_curve():
    result = _sweep_with_fake_points(
        disambiguation_by_k={1: 0.7, 3: 0.7, 5: 0.6, 10: 0.6},
        recall_by_k={1: 0.5, 3: 0.8, 5: 0.9, 10: 1.0},
    )
    report = result.monotonicity_check("plain_vector")
    assert report["non_increasing"] is True
    assert report["violations"] == []
    assert report["total_drop"] == pytest.approx(0.1)


def test_monotonicity_check_flags_a_rise():
    result = _sweep_with_fake_points(
        disambiguation_by_k={1: 0.5, 5: 0.6},  # a rise: must be flagged, not hidden
        recall_by_k={1: 0.5, 5: 0.9},
    )
    report = result.monotonicity_check("plain_vector")
    assert report["non_increasing"] is False
    assert report["violations"] == [(1, 0.5, 5, 0.6)]


def test_recall_saturation_finds_the_plateau_k():
    result = _sweep_with_fake_points(
        disambiguation_by_k={1: 0.6, 5: 0.6, 10: 0.6},
        recall_by_k={1: 0.6, 5: 0.995, 10: 1.0},  # within 1pt of ceiling starting at k=5
    )
    report = result.recall_saturation("plain_vector")
    assert report["ceiling_at_max_k"] == pytest.approx(1.0)
    assert report["plateau_k_within_1pt"] == 5


# -- one real, small end-to-end pass (smoke corpus, not the frozen one) -----


def test_evaluate_point_and_coverage_point_run_on_the_smoke_corpus():
    """A real integration pass, kept fast by using the 6-episode smoke corpus
    (data/corpus/synthetic-smoke) rather than the frozen 120-episode one."""
    corpus = load_corpus(CORPUS_DIR / "synthetic-smoke")
    point = evaluate_point("bm25", corpus, k=2, seeds=(0, 1))
    assert 0.0 <= point.recall_at_k.mean <= 1.0
    assert 0.0 <= point.disambiguation.mean <= 1.0
    assert point.precision_at_k.mean == pytest.approx(point.recall_at_k.mean / 2.0)

    cov = coverage_point("bm25", corpus, k=2, n_resamples=200)
    assert cov.verified_deterministic is True
    assert 0.0 <= cov.coverage.mean <= 1.0
    assert cov.max_cluster_siblings == 2  # smoke corpus: one cluster size, m=3
