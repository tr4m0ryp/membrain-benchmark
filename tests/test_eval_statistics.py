"""The intervals: resampling unit, calibration against known truth, the curve.

The load-bearing test is `test_cluster_bootstrap_covers_where_query_level_fails`:
it simulates clustered data with a known truth and shows that the query-level
interval covers it about 58 percent of the time while the cluster interval covers
it about 94 percent. That is the failure mode the whole module exists to prevent,
so it is measured here rather than asserted in a docstring. The BCa machinery
itself is checked against `scipy.stats.bootstrap` on the case where the two agree
by construction (one observation per cluster).

The last section pins the harness to the numbers already in the record (F11 and
Task 006): plain_vector 0.6194, bm25 0.6361, write_time_mmr 0.5139, and the
plain-vector accuracy-vs-cluster-size curve with rho = 0.194.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from brain.config import get_rng
from brain.corpus import load_corpus
from brain.corpus.loader import SYNTHETIC_DIR
from brain.eval import (
    cluster_bootstrap_ci,
    curve_decay,
    paired_delta_ci,
    run_arm,
    stratified_delta_ci,
)

#: Query counts of the frozen corpus: 33 clusters of 3 queries per episode.
CORPUS_SHAPE = [3 * size for size, count in ((2, 12), (3, 8), (4, 6), (6, 4), (8, 3)) for _ in range(count)]
TRUE_P = 0.62


def clustered_dataset(rng: np.random.Generator, icc: float) -> tuple[dict, dict]:
    """Beta-binomial draw: cluster-level rates, then per-query answers.

    `icc` is the intra-cluster correlation, so it controls exactly how wrong the
    independence assumption is. Returns the same data grouped by cluster and
    grouped one query per group (the naive unit).
    """
    concentration = (1.0 / icc) - 1.0
    alpha, beta = TRUE_P * concentration, (1.0 - TRUE_P) * concentration
    clusters: dict[str, list[float]] = {}
    for index, n_queries in enumerate(CORPUS_SHAPE):
        rate = rng.beta(alpha, beta)
        clusters[f"c{index:03d}"] = (rng.random(n_queries) < rate).astype(float).tolist()
    queries = {
        f"q{position:04d}": [value]
        for position, value in enumerate(v for key in sorted(clusters) for v in clusters[key])
    }
    return clusters, queries


# -- the resampling unit ---------------------------------------------------


def test_cluster_bootstrap_covers_where_query_level_fails() -> None:
    """Coverage under known truth. The cluster unit is right, the query unit is not."""
    rng = np.random.default_rng(7)
    trials, covered_cluster, covered_query = 150, 0, 0
    width_cluster = width_query = 0.0
    for _ in range(trials):
        clusters, queries = clustered_dataset(rng, icc=0.30)
        by_cluster = cluster_bootstrap_ci(clusters, n_resamples=499, rng=rng)
        by_query = cluster_bootstrap_ci(queries, n_resamples=499, rng=rng)
        covered_cluster += int(by_cluster.lower <= TRUE_P <= by_cluster.upper)
        covered_query += int(by_query.lower <= TRUE_P <= by_query.upper)
        width_cluster += by_cluster.width
        width_query += by_query.width
    cluster_coverage = covered_cluster / trials
    query_coverage = covered_query / trials
    assert cluster_coverage >= 0.88, cluster_coverage  # nominal 0.95, measured ~0.94
    assert query_coverage <= 0.80, query_coverage  # measured ~0.58: badly anti-conservative
    assert cluster_coverage - query_coverage >= 0.10
    assert width_cluster > 1.5 * width_query  # the honest interval is much wider


def test_bca_endpoints_match_scipy_on_the_iid_case() -> None:
    """One observation per cluster reduces to the ordinary bootstrap, so scipy checks it.

    Agreement is to within one atom of the discrete bootstrap distribution
    (1/n), which is the resolution the quantile grid has on binary data.
    """
    from scipy import stats

    n = 40
    data = (np.random.default_rng(3).random(n) < 0.62).astype(float)
    mine = cluster_bootstrap_ci(
        {f"c{i:03d}": [value] for i, value in enumerate(data)},
        n_resamples=20_000, rng=np.random.default_rng(1),
    )
    reference = stats.bootstrap(
        (data,), np.mean, confidence_level=0.95, method="BCa",
        n_resamples=20_000, random_state=np.random.default_rng(2),
    ).confidence_interval
    assert mine.mean == pytest.approx(float(data.mean()))
    assert mine.lower == pytest.approx(reference.low, abs=1.05 / n)
    assert mine.upper == pytest.approx(reference.high, abs=1.05 / n)
    assert mine.method.startswith("cluster-bootstrap bca")


def test_the_point_estimate_is_the_pooled_mean() -> None:
    groups = {"c0": [1.0, 0.0, 1.0], "c1": [0.0], "c2": [1.0, 1.0]}
    interval = cluster_bootstrap_ci(groups, n_resamples=500, rng=get_rng(0, stream="test"))
    assert interval.mean == pytest.approx(4 / 6)  # ratio estimator, not a mean of means
    assert (interval.n, interval.alpha) == (3, 0.05)
    assert 0.0 <= interval.lower <= interval.mean <= interval.upper <= 1.0


def test_a_saturated_metric_gives_a_flagged_zero_width_interval() -> None:
    interval = cluster_bootstrap_ci({"c0": [1.0, 1.0], "c1": [1.0]}, n_resamples=200)
    assert (interval.mean, interval.width) == (1.0, 0.0)
    assert interval.degenerate and "degenerate" in interval.method


def test_bootstrap_is_reproducible_from_the_seed() -> None:
    groups = {f"c{i}": [float((i + j) % 2) for j in range(4)] for i in range(20)}
    first = cluster_bootstrap_ci(groups, n_resamples=800, rng=get_rng(11, stream="eval.bootstrap"))
    second = cluster_bootstrap_ci(groups, n_resamples=800, rng=get_rng(11, stream="eval.bootstrap"))
    assert tuple(first) == tuple(second)


def test_paired_delta_needs_the_same_clusters() -> None:
    strong = {f"c{i}": [1.0, 1.0, 0.0] for i in range(12)}
    weak = {f"c{i}": [1.0, 0.0, 0.0] for i in range(12)}
    delta = paired_delta_ci(strong, weak, n_resamples=1000, rng=get_rng(0, stream="test"))
    assert delta.mean == pytest.approx(1 / 3)
    assert delta.lower > 0.0  # a real, consistent gain
    with pytest.raises(ValueError, match="same clusters"):
        paired_delta_ci(strong, {f"d{i}": [1.0] for i in range(12)}, n_resamples=10)


def test_stratified_delta_refuses_shared_clusters() -> None:
    small = {f"s{i}": [1.0, 1.0, 0.0] for i in range(8)}
    large = {f"l{i}": [0.0, 0.0, 1.0] for i in range(8)}
    delta = stratified_delta_ci(small, large, n_resamples=1000, rng=get_rng(0, stream="test"))
    assert delta.mean == pytest.approx(1 / 3) and delta.lower > 0.0
    with pytest.raises(ValueError, match="disjoint"):
        stratified_delta_ci(small, small, n_resamples=10)


# -- the curve -------------------------------------------------------------


def _strata(pattern: dict[int, float]) -> dict[int, dict[str, list[float]]]:
    """Synthetic strata: `pattern` maps cluster size to the accuracy at that size."""
    return {
        size: {f"c{size}-{i}": [1.0] * round(accuracy * 6) + [0.0] * (6 - round(accuracy * 6))
               for i in range(6)}
        for size, accuracy in pattern.items()
    }


def test_curve_decay_separates_a_decaying_curve_from_a_flat_one() -> None:
    decaying = curve_decay(_strata({2: 5 / 6, 4: 3 / 6, 8: 1 / 6}), n_resamples=1000,
                           rng=get_rng(0, stream="test"))
    assert decaying["decays"] and not decaying["flat"]
    assert decaying["drop"] == pytest.approx(4 / 6) and decaying["drop_ci"]["lower"] > 0.0
    assert decaying["m_a_fit"]["rho"] > 0.0
    flat = curve_decay(_strata({2: 4 / 6, 4: 4 / 6, 8: 4 / 6}), n_resamples=1000,
                       rng=get_rng(0, stream="test"))
    assert flat["flat"] and not flat["decays"] and flat["drop"] == pytest.approx(0.0)


# -- the instrument against the measured record ----------------------------

#: Task 006 on the frozen 120-episode / 360-query corpus. The harness must
#: reproduce these exactly, otherwise the instrument has drifted.
MEASURED = {"plain_vector": 223, "bm25": 229, "write_time_mmr": 185}  # correct of 360
#: F11: the plain-vector curve, and the M-A collision fit that follows from it.
F11_CURVE = {2: 0.806, 3: 0.736, 4: 0.556, 6: 0.597, 8: 0.403}


@pytest.fixture(scope="module")
def frozen_corpus():
    if not (SYNTHETIC_DIR / "manifest.json").is_file():
        pytest.skip("frozen corpus not present")
    return load_corpus()


@pytest.mark.parametrize("arm", sorted(MEASURED))
def test_harness_reproduces_the_measured_arm_accuracy(frozen_corpus, arm: str, tmp_path) -> None:
    result = run_arm(arm, frozen_corpus, seeds=(0,), k=5, name=arm, n_resamples=200,
                     results_dir=tmp_path, write=False)
    correct = round(result.disambiguation * len(frozen_corpus.queries))
    assert correct == MEASURED[arm], f"{arm}: {result.disambiguation}"
    assert result.runs[0].metrics["n_eligible"] == 360.0  # every query has a sibling


def test_disambiguation_is_stable_in_k(frozen_corpus, tmp_path) -> None:
    """k = 5 is the harness default and the depth the LLM cache was warmed at."""
    scores = [
        run_arm("bm25", frozen_corpus, seeds=(0,), k=k, name="bm25", n_resamples=100,
                results_dir=tmp_path, write=False).disambiguation
        for k in (5, 10, 120)
    ]
    assert scores[0] == scores[1] == scores[2]


def test_plain_vector_curve_and_rho_match_the_recorded_finding(frozen_corpus, tmp_path) -> None:
    result = run_arm("plain_vector", frozen_corpus, seeds=(0,), k=5, name="plain_vector",
                     n_resamples=2000, results_dir=tmp_path, write=False)
    for size, expected in F11_CURVE.items():
        assert result.curve[size].accuracy == pytest.approx(expected, abs=5e-4)
    assert result.decay["m_a_fit"]["rho"] == pytest.approx(0.194429, abs=1e-5)
    assert result.decay["decays"] is True  # the second kill clause cannot misfire
    # The cluster interval is the honest one: on this arm the query-level
    # interval would be about 1.5 times narrower (design effect 2.4).
    naive = cluster_bootstrap_ci(
        {f"q{i:04d}": [value] for i, value in enumerate(
            v for cluster in sorted(result.clusters) for v in result.clusters[cluster])},
        n_resamples=2000, rng=get_rng(0, stream="eval.bootstrap"),
    )
    assert result.bootstrap.width > 1.4 * naive.width
    assert math.isfinite(result.bootstrap.lower) and result.bootstrap.lower < result.disambiguation
