"""Tests for task 006, the threshold-ceiling gate (K1).

Four things are load bearing and each has a test that would fail loudly:

1. The ceiling is the EXACT optimum over all thresholds and both orientations,
   not a grid maximum. Checked against a brute-force sweep and against the
   closed form `0.5 + KS / 2`.
2. The comparison is SUPERSESSION versus INTERFERENCE, and the pair populations
   follow `manifest.conventions` rather than the literal action map. Computing
   the ceiling over the wrong pair set silently answers a different question.
3. The bootstrap resamples CLUSTERS, never pairs. A pair-level interval would be
   far too narrow, which is the error that manufactured a false win once already.
4. Both corpora stay byte-identical. Every result to date is stated against those
   hashes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from brain.corpus.loader import CorpusNotFrozen
from brain.corpus.relations import (
    COINCIDENCE,
    DUPLICATE,
    INTERFERENCE,
    SUPERSESSION,
    SYNTHETIC_CONTENT_SHA256,
    assert_synthetic_corpus_unchanged,
    load_relations,
)
from brain.experiments.gate_threshold import (
    K1_THRESHOLD,
    PairRow,
    best_threshold,
    ceiling_interval,
    direction_report,
    distribution_table,
    expected_loss,
    harm_table,
    k1_verdict,
    loco_balanced_accuracy,
    overlap_bayes_error,
    population,
    stratify,
    subsample,
    summarize,
    support_overlap,
)
from brain.experiments.gate_threshold.report import build_payload, render_markdown, run_experiment

RELATIONS_SHA = "aba780bd15ffeb303b065b1812c0f716e75397cae99112ad73b81ee2197b6e6d"


class StubEmbedder:
    """Deterministic 3-d encoder: no model download, no torch, still normalised."""

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            digest = [float((hash(text[i::3]) % 1000) + 1) for i in range(3)]
            rows.append(digest)
        matrix = np.asarray(rows, dtype=np.float64)
        return matrix / np.linalg.norm(matrix, axis=1)[:, None]


@pytest.fixture(scope="module")
def corpus():
    try:
        return load_relations()
    except CorpusNotFrozen:  # pragma: no cover - only before task 001 freezes
        pytest.skip("no frozen relational corpus at data/corpus/relations")


def _rows(sup, inter, *, regime="long", same_topic=1, clusters=None):
    rows = []
    for i, score in enumerate(sup):
        cluster = clusters[SUPERSESSION][i] if clusters else f"c{i % 3}"
        rows.append(PairRow(f"s{i:03d}", SUPERSESSION, regime, same_topic, cluster, float(score)))
    for i, score in enumerate(inter):
        cluster = clusters[INTERFERENCE][i] if clusters else f"c{i % 3}"
        rows.append(PairRow(f"i{i:03d}", INTERFERENCE, regime, same_topic, cluster, float(score)))
    return rows


def _brute_force(sup, inter, step=0.0005):
    """Independent dense sweep over both orientations. Deliberately naive."""
    best = 0.0
    grid = np.arange(min(min(sup), min(inter)) - 0.01, max(max(sup), max(inter)) + 0.01, step)
    sup, inter = np.asarray(sup), np.asarray(inter)
    for tau in grid:
        high = 0.5 * (np.mean(sup >= tau) + np.mean(inter < tau))
        low = 0.5 * (np.mean(sup <= tau) + np.mean(inter > tau))
        best = max(best, float(high), float(low))
    return best


# --- 1. the ceiling is exact ---------------------------------------------


def test_ceiling_matches_a_dense_brute_force_sweep() -> None:
    rng = np.random.default_rng(11)
    sup = rng.normal(0.80, 0.06, 40)
    inter = rng.normal(0.74, 0.06, 60)
    exact = best_threshold(sup, inter)
    assert exact.balanced_accuracy == pytest.approx(_brute_force(sup, inter), abs=1e-6)
    assert exact.balanced_accuracy >= _brute_force(sup, inter) - 1e-9


def test_ceiling_equals_half_plus_half_the_ks_statistic() -> None:
    from scipy import stats

    rng = np.random.default_rng(3)
    sup, inter = rng.normal(0.7, 0.1, 50), rng.normal(0.6, 0.15, 70)
    exact = best_threshold(sup, inter)
    ks = float(stats.ks_2samp(sup, inter).statistic)
    assert exact.balanced_accuracy == pytest.approx(0.5 + ks / 2.0, abs=1e-9)
    assert exact.ks_statistic == pytest.approx(ks, abs=1e-9)


def test_perfectly_separated_classes_reach_one_and_overlapping_ones_do_not() -> None:
    clean = best_threshold([0.9, 0.95, 0.99], [0.1, 0.2, 0.3])
    assert clean.balanced_accuracy == pytest.approx(1.0)
    assert clean.threshold_bayes_error == pytest.approx(0.0)
    identical = best_threshold([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert identical.balanced_accuracy == pytest.approx(0.5)


def test_both_orientations_are_searched() -> None:
    """The spec WITHDREW the direction, so a one-sided oracle would be too weak."""
    flipped = best_threshold([0.1, 0.15, 0.2], [0.8, 0.85, 0.9])
    assert flipped.balanced_accuracy == pytest.approx(1.0)
    assert flipped.orientation.startswith("s <=")


def test_tau_star_sits_inside_its_plateau_and_is_not_a_grid_point() -> None:
    exact = best_threshold([0.90, 0.92, 0.94], [0.10, 0.12, 0.80])
    lower, upper = exact.plateau
    assert lower <= exact.tau_star <= upper
    assert lower <= exact.tau_midpoint <= upper


def test_empty_class_refuses_rather_than_returning_a_number() -> None:
    with pytest.raises(ValueError):
        best_threshold([], [0.1, 0.2])


# --- 2. the Bayes error of overlapping continuous distributions ----------


def test_overlap_bayes_error_is_zero_for_separated_and_half_for_identical() -> None:
    rng = np.random.default_rng(5)
    far = overlap_bayes_error(rng.normal(5.0, 0.2, 400), rng.normal(-5.0, 0.2, 400))
    assert far["bayes_error_any_f"] == pytest.approx(0.0, abs=1e-3)
    same = rng.normal(0.0, 1.0, 4000)
    identical = overlap_bayes_error(same, rng.normal(0.0, 1.0, 4000))
    assert identical["bayes_error_any_f"] == pytest.approx(0.5, abs=0.03)
    assert identical["integrated_mass_check"] == pytest.approx(1.0, abs=0.01)


def test_overlap_bayes_error_tracks_the_threshold_bayes_error_at_large_n() -> None:
    """Any f(s) beats the best threshold, so at large n its error sits at or below.

    Both estimates are biased in known directions: the oracle threshold error is
    biased DOWN (tau is fitted on the same pairs) and the KDE overlap is biased UP
    (smoothing widens each density). At n = 4000 both biases are small, so the
    ordering the theory predicts is recoverable.
    """
    rng = np.random.default_rng(9)
    sup, inter = rng.normal(0.8, 0.08, 4000), rng.normal(0.7, 0.12, 4000)
    smooth = overlap_bayes_error(sup, inter)["bayes_error_any_f"]
    assert smooth <= best_threshold(sup, inter).threshold_bayes_error + 0.01


def test_degenerate_input_reports_unavailable_rather_than_nan() -> None:
    assert overlap_bayes_error([0.5, 0.5], [0.5, 0.5])["available"] is False


# --- 3. the bootstrap resamples clusters ---------------------------------


def test_bootstrap_resamples_clusters_and_pays_the_design_effect() -> None:
    """A real cluster effect must widen the interval; ignoring it narrows it.

    Same rows, same number of groups, same group sizes. The only difference is
    whether the grouping matches the correlation structure. If the bootstrap were
    resampling pairs rather than blocks, both intervals would be the same width,
    which is exactly the error that manufactured a false win once already.
    """
    rng = np.random.default_rng(2)
    sup, inter, clusters = [], [], {SUPERSESSION: [], INTERFERENCE: []}
    for cluster in range(10):
        shift = float(rng.normal(0.0, 0.20))  # the cluster effect, which is the whole point
        for _ in range(12):
            sup.append(0.82 + shift + float(rng.normal(0, 0.02)))
            clusters[SUPERSESSION].append(f"c{cluster}")
            inter.append(0.74 + shift + float(rng.normal(0, 0.02)))
            clusters[INTERFERENCE].append(f"c{cluster}")
    rows = _rows(sup, inter, clusters=clusters)
    shuffled_ids = [f"c{i % 10}" for i in rng.permutation(len(rows))]
    scrambled = [
        PairRow(r.pair_id, r.cls, r.regime, r.same_topic, cid, r.score)
        for r, cid in zip(rows, shuffled_ids)
    ]
    clustered = ceiling_interval(rows, n_resamples=800, seeds=(0,))
    broken = ceiling_interval(scrambled, n_resamples=800, seeds=(0,))
    assert clustered["n_clusters"] == broken["n_clusters"] == 10
    assert clustered["resampling_unit"].startswith("cluster_id")
    width = lambda block: block["interval"]["upper"] - block["interval"]["lower"]
    assert width(clustered) > width(broken)
    assert "cluster-bootstrap" in clustered["interval"]["method"]


def test_bootstrap_point_estimate_is_seed_independent() -> None:
    rows = _rows(np.linspace(0.7, 0.95, 30), np.linspace(0.5, 0.85, 30))
    block = ceiling_interval(rows, n_resamples=300, seeds=(0, 1, 2, 3, 4))
    assert len(block["per_seed"]) == 5
    assert len({round(row["mean"], 12) for row in block["per_seed"]}) == 1


def test_too_few_clusters_returns_an_undefined_interval_not_a_fake_one() -> None:
    rows = _rows([0.9, 0.8], [0.4, 0.3], clusters={SUPERSESSION: ["c0"] * 2, INTERFERENCE: ["c0"] * 2})
    block = ceiling_interval(rows, n_resamples=50, seeds=(0,))
    assert block["n_clusters"] == 1
    assert not math.isfinite(block["interval"]["lower"])


def test_leave_one_cluster_out_is_never_above_the_oracle_ceiling() -> None:
    rng = np.random.default_rng(4)
    rows = _rows(rng.normal(0.8, 0.05, 60), rng.normal(0.7, 0.05, 60))
    cv = loco_balanced_accuracy(rows)
    oracle = best_threshold([r.score for r in rows if r.cls == SUPERSESSION],
                            [r.score for r in rows if r.cls == INTERFERENCE])
    assert cv["available"] is True
    assert cv["balanced_accuracy"] <= oracle.balanced_accuracy + 1e-9


# --- 4. K1 is evaluated mechanically -------------------------------------


def test_k1_fires_exactly_when_the_ceiling_exceeds_the_bar() -> None:
    fires = best_threshold([0.9, 0.95, 0.99, 0.97], [0.1, 0.2, 0.3, 0.15])
    verdict = k1_verdict("clean", fires, {"interval": {"lower": 0.9, "upper": 1.0}})
    assert verdict["threshold"] == K1_THRESHOLD
    assert verdict["fires"] is True and verdict["status"] == "K1 FIRES"
    muddy = best_threshold([0.80, 0.82, 0.84, 0.86], [0.79, 0.81, 0.83, 0.85])
    assert k1_verdict("muddy", muddy, {"interval": {}})["fires"] is False


def test_k1_does_not_fire_at_exactly_the_bar() -> None:
    """`exceeds 0.90` is strict; a tie is not a fired criterion."""
    at_bar = best_threshold([0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.65])
    assert at_bar.balanced_accuracy == pytest.approx(0.90)
    assert k1_verdict("tie", at_bar, {"interval": {}})["fires"] is False


# --- 5. populations, harm table, expected loss ---------------------------


def test_derivable_population_drops_cross_workstream_interference_only() -> None:
    rows = [
        PairRow("a", INTERFERENCE, "long", 1, "c0", 0.9),
        PairRow("b", INTERFERENCE, "long", 0, "c1", 0.2),
        PairRow("c", COINCIDENCE, "long", 0, "c2", 0.3),
        PairRow("d", SUPERSESSION, "long", 0, "c3", 0.8),
    ]
    derivable = population(rows, "derivable")
    assert {r.pair_id for r in derivable} == {"a", "c", "d"}
    assert {r.pair_id for r in population(rows, "all_interference")} == {"a", "b", "c", "d"}
    assert {r.pair_id for r in stratify(rows, same_topic_only=True)} == {"a"}


def test_harm_table_is_a_tradeoff_not_pure_harm() -> None:
    rows = [
        PairRow("s0", SUPERSESSION, "long", 1, "c0", 0.95),
        PairRow("i0", INTERFERENCE, "long", 1, "c0", 0.93),
        PairRow("i1", INTERFERENCE, "long", 1, "c1", 0.60),
        PairRow("d0", DUPLICATE, "long", 1, "c0", 0.99),
        PairRow("k0", COINCIDENCE, "long", 0, "c2", 0.10),
    ]
    table = {row["tau"]: row for row in harm_table(rows, taus=(0.5, 0.9, 0.99))}
    assert table[0.9]["distinct_events_destroyed"] == 1
    assert table[0.9]["supersessions_correctly_merged"] == 1
    assert table[0.9]["supersessions_missed"] == 0
    assert table[0.9]["merge_precision"] == pytest.approx(2 / 3)
    assert table[0.99]["distinct_events_destroyed"] == 0
    assert table[0.99]["supersessions_missed"] == 1
    assert table[0.9]["clusters_touched"] == 1 and table[0.9]["clusters_total"] == 2


def test_expected_loss_pushes_tau_up_as_the_cost_of_a_false_supersede_rises() -> None:
    rng = np.random.default_rng(8)
    rows = _rows(rng.normal(0.85, 0.05, 80), rng.normal(0.78, 0.05, 80))
    curves = expected_loss(rows, ratios=(1.0, 5.0, 20.0))["curves"]
    taus = [c["tau_argmin"] for c in curves]
    assert taus == sorted(taus)
    assert curves[-1]["tau_argmin"] >= curves[0]["tau_argmin"]


def test_support_overlap_detects_the_theorem_1_prediction() -> None:
    overlapping = support_overlap(np.array([0.5, 0.9]), np.array([0.4, 0.7]))
    assert overlapping["overlaps"] is True and overlapping["interval"] == [0.5, 0.7]
    disjoint = support_overlap(np.array([0.8, 0.9]), np.array([0.1, 0.2]))
    assert disjoint["overlaps"] is False


def test_summarize_handles_an_empty_cell() -> None:
    assert summarize(np.array([]))["n"] == 0
    stats = summarize(np.array([0.1, 0.2, 0.3]))
    assert stats["n"] == 3 and stats["p50"] == pytest.approx(0.2)


# --- 6. the corpora stay byte-identical, end to end ----------------------


def test_both_corpus_hashes_are_unchanged(corpus) -> None:
    assert corpus.content_hash == RELATIONS_SHA
    assert assert_synthetic_corpus_unchanged() == SYNTHETIC_CONTENT_SHA256


def test_end_to_end_on_a_stub_embedder_produces_a_verdict(corpus) -> None:
    small = subsample(corpus, per_class=8)
    result = run_experiment(small, embedder=StubEmbedder(), n_resamples=120, seeds=(0, 1))
    payload = build_payload(
        result,
        corpus=small,
        command="pytest",
        synthetic_sha=SYNTHETIC_CONTENT_SHA256,
        dir_shas={"directory_sha_before": {}, "directory_sha_after": {}, "directories_unchanged": True},
        n_resamples=120,
        seeds=(0, 1),
    )
    assert payload["verdict_line"].startswith("VERDICT: K1")
    assert payload["corpus"]["relations_sha_matches"] is True
    assert payload["corpus"]["synthetic_sha_matches"] is True
    assert payload["config"]["llm_calls"] == 0
    assert any(v["population"] == "derivable" and v["stratum"] == "same_topic"
               for v in payload["k1_verdicts"])
    text = render_markdown(payload)
    assert "the threshold-ceiling gate" in text
    assert "PRIMARY" in text
    assert "—" not in text  # no em-dashes in the emitted report


def test_distribution_and_direction_tables_cover_every_regime(corpus) -> None:
    from brain.experiments.gate_threshold import all_pair_rows, encode_corpus, gram_matrix

    small = subsample(corpus, per_class=8)
    ids, vectors = encode_corpus(small, StubEmbedder())
    rows = population(all_pair_rows(small, ids, gram_matrix(vectors)), "derivable")
    table = distribution_table(rows)
    assert {(r["cls"], r["regime"]) for r in table} >= {
        (SUPERSESSION, "long"), (SUPERSESSION, "short"), (INTERFERENCE, "long")
    }
    report = direction_report(rows, same_topic_only=True)
    assert set(report["per_regime"]) <= {"long", "short"}
    assert isinstance(report["flips_between_regimes"], bool)
