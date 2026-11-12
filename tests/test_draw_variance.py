"""Tests for the draw-variance package (`brain.experiments.draw_variance`).

Offline throughout: a deterministic stub embedder (no torch, no model
download) over hand-planted or smoke-sized corpora, matching the pattern
`tests/test_gate_threshold.py` and `tests/test_consolidation_delta.py` use.
The live measurement (fresh LLM calls per draw) ran separately with
`python -m brain.experiments.draw_variance.runner`.

Three things are load-bearing here and each has a test that would fail
loudly:

1. `stats_summary.summarize_draws` computes rank and z-score correctly, and
   flags the frozen value as an extremum when it is one -- this is the whole
   point of the package (F25's draw-luck catch, generalised).
2. `synthetic_harm.bucket_pairs` reproduces F17's three-class bucketing
   (interference = same cluster, coincidence = same diagnosis different
   cluster, unrelated = neither) exactly, on a hand-planted corpus where the
   classification is known in advance.
3. `runner.ScratchStore` replays a seed instead of re-running it, which is
   what makes a crash mid-sweep of a live-LLM statistic (F21, F23) cheap to
   resume rather than a full re-pay.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from brain.corpus.relations.generate import default_relations_config, generate_relations
from brain.corpus.types import Corpus, Episode
from brain.experiments.draw_variance.preregistration import (
    N_DRAWS,
    SEEDS,
    STATISTICS,
    build_preregistration,
)
from brain.experiments.draw_variance.relations_harm import draw_relations_harm
from brain.experiments.draw_variance.runner import ScratchStore, assemble_payload
from brain.experiments.draw_variance.stats_summary import summarize_draws
from brain.experiments.draw_variance.synthetic_harm import (
    PairBuckets,
    bucket_pairs,
    draw_merge_harm,
    merge_harm_rate,
)


class StubEmbedder:
    """Deterministic 3-d encoder: no model download, no torch, still normalised."""

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            digest = [float((hash(text[i::3]) % 1000) + 1) for i in range(3)]
            rows.append(digest)
        matrix = np.asarray(rows, dtype=np.float64)
        return matrix / np.linalg.norm(matrix, axis=1)[:, None]


# --- 1. stats_summary: rank, z-score, extremum ----------------------------


def test_summarize_draws_matches_hand_computed_mean_sd_rank_z() -> None:
    seeds = list(range(6))
    values = [0.60, 0.62, 0.58, 0.90, 0.61, 0.59]  # frozen (seed 0) = 0.60
    summary = summarize_draws("toy", seeds, values, frozen_seed=0)
    array = np.array(values)
    assert summary.mean == pytest.approx(float(array.mean()))
    assert summary.sd == pytest.approx(float(array.std(ddof=1)))
    assert summary.minimum == pytest.approx(0.58)
    assert summary.maximum == pytest.approx(0.90)
    assert summary.ascending_rank == 3  # 0.58, 0.59 and 0.60 itself are <= 0.60
    assert summary.z_score == pytest.approx((0.60 - array.mean()) / array.std(ddof=1))
    assert summary.is_extremum == ""


def test_frozen_value_as_maximum_is_flagged_draw_lucky() -> None:
    summary = summarize_draws("toy", range(4), [0.90, 0.70, 0.65, 0.60], frozen_seed=0)
    assert summary.is_extremum == "maximum"
    assert summary.ascending_rank == 4
    assert "DRAW-LUCKY" in summary.to_dict()["draw_robust"]
    assert "maximum of 4" in summary.to_dict()["draw_robust"]


def test_frozen_value_as_minimum_is_flagged_draw_lucky() -> None:
    summary = summarize_draws("toy", range(4), [0.10, 0.70, 0.65, 0.60], frozen_seed=0)
    assert summary.is_extremum == "minimum"
    assert summary.ascending_rank == 1


def test_summarize_draws_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="not among"):
        summarize_draws("toy", [1, 2, 3], [0.1, 0.2, 0.3], frozen_seed=0)
    with pytest.raises(ValueError, match="seeds but"):
        summarize_draws("toy", [0, 1], [0.1, 0.2, 0.3], frozen_seed=0)
    with pytest.raises(ValueError, match="at least two"):
        summarize_draws("toy", [0], [0.1], frozen_seed=0)


def test_degenerate_spread_gives_nan_z_not_a_crash() -> None:
    summary = summarize_draws("toy", range(3), [0.5, 0.5, 0.5], frozen_seed=0)
    assert math.isnan(summary.z_score)
    assert "undetermined" in summary.to_dict()["draw_robust"]


# --- 2. synthetic_harm: F17's bucketing, on a hand-planted corpus ---------


def _episode(trace_id: str, cluster_id: str, diagnosis: str) -> Episode:
    return Episode(
        trace_id=trace_id, cluster_id=cluster_id, cluster_size=2, date="2026-01-05",
        text=f"episode {trace_id} body text", facts={"diagnosis": diagnosis},
    )


def test_bucket_pairs_classifies_interference_coincidence_unrelated() -> None:
    # c0 holds e0/e1 (interference, distinct diagnoses): c1's e2 shares e0's
    # diagnosis but a different cluster (coincidence); e3 shares nothing (unrelated).
    episodes = [
        _episode("e0", "c0", "diag-a"),
        _episode("e1", "c0", "diag-b"),
        _episode("e2", "c1", "diag-a"),
        _episode("e3", "c1", "diag-c"),
    ]
    corpus = Corpus(episodes=episodes, queries=[], manifest={})
    buckets = bucket_pairs(corpus, StubEmbedder())
    assert isinstance(buckets, PairBuckets)
    # pairs: (e0,e1)=interference, (e0,e2)=coincidence, (e0,e3)=unrelated,
    # (e1,e2)=unrelated, (e1,e3)=unrelated, (e2,e3)=interference
    assert buckets.interference.size == 2
    assert buckets.coincidence.size == 1
    assert buckets.unrelated.size == 3


def test_merge_harm_rate_counts_only_interference_pairs_above_tau() -> None:
    buckets = PairBuckets(
        interference=np.array([0.70, 0.86, 0.90, 0.60]),
        coincidence=np.array([0.99]),  # must never be counted, even though it clears tau
        unrelated=np.array([0.99, 0.99]),
    )
    got = merge_harm_rate(buckets, tau=0.85)
    assert got["n_interference"] == 4
    assert got["merged"] == 2
    assert got["rate"] == pytest.approx(0.5)
    assert got["n_coincidence"] == 1 and got["n_unrelated"] == 2


def test_merge_harm_rate_on_empty_interference_is_nan_not_zero() -> None:
    buckets = PairBuckets(np.array([]), np.array([0.5]), np.array([0.5]))
    got = merge_harm_rate(buckets, tau=0.85)
    assert got["n_interference"] == 0
    assert math.isnan(got["rate"])


def test_draw_merge_harm_reproduces_frozen_seed_on_the_real_generator() -> None:
    """Seed 0 through the real generator IS the frozen corpus (F17: 79/216)."""
    pytest.importorskip("torch")
    from brain.embed.embedder import get_embedder

    got = draw_merge_harm(0, get_embedder(), tau=0.85)
    assert got["n_interference"] == 216
    assert got["merged"] == 79
    assert got["rate"] == pytest.approx(79 / 216, abs=1e-9)


# --- 3. relations_harm: wiring, on the smoke-sized relational corpus ------


def test_draw_relations_harm_returns_a_rate_in_bounds(monkeypatch) -> None:
    """Wiring-only: the smoke config is too small to match F17b's number, but
    the plumbing (encode -> label -> harm_table -> the reported fields) must
    hold together and the rate must be a well-formed fraction. Smoke-sized, so
    this exercises `draw_relations_harm` itself rather than reimplementing it,
    just against `default_relations_config` instead of the v2 scale."""
    import brain.experiments.draw_variance.relations_harm as relations_harm_module

    monkeypatch.setattr(relations_harm_module, "generate_relations", generate_relations)
    monkeypatch.setattr(
        relations_harm_module, "scaled_relations_config",
        lambda scale, **overrides: default_relations_config(smoke=True, **overrides),
    )
    got = draw_relations_harm(1, StubEmbedder())
    assert got["seed"] == 1
    assert 0.0 <= got["rate"] <= 1.0
    assert got["n_interference"] >= 0


# --- 4. runner: scratch replay and payload assembly -----------------------


def test_scratch_store_replays_instead_of_recomputing(tmp_path) -> None:
    store = ScratchStore(tmp_path, "toy_stat")
    assert store.get(0) is None
    store.put(0, {"seed": 0, "rate": 0.5})
    reloaded = ScratchStore(tmp_path, "toy_stat")  # simulates a fresh process
    assert reloaded.get(0) == {"seed": 0, "rate": 0.5}
    on_disk = json.loads((tmp_path / "toy_stat.json").read_text())
    assert on_disk == {"0": {"seed": 0, "rate": 0.5}}


def test_preregistration_shape_matches_the_task() -> None:
    prereg = build_preregistration()
    assert prereg["n_draws"] == N_DRAWS == 12
    assert prereg["seeds"] == list(range(12))
    assert prereg["frozen_seed"] == 0
    ids = {s["id"] for s in STATISTICS}
    assert ids == {
        "F17a_merge_harm_synthetic", "F17b_merge_harm_relations_v2",
        "F21_memory_precision", "F23_consolidation_delta",
    }
    # the pre-registered frozen values must match what the research doc reports
    by_id = {s["id"]: s["frozen_value"] for s in STATISTICS}
    assert by_id["F17a_merge_harm_synthetic"] == pytest.approx(0.365741, abs=1e-6)
    assert by_id["F17b_merge_harm_relations_v2"] == pytest.approx(0.470609, abs=1e-6)
    assert by_id["F21_memory_precision"] == pytest.approx(0.9498, abs=1e-4)
    assert by_id["F23_consolidation_delta"] == pytest.approx(-0.302, abs=1e-3)


def test_assemble_payload_reports_f21_blocked_without_fabricating_a_value() -> None:
    before = {"synthetic": "x", "relations_v1": "y", "relations_v2": "z"}
    results = {
        "f17a": [{"seed": s, "rate": 0.3 + 0.001 * s} for s in range(2)],
        "f17b": [{"seed": s, "rate": 0.4 + 0.001 * s} for s in range(2)],
        "f23": [{"seed": s, "delta_mean": -0.3 + 0.001 * s} for s in range(2)],
        "f21": [{"seed": 0, "score": 0.95}],  # only one draw before it blocked
    }
    import brain.experiments.draw_variance.integrity as integrity_module

    def _fake_assert_unchanged(snapshot):
        return {"directory_sha_before": snapshot, "directory_sha_after": snapshot,
                "directories_unchanged": True, "synthetic_content_sha256": "x",
                "relations_v1_content_sha256": "y", "relations_v2_content_sha256": "z"}

    original = integrity_module.assert_unchanged
    integrity_module.assert_unchanged = _fake_assert_unchanged
    try:
        payload = assemble_payload(
            before=before, results=results, f21_blocked="no ANTHROPIC_API_KEY; CLI blocked",
            started_at="t0", command="test",
        )
    finally:
        integrity_module.assert_unchanged = original
    f21 = payload["summaries"]["F21_memory_precision"]
    assert f21["status"] == "BLOCKED"
    assert "ANTHROPIC_API_KEY" in f21["reason"]
    assert f21["partial_draws"] == results["f21"]
    # the other three statistics must still summarize normally
    assert payload["summaries"]["F23_consolidation_delta"]["n_draws"] == 2
