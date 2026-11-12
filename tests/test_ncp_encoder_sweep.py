"""Sweep-the-winner contract: pre-registration is fixed, gamma=0 holds for every
encoder, the strict local-vs-global ablation counts both win directions.

Mirrors `tests/test_ncp.py`'s load-bearing check (`gamma = 0` must reproduce
mean pooling byte-identically) but parametrised over every encoder in the
pre-registered list, because F26 only ever proved that identity for MiniLM.

Everything here is offline once the five small `sentence-transformers`
checkpoints are cached locally; no LLM transport is imported or reachable.
"""

from __future__ import annotations

import pytest

from brain.corpus import load_corpus
from brain.corpus.relations.loader import SYNTHETIC_CONTENT_SHA256
from brain.corpus.types import Corpus
from brain.experiments.gate_ncp.runner import ALPHA_GRID, GAMMA_GRID
from brain.experiments.ncp_encoder_sweep.analysis import (
    best_cell,
    encoder_summary,
    matched_table_at_k,
    win_counts,
)
from brain.experiments.ncp_encoder_sweep.preregistration import (
    ENCODERS,
    K_NEIGHBOURS_GRID,
    PRIMARY_ALPHA,
    PRIMARY_CELL,
    PRIMARY_GAMMA,
    PRIMARY_K,
    PRIMARY_QUERY_SIDE,
    RUNTIME_BUDGET_SECONDS,
    preregistration,
)
from brain.experiments.ncp_encoder_sweep.specs import (
    all_cell_specs,
    global_cell_specs,
    local_cell_specs,
    sif_cell_specs,
)
from brain.experiments.ncp_encoder_sweep.runner import run_encoder

_TEST_N_RESAMPLES = 200  # small: these tests check shape and sign, not CI precision


@pytest.fixture(scope="module")
def tiny_corpus() -> Corpus:
    """The frozen corpus, all episodes (cluster structure must stay intact), few queries."""
    corpus = load_corpus()
    queries = sorted(corpus.queries, key=lambda q: str(q.query_id))[:10]
    return Corpus(corpus.episodes, queries, corpus.manifest)


# -- pre-registration: fixed before any number is read -------------------------
def test_frozen_corpus_is_unchanged():
    assert load_corpus().content_hash == SYNTHETIC_CONTENT_SHA256


def test_preregistration_matches_the_documented_constants():
    reg = preregistration()
    assert reg["encoders"] == list(ENCODERS)
    assert reg["encoders"][0] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(ENCODERS) >= 4  # baseline plus at least three CPU-friendly encoders
    assert reg["k_neighbours_grid"] == list(K_NEIGHBOURS_GRID)
    assert set(K_NEIGHBOURS_GRID) == {4, 8, 16, 32, 64}
    assert reg["primary_cell"] == dict(PRIMARY_CELL)
    assert reg["runtime_budget_seconds"] == RUNTIME_BUDGET_SECONDS
    # The gamma/alpha/query-side grid is F26's, imported, not reinvented.
    assert reg["gamma_grid"] == list(GAMMA_GRID)
    assert reg["alpha_grid"] == list(ALPHA_GRID)


def test_primary_cell_is_f26s_preregistered_cell():
    # Identical anchor to F26's own PREREGISTERED_CELL, so the two sweeps are
    # directly comparable and neither number is a hindsight pick.
    assert PRIMARY_GAMMA == 1.0 and PRIMARY_ALPHA == 0.5
    assert PRIMARY_K == 8 and PRIMARY_QUERY_SIDE == "mean"


# -- cell-spec generation -------------------------------------------------------
def test_local_cell_specs_count_scales_with_k_grid():
    k_grid = (4, 8, 16)
    specs = local_cell_specs(k_grid)
    assert len(specs) == len(k_grid) * len(GAMMA_GRID) * 2
    assert {s["k_neighbours"] for s in specs} == set(k_grid)
    assert all(s["weights"] == "local" for s in specs)


def test_global_and_sif_cell_specs_are_k_independent_in_content():
    specs_a = global_cell_specs(k_label=4)
    specs_b = global_cell_specs(k_label=64)
    stripped = lambda rows: [{k: v for k, v in s.items() if k != "k_neighbours"} for s in rows]
    assert stripped(specs_a) == stripped(specs_b)
    assert len(specs_a) == len(GAMMA_GRID) * 2
    sif_specs = sif_cell_specs(k_label=8)
    assert len(sif_specs) == 2 * 2 * 2  # a_grid x remove_pc x query_side
    assert all(s["gamma"] == 0.0 for s in sif_specs)


def test_all_cell_specs_is_the_union():
    k_grid = (4, 8)
    specs = all_cell_specs(k_grid, k_label=4)
    assert len(specs) == (len(k_grid) * len(GAMMA_GRID) * 2 + len(GAMMA_GRID) * 2 + 2 * 2 * 2)
    assert {s["weights"] for s in specs} == {"local", "global", "sif"}


# -- gamma=0 identity, for EVERY pre-registered encoder -------------------------
@pytest.mark.parametrize("encoder", ENCODERS)
def test_gamma_zero_reproduces_mean_pooling_for_every_encoder(tiny_corpus, encoder):
    """The free correctness test the task's hard constraints require: asserted,
    not assumed, per encoder. `run_encoder` itself raises on this failing; this
    test additionally checks the diagnostic the report relies on to decide that.
    """
    run = run_encoder(encoder, tiny_corpus, seeds=(0,), k_grid=(PRIMARY_K,),
                      n_resamples=_TEST_N_RESAMPLES)
    assert run.diagnostics["gamma_zero_identity_ok"] is True
    assert run.diagnostics["gamma_zero_identity_failures"] == []
    assert run.diagnostics["n_cells"] > 0
    # Every gamma=0 local/global cell, at every alpha, must equal plain_vector.
    zero_cells = [c for c in run.cells if float(c["gamma"]) == 0.0 and c["weights"] in ("local", "global")]
    base = zero_cells[0]["metrics"]["disambiguation"]
    assert all(abs(c["metrics"]["disambiguation"] - base) < 1e-12 for c in zero_cells)


def test_sibling_cosine_floor_is_reported_and_bounded(tiny_corpus):
    run = run_encoder(ENCODERS[0], tiny_corpus, seeds=(0,), k_grid=(4, 8), n_resamples=_TEST_N_RESAMPLES)
    floor = run.diagnostics["sibling_cosine_floor"]
    assert floor is not None
    assert -1.0 - 1e-6 <= floor <= 1.0 + 1e-6
    assert run.diagnostics["sibling_cosine_floor_cell"]


# -- the strict one-variable ablation: both directions counted ------------------
def test_matched_table_pairs_every_gamma_alpha_query_side_cell(tiny_corpus):
    run = run_encoder(ENCODERS[0], tiny_corpus, seeds=(0,), k_grid=(4,), n_resamples=_TEST_N_RESAMPLES)
    matched = matched_table_at_k(run, 4, n_resamples=_TEST_N_RESAMPLES)
    assert len(matched) == len(GAMMA_GRID) * len(ALPHA_GRID) * 2  # gamma x alpha x query_side
    for row in matched:
        assert row["delta"]["mean"] == pytest.approx((row["local"] - row["global"]))
    # gamma=0 rows must have an exactly-zero delta: both sides are plain_vector.
    zero_rows = [r for r in matched if r["gamma"] == 0.0]
    assert zero_rows and all(abs(r["delta"]["mean"]) < 1e-12 for r in zero_rows)


def test_win_counts_reports_both_directions_never_only_one(tiny_corpus):
    run = run_encoder(ENCODERS[0], tiny_corpus, seeds=(0,), k_grid=(4,), n_resamples=_TEST_N_RESAMPLES)
    matched = matched_table_at_k(run, 4, n_resamples=_TEST_N_RESAMPLES)
    counts = win_counts(matched)
    assert counts["cells_compared"] == len(matched)
    assert 0 <= counts["local_beats_global_conclusively"] <= counts["cells_compared"]
    assert 0 <= counts["global_beats_local_conclusively"] <= counts["cells_compared"]
    # The two conclusive-win counts can never both exceed the total together
    # (a cell cannot win conclusively for both sides at once).
    assert (counts["local_beats_global_conclusively"]
           + counts["global_beats_local_conclusively"]) <= counts["cells_compared"]


def test_best_cell_is_a_max_over_cells_not_the_primary_cell(tiny_corpus):
    run = run_encoder(ENCODERS[0], tiny_corpus, seeds=(0,), k_grid=(4, 8), n_resamples=_TEST_N_RESAMPLES)
    best_local = best_cell(run, "local")
    all_local = [c for c in run.cells if c["weights"] == "local"]
    assert best_local["metrics"]["disambiguation"] == max(
        c["metrics"]["disambiguation"] for c in all_local)


def test_encoder_summary_carries_primary_and_best_readings_separately(tiny_corpus):
    run = run_encoder(ENCODERS[0], tiny_corpus, seeds=(0,), k_grid=(4, 8), n_resamples=_TEST_N_RESAMPLES)
    summary = encoder_summary(run, primary_gamma=1.0, primary_alpha=0.5, primary_k=8,
                              primary_query_side="mean", n_resamples=_TEST_N_RESAMPLES)
    assert summary["primary_cell"]["gamma"] == 1.0 and summary["primary_cell"]["k_neighbours"] == 8
    assert summary["best_local_disambiguation"] >= summary["primary_cell"]["local"] - 1e-9
    assert "MAX-OVER-CELLS" in summary["best_local_minus_best_global"]["label"]
    assert summary["gamma_zero_identity_ok"] is True
    assert set(summary["strict_matched_win_counts_by_k"]) == {4, 8}
