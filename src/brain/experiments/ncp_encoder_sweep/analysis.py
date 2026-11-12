"""Aggregation for one encoder's `EncoderRun`: best cells, the strict paired table, win counts.

Mirrors `brain.experiments.gate_ncp.verdict.comparisons` in spirit (paired
cluster-BCa deltas, both win directions counted) but computed per encoder and
per k rather than once at a single frozen k, and importing `paired_delta`
rather than reimplementing it.
"""

from __future__ import annotations

from typing import Any

from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.experiments.ncp_encoder_sweep.runner import EncoderRun, paired_delta

__all__ = [
    "best_cell", "matched_table_at_k", "matched_table_all_k",
    "win_counts", "encoder_summary",
]


def _score(cell: dict[str, Any]) -> float:
    return float(cell["metrics"]["disambiguation"])


def best_cell(run: EncoderRun, kind: str) -> dict[str, Any] | None:
    """The max-over-cells cell of one weight family. Selection-optimistic; label it as such."""
    cells = [c for c in run.cells if c["weights"] == kind]
    return max(cells, key=_score) if cells else None


def matched_table_at_k(run: EncoderRun, k: int, *, n_resamples: int = DEFAULT_RESAMPLES,
                       bootstrap_seed: int = 0) -> list[dict[str, Any]]:
    """The strict one-variable ablation at one k: same gamma/alpha/query-side, local vs global."""
    locals_by_key = {(float(c["gamma"]), float(c["alpha"]), c["query_side"]): c
                     for c in run.cells if c["weights"] == "local" and int(c["k_neighbours"]) == k}
    globals_by_key = {(float(c["gamma"]), float(c["alpha"]), c["query_side"]): c
                      for c in run.cells if c["weights"] == "global"}
    matched: list[dict[str, Any]] = []
    for key in sorted(locals_by_key):
        local_cell = locals_by_key[key]
        global_cell = globals_by_key.get(key)
        if global_cell is None:
            continue
        delta = paired_delta(run.groups[local_cell["arm"]], run.groups[global_cell["arm"]],
                             n_resamples=n_resamples, bootstrap_seed=bootstrap_seed)
        matched.append({
            "encoder": run.encoder, "k_neighbours": k,
            "gamma": key[0], "alpha": key[1], "query_side": key[2],
            "local": _score(local_cell), "global": _score(global_cell),
            "delta_points": round((_score(local_cell) - _score(global_cell)) * 100.0, 4),
            "delta": delta.to_dict(),
        })
    return matched


def matched_table_all_k(run: EncoderRun, *, n_resamples: int = DEFAULT_RESAMPLES,
                        bootstrap_seed: int = 0) -> list[dict[str, Any]]:
    """`matched_table_at_k` for every k this encoder was swept over, concatenated."""
    out: list[dict[str, Any]] = []
    for k in run.k_grid:
        out.extend(matched_table_at_k(run, k, n_resamples=n_resamples, bootstrap_seed=bootstrap_seed))
    return out


def win_counts(matched: list[dict[str, Any]]) -> dict[str, Any]:
    """Both conclusive-win directions, counted. Never report only the favourable one."""
    n = len(matched)
    local_wins = sum(1 for m in matched if float(m["delta"]["lower"]) > 0.0)
    global_wins = sum(1 for m in matched if float(m["delta"]["upper"]) < 0.0)
    positive = sum(1 for m in matched if float(m["delta"]["mean"]) > 0.0)
    mean_delta = (round(100.0 * sum(float(m["delta"]["mean"]) for m in matched) / n, 4)
                 if n else None)
    return {
        "cells_compared": n,
        "local_beats_global_conclusively": local_wins,
        "global_beats_local_conclusively": global_wins,
        "positive_point_estimate": positive,
        "mean_delta_points": mean_delta,
    }


def encoder_summary(run: EncoderRun, *, primary_gamma: float, primary_alpha: float,
                    primary_k: int, primary_query_side: str,
                    n_resamples: int = DEFAULT_RESAMPLES,
                    bootstrap_seed: int = 0,
                    matched_all: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Everything the final report needs for one encoder, in one record.

    `matched_all` may be passed in (already computed by the caller) so a
    multi-encoder report does not bootstrap the same 5*70 matched pairs twice.
    """
    best_local = best_cell(run, "local")
    best_global = best_cell(run, "global")
    best_sif = best_cell(run, "sif")
    if matched_all is None:
        matched_all = matched_table_all_k(run, n_resamples=n_resamples, bootstrap_seed=bootstrap_seed)
    counts_all = win_counts(matched_all)
    # Grouped from `matched_all` rather than recomputed, so every matched pair
    # is bootstrapped exactly once.
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in matched_all:
        by_k.setdefault(int(row["k_neighbours"]), []).append(row)
    per_k = {k: win_counts(by_k.get(k, [])) for k in run.k_grid}

    primary_local = next((c for c in run.cells if c["weights"] == "local"
                          and int(c["k_neighbours"]) == primary_k
                          and float(c["gamma"]) == primary_gamma
                          and float(c["alpha"]) == primary_alpha
                          and c["query_side"] == primary_query_side), None)
    primary_global = next((c for c in run.cells if c["weights"] == "global"
                           and float(c["gamma"]) == primary_gamma
                           and float(c["alpha"]) == primary_alpha
                           and c["query_side"] == primary_query_side), None)
    # Reuse the matched row for the primary cell when it is already in
    # `matched_all` (it always is, when primary_k is in run.k_grid), rather
    # than paying for a second bootstrap of the same pair.
    primary_row = next((m for m in by_k.get(primary_k, [])
                        if m["gamma"] == primary_gamma and m["alpha"] == primary_alpha
                        and m["query_side"] == primary_query_side), None)
    if primary_row is not None:
        primary_delta = primary_row["delta"]
    elif primary_local is not None and primary_global is not None:
        primary_delta = paired_delta(run.groups[primary_local["arm"]], run.groups[primary_global["arm"]],
                                     n_resamples=n_resamples, bootstrap_seed=bootstrap_seed).to_dict()
    else:
        primary_delta = None

    best_local_vs_global_delta = None
    if best_local is not None and best_global is not None:
        best_local_vs_global_delta = paired_delta(
            run.groups[best_local["arm"]], run.groups[best_global["arm"]],
            n_resamples=n_resamples, bootstrap_seed=bootstrap_seed).to_dict()
    best_local_vs_sif_delta = None
    if best_local is not None and best_sif is not None:
        best_local_vs_sif_delta = paired_delta(
            run.groups[best_local["arm"]], run.groups[best_sif["arm"]],
            n_resamples=n_resamples, bootstrap_seed=bootstrap_seed).to_dict()

    plain_vector = next((c for c in run.cells if c["weights"] == "local"
                         and float(c["gamma"]) == 0.0 and c["query_side"] == "mean"
                         and int(c["k_neighbours"]) == run.k_grid[0]
                         and float(c["alpha"]) == 0.0), None)

    return {
        "encoder": run.encoder, "dim": run.dim,
        "plain_vector_disambiguation": None if plain_vector is None else _score(plain_vector),
        "best_local_cell": None if best_local is None else best_local["arm"],
        "best_local_disambiguation": None if best_local is None else _score(best_local),
        "best_global_cell": None if best_global is None else best_global["arm"],
        "best_global_disambiguation": None if best_global is None else _score(best_global),
        "best_sif_cell": None if best_sif is None else best_sif["arm"],
        "best_sif_disambiguation": None if best_sif is None else _score(best_sif),
        "best_local_minus_best_global": {
            "label": "MAX-OVER-CELLS on both sides; selection-optimistic, not the primary reading",
            "delta": best_local_vs_global_delta,
        },
        "best_local_minus_best_sif": {
            "label": "MAX-OVER-CELLS on both sides; selection-optimistic, not the primary reading",
            "delta": best_local_vs_sif_delta,
        },
        "primary_cell": {
            "gamma": primary_gamma, "alpha": primary_alpha, "k_neighbours": primary_k,
            "query_side": primary_query_side,
            "local": None if primary_local is None else _score(primary_local),
            "global": None if primary_global is None else _score(primary_global),
            "delta": primary_delta,
        },
        "strict_matched_win_counts_all_k": counts_all,
        "strict_matched_win_counts_by_k": per_k,
        "sibling_cosine_floor": run.diagnostics.get("sibling_cosine_floor"),
        "sibling_cosine_floor_cell": run.diagnostics.get("sibling_cosine_floor_cell"),
        "gamma_zero_identity_ok": run.diagnostics.get("gamma_zero_identity_ok"),
        "encode_seconds": run.diagnostics.get("encode_seconds"),
        "n_cells": run.diagnostics.get("n_cells"),
        "wall_clock_seconds": round(run.wall_clock, 2),
    }
