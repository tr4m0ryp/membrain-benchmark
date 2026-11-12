"""Cell specs for one encoder's grid: local (over every swept k), global, SIF.

Structurally identical to `brain.experiments.gate_ncp.runner.cell_specs`, split
out and reparameterised over `k_grid` because that module hardcodes
`K_GRID = (4, 8, 16)`, which is exactly the frozen knob this task exists to
widen. The (gamma, alpha, query-side, SIF) values themselves are NOT
reinvented: they are imported from `gate_ncp.runner` so the grid this task
adds k and encoder to is bit-for-bit the one F26 already pre-registered.
"""

from __future__ import annotations

from typing import Any, Sequence

from brain.experiments.gate_ncp.runner import GAMMA_GRID, SIF_A_GRID, SIF_PC_GRID

__all__ = ["local_cell_specs", "global_cell_specs", "sif_cell_specs", "all_cell_specs"]


def local_cell_specs(k_grid: Sequence[int]) -> list[dict[str, Any]]:
    """Local (neighbourhood-conditional) specs, one family per swept k."""
    specs: list[dict[str, Any]] = []
    for k in k_grid:
        for gamma in GAMMA_GRID:
            for side in ("mean", "idf"):
                specs.append({"weights": "local", "gamma": gamma, "k_neighbours": int(k),
                              "query_side": side, "sif_a": 0.0, "remove_pc": False})
    return specs


def global_cell_specs(*, k_label: int) -> list[dict[str, Any]]:
    """Global-df specs. Weight computation ignores k; `k_label` only tags the record."""
    specs: list[dict[str, Any]] = []
    for gamma in GAMMA_GRID:
        for side in ("mean", "idf"):
            specs.append({"weights": "global", "gamma": gamma, "k_neighbours": int(k_label),
                          "query_side": side, "sif_a": 0.0, "remove_pc": False})
    return specs


def sif_cell_specs(*, k_label: int) -> list[dict[str, Any]]:
    """SIF specs. Weight computation ignores k and gamma; `k_label` only tags the record."""
    specs: list[dict[str, Any]] = []
    for a in SIF_A_GRID:
        for pc in SIF_PC_GRID:
            for side in ("mean", "idf"):
                specs.append({"weights": "sif", "gamma": 0.0, "k_neighbours": int(k_label),
                              "query_side": side, "sif_a": a, "remove_pc": pc})
    return specs


def all_cell_specs(k_grid: Sequence[int], *, k_label: int) -> list[dict[str, Any]]:
    """Every spec this task evaluates for one encoder, order fixed and data-independent."""
    return (local_cell_specs(k_grid) + global_cell_specs(k_label=k_label)
            + sif_cell_specs(k_label=k_label))
