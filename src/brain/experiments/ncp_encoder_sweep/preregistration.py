"""What was decided before any number from this task was read.

Fixed here, imported everywhere else in the package, and echoed verbatim into
every result JSON so a reader can check the run against what was promised
rather than against what the run happened to produce.

The (gamma, alpha, query-side) grid and the SIF grid are NOT re-decided: they
are imported from `brain.experiments.gate_ncp.runner`, the module F26 already
pre-registered them in, because this task adds two new swept dimensions
(encoder, k_neighbours) to an existing design rather than re-litigating it.
"""

from __future__ import annotations

from typing import Any

from brain.experiments.gate_ncp.runner import (
    ALPHA_GRID,
    GAMMA_GRID,
    SIF_A_GRID,
    SIF_PC_GRID,
)

__all__ = [
    "ENCODERS", "K_NEIGHBOURS_GRID", "PRIMARY_ENCODER", "PRIMARY_K", "PRIMARY_GAMMA",
    "PRIMARY_ALPHA", "PRIMARY_QUERY_SIDE", "PRIMARY_CELL", "RUNTIME_BUDGET_SECONDS",
    "GAMMA_GRID", "ALPHA_GRID", "SIF_A_GRID", "SIF_PC_GRID", "preregistration",
]

# -- THE ENCODER SWEEP. Fixed before any number from this task was seen. -------
#: The frozen baseline plus four CPU-friendly sentence-transformers spanning
#: architecture family (BERT/MPNet/BGE/GTE/E5) and training objective
#: (general STS, retrieval-contrastive, instruction-free retrieval). All five
#: load and run token-level extraction on CPU; none require a GPU or a paid API.
ENCODERS: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",   # F26's baseline, 384d, 2021
    "sentence-transformers/all-mpnet-base-v2",  # 768d, stronger STS backbone
    "BAAI/bge-small-en-v1.5",                   # 384d, retrieval-contrastive
    "thenlper/gte-small",                       # 384d, general text embedding
    "intfloat/e5-small-v2",                     # 384d, retrieval-contrastive
)
#: No query/passage prefixing is applied for e5, even though its own training
#: recipe recommends "query: " / "passage: " prefixes. NCP's weight function
#: operates on whatever tokens the tokenizer returns for the raw text, and every
#: encoder in this sweep is fed the corpus texts identically, so this keeps the
#: comparison a controlled ablation of the ENCODER rather than of a prompting
#: convention. Recorded here so the choice is visible, not silently decided.
E5_PREFIX_APPLIED = False

# -- THE NEIGHBOURHOOD SIZE SWEEP. Fixed before any number was seen. -----------
K_NEIGHBOURS_GRID: tuple[int, ...] = (4, 8, 16, 32, 64)

# -- THE PRIMARY CELL. Named before the run so the headline number is never a
# hindsight pick. Identical to F26's own PREREGISTERED_CELL (gamma=1, alpha=0.5,
# k=8, query_side=mean) so the new sweep's anchor is directly comparable to the
# old one; MiniLM is the primary encoder because it is F26's baseline.
PRIMARY_ENCODER = ENCODERS[0]
PRIMARY_K = 8
PRIMARY_GAMMA = 1.0
PRIMARY_ALPHA = 0.5
PRIMARY_QUERY_SIDE = "mean"
PRIMARY_CELL = {
    "encoder": PRIMARY_ENCODER, "weights": "local", "gamma": PRIMARY_GAMMA,
    "alpha": PRIMARY_ALPHA, "k_neighbours": PRIMARY_K, "query_side": PRIMARY_QUERY_SIDE,
}

#: Wall-clock ceiling for the WHOLE sweep (all encoders), checked between
#: encoders. All five models were verified to load and run on this machine
#: before the run started (network reachable, CPU-only); at that measured cost
#: the full sweep is expected to finish in well under half this budget, so the
#: guard is a safety stop, not the expected path. If exceeded, the report stops
#: cleanly after the encoder in progress and records which encoders were
#: skipped rather than truncating a partial encoder's cell grid.
RUNTIME_BUDGET_SECONDS = 3600


def preregistration() -> dict[str, Any]:
    """The whole pre-registered design, printed BEFORE the run and stored with it."""
    return {
        "encoders": list(ENCODERS),
        "e5_prefix_applied": E5_PREFIX_APPLIED,
        "k_neighbours_grid": list(K_NEIGHBOURS_GRID),
        "gamma_grid": list(GAMMA_GRID),
        "alpha_grid": list(ALPHA_GRID),
        "query_side_variants": ["mean", "idf"],
        "sif_a_grid": list(SIF_A_GRID),
        "sif_remove_pc_grid": list(SIF_PC_GRID),
        "primary_cell": dict(PRIMARY_CELL),
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "note": (
            "the strict one-variable ablation (same gamma, alpha, query side, k; only "
            "where df is measured changes) is reported per encoder AND per (encoder, k); "
            "both conclusive-win directions are counted; the max-over-cells number is "
            "reported alongside the primary cell and labelled as selection-optimistic"
        ),
    }
