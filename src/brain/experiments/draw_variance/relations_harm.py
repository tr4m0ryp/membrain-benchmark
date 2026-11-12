"""F17b: merge-harm rate on independent draws of the relations-v2-scale corpus.

Reuses `brain.experiments.gate_threshold` unchanged, the same module the task
007 driver ran against the frozen v2 split: `harm_table` on the `derivable`
population's `long` regime cell, at tau=0.85. Verified against the frozen v2
split before any draw runs: reproduces `distinct_events_destroyed_rate` =
1345/2858 = 47.0609%, matching `results/threshold-ceiling/threshold-20260725T185408Z.json`
to sixteen significant figures.
"""

from __future__ import annotations

from typing import Any

from brain.corpus.relations import INTERFERENCE
from brain.corpus.relations.generate import generate_relations, scaled_relations_config
from brain.experiments.gate_threshold import ceiling as C
from brain.experiments.gate_threshold import distributions as D

__all__ = ["MERGE_TAU", "V2_SCALE", "REGIME", "draw_relations_harm"]

MERGE_TAU = 0.85
V2_SCALE = 3
REGIME = "long"


def draw_relations_harm(
    seed: int, embedder: Any, *, tau: float = MERGE_TAU, scale: int = V2_SCALE
) -> dict[str, Any]:
    """One draw: generate the v2-scale relational corpus at `seed`, score the ceiling harm."""
    config = scaled_relations_config(scale, seed=int(seed))
    corpus = generate_relations(config)
    ids, vectors = D.encode_corpus(corpus, embedder)
    gram = D.gram_matrix(vectors)
    labelled = D.all_pair_rows(corpus, ids, gram)
    derivable = D.population(labelled, "derivable")
    long_rows = [row for row in derivable if row.regime == REGIME]
    row = C.harm_table(long_rows, taus=(tau,))[0]
    n_interference = int(D.by_class(long_rows, INTERFERENCE).size)
    return {
        "seed": int(seed),
        "tau": float(tau),
        "rate": row["distinct_events_destroyed_rate"],
        "merged": row["distinct_events_destroyed"],
        "n_interference": n_interference,
        "n_episodes": len(corpus.episodes),
        "n_slots": config.n_slots,
    }
