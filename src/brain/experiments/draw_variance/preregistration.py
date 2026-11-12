"""What is fixed BEFORE any draw statistic is read.

`build_preregistration` is written to the result file first, unconditionally,
so a reader can see the committed N, seeds and statistic list independent of
whether every statistic later completed. Nothing here depends on a measured
value.
"""

from __future__ import annotations

from typing import Any

__all__ = ["N_DRAWS", "SEEDS", "MERGE_TAU", "CONSOLIDATION_TAU", "STATISTICS",
           "build_preregistration"]

#: Matches the precedent (F25's twelve-draw decomposition of the threshold
#: ceiling; `scripts/task-007/draw_variability.py` used N=8 at v1 size).
N_DRAWS = 12
SEEDS: tuple[int, ...] = tuple(range(N_DRAWS))
#: Seed 0 reproduces the frozen corpora byte-for-byte (verified below), so it
#: is both a draw AND the historically-reported number.
FROZEN_SEED = 0

#: F17's shipped merge threshold on both corpora.
MERGE_TAU = 0.85
#: F23's headline threshold.
CONSOLIDATION_TAU = 0.75

STATISTICS: tuple[dict[str, Any], ...] = (
    {
        "id": "F17a_merge_harm_synthetic",
        "label": "F17 merge harm, synthetic corpus, tau=0.85",
        "frozen_value": 79 / 216,
        "frozen_value_reported": "36.57%",
        "corpus": "synthetic (data/corpus/synthetic, SHA 25c2953a...)",
        "draw_generator": "brain.corpus.generate.generate_corpus(GeneratorConfig(seed=k))",
        "llm_calls_required": False,
    },
    {
        "id": "F17b_merge_harm_relations_v2",
        "label": "F17 merge harm, relations-v2 corpus, tau=0.85, long regime",
        "frozen_value": 1345 / 2858,
        "frozen_value_reported": "47.06%",
        "corpus": "relations-v2 (data/corpus/relations-v2, SHA 24ff2689...)",
        "draw_generator": (
            "brain.corpus.relations.generate.generate_relations("
            "scaled_relations_config(3, seed=k))"
        ),
        "llm_calls_required": False,
    },
    {
        "id": "F21_memory_precision",
        "label": "F21 memory precision (entailment judge)",
        "frozen_value": 0.9497716894977168,
        "frozen_value_reported": "0.9498 [0.9098, 0.9770]",
        "corpus": "synthetic (data/corpus/synthetic, SHA 25c2953a...)",
        "draw_generator": "brain.corpus.generate.generate_corpus(GeneratorConfig(seed=k))",
        "llm_calls_required": True,
        "llm_note": (
            "extraction (1 call/transcript) and entailment judging (~7 calls/transcript) "
            "over 30 transcripts; only seed 0 (byte-identical to the frozen corpus) can "
            "replay from data/cache/entailment/, every other seed is an uncached live call"
        ),
    },
    {
        "id": "F23_consolidation_delta",
        "label": "F23 consolidation delta, destructive@tau=0.75, expected@full",
        "frozen_value": -0.3020767195767196,
        "frozen_value_reported": "-0.302",
        "corpus": "synthetic (data/corpus/synthetic, SHA 25c2953a...)",
        "draw_generator": "brain.corpus.generate.generate_corpus(GeneratorConfig(seed=k))",
        "llm_calls_required": True,
        "llm_note": (
            "one merge call per group at tau=0.75 (~35 groups on the frozen corpus); only "
            "seed 0 can replay from data/cache/consolidate/, every other seed is live. "
            "Priority statistic: the one that would hurt most if it moved."
        ),
    },
)


def build_preregistration() -> dict[str, Any]:
    return {
        "n_draws": N_DRAWS,
        "seeds": list(SEEDS),
        "frozen_seed": FROZEN_SEED,
        "statistics": [dict(s) for s in STATISTICS],
        "method": (
            "generate N independent corpus draws varying ONLY the generator seed; "
            "recompute the statistic on each draw; report mean, sd (ddof=1), min, max, "
            "the N raw values, and the frozen seed's rank and z-score among them"
        ),
        "hard_constraints": [
            "frozen corpora on disk (data/corpus/synthetic, relations, relations-v2) are "
            "never written to; every draw is generated in memory or under a scratch "
            "directory, and the frozen content SHAs are asserted unchanged before and "
            "after the run",
            "all intervals reported alongside a point estimate are cluster-level BCa "
            "bootstrap via brain.eval.bootstrap, never query-level",
            "no number is fabricated or extrapolated; a statistic that needs uncached "
            "LLM calls and cannot get them is marked BLOCKED with the precise reason",
        ],
    }
