"""F23: consolidation-delta draws, the priority statistic.

"F23 is the result that would hurt most if it moved" (task instruction). Each
fresh seed needs live merge calls for the groups tau=0.75 forms on that draw
(~35 on the frozen corpus): `data/cache/consolidate/` is keyed on member
content SHAs, so new episode text always misses. Seed 0 reproduces the frozen
corpus byte-for-byte and replays entirely from cache, reproducing F23's
-0.3020767195767196 exactly.

Calls `brain.eval.consolidation_delta.measure` unchanged, restricted to
tau=0.75 (the headline threshold) to bound the live-call cost: the module's
own 5-seed loop is a no-op average over identical merges (grouping and the
merge call are both deterministic at temperature 0), so it costs no extra
LLM calls, only extra retrieval passes.
"""

from __future__ import annotations

from typing import Any

from brain.corpus.generate import GeneratorConfig, generate_corpus
from brain.eval.consolidation_delta import measure

__all__ = ["TAU", "MODE", "METRIC", "draw_consolidation_delta"]

TAU = 0.75
MODE = "destructive"
METRIC = "expected@full"


def draw_consolidation_delta(seed: int, *, tau: float = TAU, n_resamples: int = 10_000) -> dict[str, Any]:
    """One draw: fresh corpus, one live-priced consolidation pass, the headline delta."""
    corpus = generate_corpus(GeneratorConfig(seed=int(seed)))
    report = measure(corpus=corpus, thresholds=(tau,), n_resamples=n_resamples)
    name = f"{MODE}@{tau:g}"
    condition = report["conditions"][name]
    cell = condition["metrics"][METRIC]["delta_vs_before"]
    return {
        "seed": int(seed),
        "tau": float(tau),
        "delta_mean": cell["mean"],
        "delta_lower": cell["lower"],
        "delta_upper": cell["upper"],
        "n_groups": condition["stats"].get("n_groups"),
        "n_merged": condition["stats"].get("n_merged"),
        "api_calls": condition["stats"].get("spend", {}).get("api_calls"),
        "cost_usd": condition["stats"].get("spend", {}).get("cost_usd"),
    }
