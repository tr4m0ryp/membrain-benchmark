"""Draw-to-draw variance for three single-draw results: F17, F21, F23.

Only the threshold ceiling (F22/F25) ever got the twelve-draw treatment, and
that is how F22's 0.8686 was caught as the MAXIMUM of eight equivalent draws
rather than a representative number. This package applies the same method to
the three results that never had it: F17's merge-harm rate (on both the
synthetic and relations-v2 corpora), F21's memory precision, and F23's
consolidation delta. F23 is the priority: it is the result that would hurt
most if it moved.

Method: generate N independent corpus draws by varying ONLY the generator
seed (`brain.corpus.generate`, `brain.corpus.relations.generate`), recompute
the statistic per draw, and report where the historically-reported (frozen
seed 0) value sits in that distribution: mean, sd, min, max, and the frozen
seed's rank and z-score. The frozen corpora on disk are never modified; every
draw is generated in memory and, where persisted at all, written under a
scratch directory. See `preregistration.py` for what is pre-registered before
any number is read, and `runner.py` for the orchestration.
"""

from __future__ import annotations

from brain.experiments.draw_variance.preregistration import (
    N_DRAWS,
    SEEDS,
    STATISTICS,
    build_preregistration,
)
from brain.experiments.draw_variance.stats_summary import DrawSummary, summarize_draws

__all__ = [
    "N_DRAWS",
    "SEEDS",
    "STATISTICS",
    "build_preregistration",
    "DrawSummary",
    "summarize_draws",
]
