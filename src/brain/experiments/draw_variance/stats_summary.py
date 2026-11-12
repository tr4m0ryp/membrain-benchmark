"""Where the frozen seed sits among N independent draws.

The whole point of this package (task instruction, verbatim): "Report per
statistic: mean, sd, min, max, the 12 raw values, and WHERE THE FROZEN SEED
SITS (its rank among the draws and its z-score). That last number is the
whole point." This module is the one place that number is computed, so every
statistic reads it the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["DrawSummary", "summarize_draws"]


@dataclass(frozen=True, slots=True)
class DrawSummary:
    """Descriptive stats over N draws, plus the frozen seed's position in them."""

    statistic: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]  # in seed order
    frozen_seed: int
    frozen_value: float
    mean: float
    sd: float
    minimum: float
    maximum: float
    ascending_rank: int  # 1 = smallest of N, N = largest
    z_score: float
    is_extremum: str  # "minimum" | "maximum" | ""

    @property
    def n(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "n_draws": self.n,
            "seeds": list(self.seeds),
            "values": list(self.values),
            "frozen_seed": self.frozen_seed,
            "frozen_value": self.frozen_value,
            "mean": self.mean,
            "sd": self.sd,
            "min": self.minimum,
            "max": self.maximum,
            "frozen_ascending_rank": self.ascending_rank,
            "frozen_rank_label": f"{self.ascending_rank} of {self.n} (ascending)",
            "frozen_z_score": self.z_score,
            "frozen_is_extremum": self.is_extremum,
            "draw_robust": self._verdict(),
        }

    def _verdict(self) -> str:
        if not math.isfinite(self.z_score):
            return "undetermined (degenerate spread across draws)"
        if self.is_extremum:
            return f"DRAW-LUCKY: the frozen value is the {self.is_extremum} of {self.n} draws"
        if abs(self.z_score) >= 1.0:
            return f"draw-sensitive: frozen value is {self.z_score:+.2f} sd from the draw mean"
        return "draw-robust: frozen value sits near the centre of the draw distribution"


def summarize_draws(
    statistic: str, seeds: Sequence[int], values: Sequence[float], *, frozen_seed: int = 0
) -> DrawSummary:
    """Build a `DrawSummary` from parallel `seeds` and `values` sequences.

    `frozen_seed` must be present in `seeds`: the frozen result IS one of the
    draws (seed 0 reproduces the frozen corpus byte-for-byte), never a value
    computed separately and then compared in.
    """
    seeds_t = tuple(int(s) for s in seeds)
    values_t = tuple(float(v) for v in values)
    if len(seeds_t) != len(values_t):
        raise ValueError(f"{len(seeds_t)} seeds but {len(values_t)} values")
    if len(seeds_t) < 2:
        raise ValueError("need at least two draws to report a spread")
    if frozen_seed not in seeds_t:
        raise ValueError(f"frozen_seed={frozen_seed} is not among the drawn seeds {seeds_t}")
    array = np.array(values_t, dtype=np.float64)
    frozen_index = seeds_t.index(int(frozen_seed))
    frozen_value = float(array[frozen_index])
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    minimum, maximum = float(array.min()), float(array.max())
    ascending_rank = int(np.sum(array <= frozen_value))
    z_score = (frozen_value - mean) / sd if sd > 0.0 else math.nan
    is_extremum = ""
    if math.isclose(frozen_value, maximum, rel_tol=0.0, abs_tol=1e-12):
        is_extremum = "maximum"
    elif math.isclose(frozen_value, minimum, rel_tol=0.0, abs_tol=1e-12):
        is_extremum = "minimum"
    return DrawSummary(
        statistic=statistic, seeds=seeds_t, values=values_t,
        frozen_seed=int(frozen_seed), frozen_value=frozen_value,
        mean=mean, sd=sd, minimum=minimum, maximum=maximum,
        ascending_rank=ascending_rank, z_score=z_score, is_extremum=is_extremum,
    )
