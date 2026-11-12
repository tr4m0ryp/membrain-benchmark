"""Intervals, effect sizes, and the overlap test the kill criterion is written in.

Two different uncertainties are reported for every arm and they must not be
confused, because only one of them can carry the pre-registered verdict:

* **Across-seed interval** (`mean_ci`, Student t): how much the arm's own
  randomness moves the number. Every arm in this launch is deterministic given
  the frozen corpus, so this interval is legitimately zero-width, and a zero-width
  interval is a statement about the seed, NOT about sampling error. Using it for
  the kill criterion would manufacture a result, so `mean_ci` marks degenerate
  cases and the verdict refuses them.
* **Cluster bootstrap interval** (`brain.eval.bootstrap`): how much the number
  would move on another draw of interference clusters. That is the uncertainty
  the hypothesis is about, and it is what the verdict consumes.

`mean_ci` uses a t-interval with n - 1 degrees of freedom (small n; z would be
too narrow at n = 5 by about 25 percent) and returns a NaN interval at n = 1
rather than a fake zero-width one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "Interval",
    "mean_ci",
    "cohens_d",
    "cohens_h",
    "non_overlapping",
    "as_bounds",
]

#: Absorbs float representation error only (a 5.000-point margin can land at
#: 0.049999999999999996). Far below any difference the experiment can resolve.
_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class Interval:
    """Point estimate plus a confidence interval, with the method that made it.

    Unpacks as `mean, lower, upper` so the contract shape
    `mean_ci(values) -> mean, lower, upper` holds literally.
    """

    mean: float
    lower: float
    upper: float
    n: int = 0
    method: str = ""
    alpha: float = 0.05

    def __iter__(self):
        return iter((self.mean, self.lower, self.upper))

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    @property
    def defined(self) -> bool:
        """True when both endpoints are finite (an n = 1 interval is not)."""
        return bool(math.isfinite(self.lower) and math.isfinite(self.upper))

    @property
    def degenerate(self) -> bool:
        """A finite interval of zero width: no variability was observed at all."""
        return self.defined and self.width <= _TOL

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "n": int(self.n),
            "method": self.method,
            "alpha": float(self.alpha),
            "degenerate": self.degenerate,
        }


def mean_ci(values: Sequence[float] | Iterable[float], alpha: float = 0.05) -> Interval:
    """Mean and a two-sided Student-t confidence interval over `values`.

    Method: mean +/- t(1 - alpha/2, n - 1) * s / sqrt(n), with s the sample
    standard deviation (ddof = 1). The t quantile, not z: at n = 5 the 95 percent
    t multiplier is 2.776 against z's 1.96, and using z would narrow every
    interval by a quarter, which is exactly the direction that turns a null
    result into a claimed one.

    n = 1 returns a NaN interval (a single run carries no variance information)
    and n = 0 returns NaN throughout. Zero sample variance returns a genuine
    zero-width interval, flagged by `Interval.degenerate`.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    sample = np.asarray([float(value) for value in values], dtype=np.float64)
    n = int(sample.size)
    if n == 0:
        return Interval(math.nan, math.nan, math.nan, 0, "t", float(alpha))
    mean = float(sample.mean())
    if n == 1:
        return Interval(mean, math.nan, math.nan, 1, "t(n=1, undefined)", float(alpha))
    from scipy import stats  # imported here so `import brain.eval` stays cheap

    sem = float(sample.std(ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(1.0 - float(alpha) / 2.0, n - 1)) * sem
    return Interval(mean, mean - half, mean + half, n, "t", float(alpha))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Independent-samples Cohen's d: (mean_a - mean_b) / pooled SD.

    Pooled SD uses ddof = 1 on both samples and n_a + n_b - 2 degrees of freedom.
    Returns NaN when either sample has fewer than two values or when the pooled
    SD is zero, which is the honest answer for deterministic arms: their
    effect size in seed units is undefined, not infinite. For the proportions
    this project reports, `cohens_h` is the better-behaved companion.
    """
    x = np.asarray([float(v) for v in a], dtype=np.float64)
    y = np.asarray([float(v) for v in b], dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return math.nan
    pooled_var = (
        (x.size - 1) * float(x.var(ddof=1)) + (y.size - 1) * float(y.var(ddof=1))
    ) / float(x.size + y.size - 2)
    if not math.isfinite(pooled_var) or pooled_var <= 0.0:
        return math.nan
    return float(x.mean() - y.mean()) / math.sqrt(pooled_var)


def cohens_h(p_a: float, p_b: float) -> float:
    """Effect size for two proportions: h = 2*asin(sqrt(p_a)) - 2*asin(sqrt(p_b)).

    The variance-stabilising transform, so h stays interpretable near 0 and 1
    where a raw difference of proportions does not. Reported alongside the
    accuracy delta because disambiguation accuracy IS a proportion, and because
    a pooled-SD d over five identical seed values is undefined.
    """
    for value in (p_a, p_b):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"proportions must be in [0, 1], got {value!r}")
    return float(2.0 * math.asin(math.sqrt(float(p_a))) - 2.0 * math.asin(math.sqrt(float(p_b))))


def as_bounds(ci: Any) -> tuple[float, float]:
    """Coerce an `Interval`, a (mean, lower, upper) triple or a (lower, upper) pair.

    The triple form matches what `mean_ci` returns when unpacked, so a caller can
    pass either shape without the two silently swapping meanings.
    """
    if isinstance(ci, Interval):
        return float(ci.lower), float(ci.upper)
    values = [float(v) for v in ci]
    if len(values) == 3:
        return values[1], values[2]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError(f"expected an Interval, a (mean, lower, upper) or a (lower, upper), got {ci!r}")


def non_overlapping(ci_a: Any, ci_b: Any) -> bool:
    """True when the two intervals are disjoint. Used directly by the kill criterion.

    Boundary rule: touching endpoints COUNT AS OVERLAPPING (`upper_a == lower_b`
    is not separation), and any non-finite endpoint returns False. Both choices
    are conservative in the same direction: an undefined or exactly-adjacent
    interval can never be used to claim a separation that was not measured.

    Note for the reader of the verdict: non-overlap of two marginal intervals is
    a STRICTER test than a test on their difference, so it can miss a real
    difference. It is what H-A was pre-registered on, so it is what is computed;
    the paired difference interval is reported next to it as a diagnostic.
    """
    lower_a, upper_a = as_bounds(ci_a)
    lower_b, upper_b = as_bounds(ci_b)
    if not all(math.isfinite(value) for value in (lower_a, upper_a, lower_b, upper_b)):
        return False
    return bool(upper_a < lower_b - _TOL or upper_b < lower_a - _TOL)
