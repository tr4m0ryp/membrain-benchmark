"""Cluster-level bootstrap. The resampling unit is the interference cluster.

**Why not the query.** On the frozen corpus three queries share one episode and
m episodes share one cluster, so the 360 query indicators are not 360 independent
observations: they are 33 correlated blocks. Resampling queries individually
treats the correlated ones as independent, shrinks the standard error by roughly
sqrt(1 + (n_bar - 1) * ICC), and hands back an interval that is far too narrow.
On this corpus the query-level interval is about half the width of the honest
one, which is precisely the size of error that manufactures a "non-overlapping
CIs" verdict. Task 004 flagged this (F11d); it is non-negotiable here.

**The estimator.** Clusters carry different numbers of queries (6 to 24), so a
resample is scored with the ratio estimator: sum of the drawn clusters' correct
answers over the sum of their query counts. That is the same quantity the point
estimate reports, computed on the resampled corpus.

**The interval.** BCa (bias-corrected and accelerated), with the acceleration
from a leave-one-cluster-out jackknife. BCa is second-order accurate and, unlike
the plain percentile interval, corrects for the skew a bounded proportion
develops near 0 or 1, which is exactly where the full-context arm sits. When the
bias correction or the acceleration is undefined (a saturated statistic makes
every resample identical) it falls back to the percentile interval and says so in
`Interval.method`, rather than silently emitting a wrong endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from brain.eval.stats import Interval

__all__ = [
    "ClusterGroups",
    "DEFAULT_RESAMPLES",
    "cluster_bootstrap_ci",
    "paired_delta_ci",
    "stratified_delta_ci",
]

#: Enough that the 2.5th percentile is stable to about 0.002 on 33 clusters, and
#: still a fraction of a second because a resample is one gather plus two sums.
DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class ClusterGroups:
    """Per-cluster (sum of indicators, number of queries), the bootstrap's input."""

    keys: tuple[str, ...]
    sums: np.ndarray
    counts: np.ndarray

    @classmethod
    def from_mapping(cls, groups: Mapping[str, Sequence[float]]) -> "ClusterGroups":
        keys = tuple(sorted(str(key) for key in groups))  # sorted: ids drive the draw
        sums = np.array([math.fsum(float(v) for v in groups[key]) for key in keys], dtype=np.float64)
        counts = np.array([len(groups[key]) for key in keys], dtype=np.float64)
        if keys and float(counts.sum()) <= 0.0:
            raise ValueError("cluster groups hold no observations")
        return cls(keys=keys, sums=sums, counts=counts)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def n_observations(self) -> int:
        return int(self.counts.sum())

    @property
    def point_estimate(self) -> float:
        """Pooled mean over every observation, NaN when there are none."""
        total = float(self.counts.sum())
        return float(self.sums.sum() / total) if total > 0 else math.nan

    def means(self, draws: np.ndarray) -> np.ndarray:
        """Ratio estimator for each row of cluster indices in `draws`."""
        if len(self) == 0:
            return np.full(draws.shape[0], math.nan)
        numerator = self.sums[draws].sum(axis=1)
        denominator = self.counts[draws].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denominator > 0, numerator / denominator, math.nan)

    def jackknife(self) -> np.ndarray:
        """Leave-one-cluster-out estimates, in key order."""
        total_sum, total_count = float(self.sums.sum()), float(self.counts.sum())
        remaining = total_count - self.counts
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(remaining > 0, (total_sum - self.sums) / remaining, math.nan)


def _draws(rng: np.random.Generator, n_clusters: int, n_resamples: int) -> np.ndarray:
    """One matrix of cluster indices, drawn with replacement."""
    return rng.integers(0, n_clusters, size=(int(n_resamples), int(n_clusters)))


def _percentile(replicates: np.ndarray, alpha: float) -> tuple[float, float, str]:
    lower = float(np.quantile(replicates, alpha / 2.0))
    upper = float(np.quantile(replicates, 1.0 - alpha / 2.0))
    method = "percentile"
    if upper - lower <= 0.0:
        method = "percentile (degenerate: every resample identical)"
    return lower, upper, method


def _bca(
    theta_hat: float, replicates: np.ndarray, jackknife: np.ndarray, alpha: float
) -> tuple[float, float, str]:
    """BCa endpoints, falling back to percentile when a correction is undefined."""
    from scipy import stats

    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0 or not math.isfinite(theta_hat):
        return math.nan, math.nan, "bca (undefined)"
    if float(finite.max() - finite.min()) <= 0.0:
        # Every resample identical (a saturated metric, or one cluster). BCa is
        # formally defined here and would return the same endpoint twice, but
        # the label has to say so: a zero-width interval means no variability
        # was observed, not that the estimate is that precise.
        return _percentile(finite, alpha)
    # Mid-rank convention for the bias correction: strictly-below plus half the
    # ties. Disambiguation accuracy is discrete (a resample can only take
    # finitely many values), so ties at theta_hat are common and counting them
    # as strictly-below or not at all shifts BOTH endpoints by a whole atom of
    # the distribution. This is also what scipy.stats.bootstrap does, which is
    # what makes the endpoints cross-checkable against it.
    proportion = float(
        ((finite < theta_hat).sum() + (finite <= theta_hat).sum()) / (2.0 * finite.size)
    )
    jack = jackknife[np.isfinite(jackknife)]
    if proportion <= 0.0 or proportion >= 1.0 or jack.size < 2:
        return _percentile(finite, alpha)
    z0 = float(stats.norm.ppf(proportion))
    deviation = float(jack.mean()) - jack
    denominator = 6.0 * float((deviation**2).sum()) ** 1.5
    acceleration = float((deviation**3).sum()) / denominator if denominator > 0.0 else 0.0
    endpoints: list[float] = []
    for tail in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = z0 + float(stats.norm.ppf(tail))
        scale = 1.0 - acceleration * z
        if not math.isfinite(scale) or scale <= 0.0:
            return _percentile(finite, alpha)
        endpoints.append(float(stats.norm.cdf(z0 + z / scale)))
    if not endpoints[0] < endpoints[1]:
        return _percentile(finite, alpha)
    lower = float(np.quantile(finite, endpoints[0]))
    upper = float(np.quantile(finite, endpoints[1]))
    return lower, upper, "bca"


def _interval(
    theta_hat: float,
    replicates: np.ndarray,
    jackknife: np.ndarray,
    *,
    alpha: float,
    n: int,
    label: str,
) -> Interval:
    lower, upper, method = _bca(theta_hat, replicates, jackknife, alpha)
    return Interval(theta_hat, lower, upper, n, f"{label} {method} B={replicates.size}", alpha)


def cluster_bootstrap_ci(
    groups: Mapping[str, Sequence[float]] | ClusterGroups,
    *,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> Interval:
    """Bootstrap CI for a mean, resampling CLUSTERS with replacement.

    `groups` maps cluster id to that cluster's per-query indicators. The interval
    is what the arm's accuracy would do under another draw of clusters from the
    same generator, which is the uncertainty H-A is stated in.
    """
    data = groups if isinstance(groups, ClusterGroups) else ClusterGroups.from_mapping(groups)
    if len(data) == 0:
        return Interval(math.nan, math.nan, math.nan, 0, "cluster-bootstrap (no clusters)", alpha)
    generator = rng if rng is not None else np.random.default_rng(0)
    draws = _draws(generator, len(data), n_resamples)
    return _interval(
        data.point_estimate,
        data.means(draws),
        data.jackknife(),
        alpha=alpha,
        n=len(data),
        label="cluster-bootstrap",
    )


def _aligned(a: Any, b: Any) -> tuple[ClusterGroups, ClusterGroups]:
    left = a if isinstance(a, ClusterGroups) else ClusterGroups.from_mapping(a)
    right = b if isinstance(b, ClusterGroups) else ClusterGroups.from_mapping(b)
    if left.keys != right.keys:
        raise ValueError("paired bootstrap needs the same clusters on both arms")
    if not np.array_equal(left.counts, right.counts):
        raise ValueError("paired bootstrap needs the same query counts per cluster")
    return left, right


def paired_delta_ci(
    a: Mapping[str, Sequence[float]] | ClusterGroups,
    b: Mapping[str, Sequence[float]] | ClusterGroups,
    *,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> Interval:
    """CI for `mean(a) - mean(b)` with both arms scored on the SAME resampled clusters.

    Paired because the arms answer the same queries: sharing the draw removes the
    corpus variance common to both and gives a far tighter, still honest interval
    than comparing two marginal intervals. Reported as a diagnostic next to the
    pre-registered non-overlap test, never in place of it.
    """
    left, right = _aligned(a, b)
    generator = rng if rng is not None else np.random.default_rng(0)
    draws = _draws(generator, len(left), n_resamples)
    delta = left.point_estimate - right.point_estimate
    return _interval(
        delta,
        left.means(draws) - right.means(draws),
        left.jackknife() - right.jackknife(),
        alpha=alpha,
        n=len(left),
        label="paired-cluster-bootstrap",
    )


def stratified_delta_ci(
    a: Mapping[str, Sequence[float]] | ClusterGroups,
    b: Mapping[str, Sequence[float]] | ClusterGroups,
    *,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> Interval:
    """CI for `mean(a) - mean(b)` when a and b hold DISJOINT clusters.

    The accuracy-vs-cluster-size curve compares one size against another, and a
    cluster has exactly one size, so those two groups share no cluster and cannot
    be paired. Each stratum is resampled independently and the difference is
    taken per resample.
    """
    left = a if isinstance(a, ClusterGroups) else ClusterGroups.from_mapping(a)
    right = b if isinstance(b, ClusterGroups) else ClusterGroups.from_mapping(b)
    overlap = sorted(set(left.keys) & set(right.keys))
    if overlap:
        raise ValueError(f"stratified bootstrap needs disjoint clusters; shared: {overlap}")
    if len(left) == 0 or len(right) == 0:
        return Interval(math.nan, math.nan, math.nan, 0, "stratified-bootstrap (empty)", alpha)
    generator = rng if rng is not None else np.random.default_rng(0)
    replicates = left.means(_draws(generator, len(left), n_resamples)) - right.means(
        _draws(generator, len(right), n_resamples)
    )
    # Jackknife over the union: dropping a cluster moves whichever stratum owns it.
    jackknife = np.concatenate(
        [left.jackknife() - right.point_estimate, left.point_estimate - right.jackknife()]
    )
    return _interval(
        left.point_estimate - right.point_estimate,
        replicates,
        jackknife,
        alpha=alpha,
        n=len(left) + len(right),
        label="stratified-cluster-bootstrap",
    )
