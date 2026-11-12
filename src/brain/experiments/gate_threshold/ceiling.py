"""The ceiling every merge-above-threshold policy sits under.

**Exact, not swept.** For `predict SUPERSESSION iff s >= tau`,
`BA(tau) = 0.5 + 0.5 * [F_I - F_S](tau^-)`; for the opposite orientation
`0.5 + 0.5 * [F_S - F_I](tau)`. Both ECDFs are step functions changing only at
observed scores, so the maximum over ALL real tau is attained at an observed score
and `ceiling = 0.5 + D / 2` with `D` the two-sample Kolmogorov-Smirnov statistic.
There is therefore no grid: `tau*` cannot be a grid artifact, and the plateau over
which the optimum holds is reported beside it. Both orientations are searched
because the spec WITHDREW the directional prediction.

**Two error floors.** `threshold_bayes_error = 1 - ceiling` is the error of the
best single threshold under equal priors, and K1 is pre-registered on that alone.
`overlap_bayes_error = 0.5 * integral min(p_S, p_I)` is the floor for EVERY
measurable `f: s -> action`, estimated from continuous densities rather than read
off the sweep. That is Theorem 1's "for every f" clause, and it is a companion
number, never a K1 input.

**The bootstrap** resamples clusters and REFITS the ceiling on each resample.
`cluster_bootstrap_ci` cannot be used directly: it estimates a mean of
per-observation indicators and the ceiling is a maximum over tau. `_bca` is that
module's pure endpoint math, reused rather than reimplemented (rules.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Sequence

import numpy as np

from brain.corpus.relations import COINCIDENCE, DUPLICATE, INTERFERENCE, SUPERSESSION
from brain.eval.bootstrap import DEFAULT_RESAMPLES, _bca
from brain.eval.stats import Interval
from brain.experiments.gate_threshold.distributions import PairRow, by_class

__all__ = [
    "K1_THRESHOLD", "K1_TEXT", "HARM_TAUS", "COST_RATIOS", "Scores", "Ceiling",
    "best_threshold", "overlap_bayes_error", "ceiling_interval", "loco_balanced_accuracy",
    "harm_table", "expected_loss"]

#: Pre-registered before any measurement (spec, Hypothesis & Kill Criteria).
K1_THRESHOLD = 0.90
K1_TEXT = (
    "K1 fires if the best single-threshold classifier exceeds 0.90 balanced accuracy on the "
    "new corpus. Then a threshold is sufficient, I1 is unnecessary, and the post-write plan "
    "collapses to I2 and I3."
)
#: F17's thresholds all lie in this grid, so the old table stays comparable.
HARM_TAUS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10)) + (0.99,)
#: c_S / c_I. The Formal Model says c_S >> c_I; 1.0 is the symmetric control.
COST_RATIOS: tuple[float, ...] = (1.0, 5.0, 20.0)
Scores = Sequence[float] | np.ndarray  # any array-like; entry points coerce to float64


@dataclass(frozen=True, slots=True)
class Ceiling:
    """The best achievable single-threshold rule, and what it costs at the optimum."""

    balanced_accuracy: float
    tau_star: float
    tau_midpoint: float
    plateau: tuple[float, float]
    orientation: str
    tpr: float
    tnr: float
    n_supersession: int
    n_interference: int
    ks_statistic: float

    @property
    def threshold_bayes_error(self) -> float:
        return 1.0 - float(self.balanced_accuracy)

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return {**data, "plateau": [float(self.plateau[0]), float(self.plateau[1])],
                "threshold_bayes_error": self.threshold_bayes_error}


def _best_ba(sup: np.ndarray, inter: np.ndarray) -> tuple[float, int, str, np.ndarray]:
    """(best BA, index into the candidate thresholds, orientation, candidates)."""
    candidates = np.unique(np.concatenate([sup, inter]))
    sup_sorted, int_sorted = np.sort(sup), np.sort(inter)
    ns, ni = float(sup_sorted.size), float(int_sorted.size)
    below_s = np.searchsorted(sup_sorted, candidates, side="left") / ns
    below_i = np.searchsorted(int_sorted, candidates, side="left") / ni
    upto_s = np.searchsorted(sup_sorted, candidates, side="right") / ns
    upto_i = np.searchsorted(int_sorted, candidates, side="right") / ni
    high = 0.5 * ((1.0 - below_s) + below_i)  # predict S iff s >= tau
    low = 0.5 * (upto_s + (1.0 - upto_i))  # predict S iff s <= tau
    i_high, i_low = int(np.argmax(high)), int(np.argmax(low))
    if float(high[i_high]) >= float(low[i_low]):
        return float(high[i_high]), i_high, "s >= tau -> SUPERSESSION", candidates
    return float(low[i_low]), i_low, "s <= tau -> SUPERSESSION", candidates


def _ceiling_value(sup: np.ndarray, inter: np.ndarray) -> float:
    """Scalar ceiling for the bootstrap loop. NaN when a class is empty."""
    return math.nan if not (sup.size and inter.size) else _best_ba(sup, inter)[0]


def best_threshold(sup: Scores, inter: Scores) -> Ceiling:
    """The exact ceiling, its tau, its plateau, and both rates at the optimum."""
    sup, inter = np.asarray(sup, dtype=np.float64), np.asarray(inter, dtype=np.float64)
    if sup.size == 0 or inter.size == 0:
        raise ValueError("the ceiling needs both SUPERSESSION and INTERFERENCE pairs")
    best, index, orientation, candidates = _best_ba(sup, inter)
    tau = float(candidates[index])
    if orientation.startswith("s >="):
        tpr, tnr = float(np.mean(sup >= tau)), float(np.mean(inter < tau))
        # BA is constant for tau in (previous observed value, tau].
        plateau = (float(candidates[index - 1]) if index > 0 else tau, tau)
    else:
        tpr, tnr = float(np.mean(sup <= tau)), float(np.mean(inter > tau))
        nxt = float(candidates[index + 1]) if index + 1 < candidates.size else tau
        plateau = (tau, nxt)
    return Ceiling(
        balanced_accuracy=best, tau_star=tau, tau_midpoint=0.5 * (plateau[0] + plateau[1]),
        plateau=plateau, orientation=orientation, tpr=tpr, tnr=tnr,
        n_supersession=int(sup.size), n_interference=int(inter.size),
        ks_statistic=float(2.0 * (best - 0.5)),
    )


def overlap_bayes_error(sup: Scores, inter: Scores, *, grid: int = 4096) -> dict[str, Any]:
    """`0.5 * integral min(p_S, p_I)`: the error floor of EVERY rule `f(s)`.

    Gaussian KDE, Silverman bandwidth per class, trapezoid integration on a common
    grid. `OVL = 1 - TV(p_S, p_I)` and the integrated mass ship with it so the
    density estimate is inspectable.
    """
    sup, inter = np.asarray(sup, dtype=np.float64), np.asarray(inter, dtype=np.float64)
    if sup.size < 2 or inter.size < 2 or sup.std() <= 0.0 or inter.std() <= 0.0:
        return {"available": False, "reason": "a class is degenerate; KDE is undefined"}
    from scipy import stats

    kde_s = stats.gaussian_kde(sup, bw_method="silverman")
    kde_i = stats.gaussian_kde(inter, bw_method="silverman")
    spread = max(float(kde_s.factor * sup.std()), float(kde_i.factor * inter.std()))
    lo = min(float(sup.min()), float(inter.min())) - 4.0 * spread
    hi = max(float(sup.max()), float(inter.max())) + 4.0 * spread
    axis = np.linspace(lo, hi, int(grid))
    ps, pi = kde_s(axis), kde_i(axis)
    ovl = float(np.trapz(np.minimum(ps, pi), axis))
    return {
        "available": True, "overlapping_coefficient": ovl, "total_variation": float(1.0 - ovl),
        "bayes_error_any_f": float(0.5 * ovl), "best_accuracy_any_f": float(1.0 - 0.5 * ovl),
        "bandwidth_rule": "silverman", "grid_points": int(grid),
        "integrated_mass_check": float(np.trapz(0.5 * (ps + pi), axis)),
    }


def _cluster_arrays(rows: Sequence[PairRow]) -> tuple[tuple[str, ...], list, list]:
    keys = tuple(sorted({row.cluster_id for row in rows}))  # sorted: ids drive the draw
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.cls in (SUPERSESSION, INTERFERENCE):
            grouped.setdefault((row.cluster_id, row.cls), []).append(row.score)
    take = lambda key, cls: np.array(grouped.get((key, cls), ()), dtype=np.float64)
    return keys, [take(k, SUPERSESSION) for k in keys], [take(k, INTERFERENCE) for k in keys]


def _joined(arrays: list, keep: Sequence[int]) -> np.ndarray:
    picked = [arrays[i] for i in keep]
    return np.concatenate(picked) if picked else np.array([], dtype=np.float64)


def ceiling_interval(rows: Sequence[PairRow], *, alpha: float = 0.05,
                     n_resamples: int = DEFAULT_RESAMPLES,
                     seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    """Cluster-level BCa interval on the CEILING itself, refit on every resample.

    The point estimate is seed independent (embedder and sweep are deterministic),
    so seeds vary only the bootstrap draw: their spread is a Monte Carlo stability
    check and seed 0 carries the headline interval.
    """
    from brain.config import get_rng

    keys, sup_by, int_by = _cluster_arrays(rows)
    whole = range(len(keys))
    theta = _ceiling_value(_joined(sup_by, whole), _joined(int_by, whole))
    if len(keys) < 2 or not math.isfinite(theta):
        thin = Interval(theta, math.nan, math.nan, len(keys), "cluster-bootstrap (too few)", alpha)
        return {"point_estimate": theta, "n_clusters": len(keys), "per_seed": [],
                "interval": thin.to_dict()}
    drop = lambda i: [j for j in whole if j != i]
    jackknife = np.array(
        [_ceiling_value(_joined(sup_by, drop(i)), _joined(int_by, drop(i))) for i in whole]
    )
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        rng = get_rng(int(seed), stream="gate-threshold-ceiling")
        draws = rng.integers(0, len(keys), size=(int(n_resamples), len(keys)))
        reps = np.array([_ceiling_value(_joined(sup_by, d), _joined(int_by, d)) for d in draws])
        lower, upper, method = _bca(theta, reps, jackknife, alpha)
        label = f"cluster-bootstrap {method} B={n_resamples}"
        interval = Interval(theta, lower, upper, len(keys), label, alpha).to_dict()
        per_seed.append({**interval, "seed": int(seed)})
    return {
        "point_estimate": float(theta), "n_clusters": len(keys),
        "resampling_unit": "cluster_id (the workstream of the earlier episode)",
        "bootstrap_resamples": int(n_resamples),
        "interval": {k: v for k, v in per_seed[0].items() if k != "seed"}, "per_seed": per_seed,
        "seed_spread_lower": [min(r["lower"] for r in per_seed), max(r["lower"] for r in per_seed)],
        "seed_spread_upper": [min(r["upper"] for r in per_seed), max(r["upper"] for r in per_seed)],
    }


def loco_balanced_accuracy(rows: Sequence[PairRow]) -> dict[str, Any]:
    """Leave-one-cluster-out balanced accuracy: what a real system could reach.

    The ceiling is an ORACLE (tau chosen with ground truth on the same pairs), so it
    is optimistically biased by construction; that is the point, since it bounds
    every shipped policy from above. This fits tau on the other clusters and scores
    the held-out one. Never used for K1.
    """
    keys, sup_by, int_by = _cluster_arrays(rows)
    if len(keys) < 2:
        return {"available": False, "reason": "fewer than two clusters"}
    tp = fn = tn = fp = folds = 0
    for i in range(len(keys)):
        rest = [j for j in range(len(keys)) if j != i]
        train_s, train_i = _joined(sup_by, rest), _joined(int_by, rest)
        if train_s.size == 0 or train_i.size == 0:
            continue
        fit = best_threshold(train_s, train_i)
        tau, high = fit.tau_midpoint, fit.orientation.startswith("s >=")
        fire = (lambda v: v >= tau) if high else (lambda v: v <= tau)
        pred_s, pred_i = fire(sup_by[i]), fire(int_by[i])
        tp, fn = tp + int(pred_s.sum()), fn + int(pred_s.size - pred_s.sum())
        fp, tn = fp + int(pred_i.sum()), tn + int(pred_i.size - pred_i.sum())
        folds += 1
    if not (tp + fn) or not (tn + fp):
        return {"available": False, "reason": "a class never appears in a held-out fold"}
    tpr, tnr = tp / (tp + fn), tn / (tn + fp)
    return {"available": True, "folds": folds, "tpr_supersession": float(tpr),
            "tnr_interference": float(tnr), "balanced_accuracy": float(0.5 * (tpr + tnr))}


def harm_table(rows: Sequence[PairRow], taus: Sequence[float] = HARM_TAUS) -> list[dict[str, Any]]:
    """F17 reproduced as a three-column tradeoff now that SUPERSESSION exists.

    A merge policy fires on `s >= tau`. With supersession present some merges are
    CORRECT: F17's 100 percent harm rate was an artifact of the missing class.
    """
    order = (SUPERSESSION, INTERFERENCE, COINCIDENCE, DUPLICATE)
    scores = {cls: by_class(rows, cls) for cls in order}
    clusters = {row.cluster_id for row in rows if row.cls == INTERFERENCE}
    n_sup, n_int = int(scores[SUPERSESSION].size), int(scores[INTERFERENCE].size)
    rate, table = lambda hit, n: (hit / n) if n else math.nan, []
    for tau in taus:
        got = {cls: int((values >= tau).sum()) for cls, values in scores.items()}
        correct = got[SUPERSESSION] + got[DUPLICATE]  # invalidate or dedupe: legitimate
        wrong = got[INTERFERENCE] + got[COINCIDENCE]  # keep-both classes: destructive
        touched = {r.cluster_id for r in rows if r.cls == INTERFERENCE and r.score >= tau}
        missed = n_sup - got[SUPERSESSION]
        table.append({
            "tau": float(tau), "distinct_events_destroyed": got[INTERFERENCE],
            "distinct_events_destroyed_rate": rate(got[INTERFERENCE], n_int),
            "supersessions_correctly_merged": got[SUPERSESSION],
            "supersessions_correctly_merged_rate": rate(got[SUPERSESSION], n_sup),
            "supersessions_missed": missed, "supersessions_missed_rate": rate(missed, n_sup),
            "coincidence_merged": got[COINCIDENCE], "duplicate_merged": got[DUPLICATE],
            "merge_precision": rate(correct, correct + wrong),
            "clusters_touched": len(touched), "clusters_total": len(clusters),
        })
    return table


def expected_loss(rows: Sequence[PairRow], *, taus: Sequence[float] = HARM_TAUS,
                  ratios: Sequence[float] = COST_RATIOS) -> dict[str, Any]:
    """`L(tau) = c_S * P(s >= tau | I) + c_I * P(s < tau | S)`, over `c_S + c_I`.

    `c_S` is a false supersede, which destroys a distinct event irrecoverably;
    `c_I` is a false keep-both, which leaves a stale fact recency usually buries.
    The Formal Model states `c_S >> c_I` and derives that this pushes the operating
    point away from merging; the symmetric ratio is here so that push is visible
    rather than asserted. Equal class priors, so the curve is a property of the two
    distributions.
    """
    sup, inter = by_class(rows, SUPERSESSION), by_class(rows, INTERFERENCE)
    if not (sup.size and inter.size):
        return {"available": False, "reason": "a class is empty"}
    false_supersede = np.array([float(np.mean(inter >= tau)) for tau in taus])
    false_keep = np.array([float(np.mean(sup < tau)) for tau in taus])
    curves = []
    for ratio in ratios:
        weight = float(ratio) / (float(ratio) + 1.0)
        loss = weight * false_supersede + (1.0 - weight) * false_keep
        best = int(np.argmin(loss))
        curves.append({"cost_ratio_cS_over_cI": float(ratio), "tau_argmin": float(taus[best]),
                       "min_expected_loss": float(loss[best]),
                       "loss_by_tau": {f"{float(t):.2f}": float(v) for t, v in zip(taus, loss)}})
    return {
        "available": True, "priors": "equal (0.5 supersession, 0.5 interference)",
        "taus": [float(tau) for tau in taus], "curves": curves,
        "p_false_supersede": [float(v) for v in false_supersede],
        "p_false_keep_both": [float(v) for v in false_keep]}
