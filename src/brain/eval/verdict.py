"""The H-A kill criterion, computed from the numbers rather than asserted.

Pre-registered (spec, Hypothesis & Kill Criteria):

    H-A dies if write-time separation fails to beat plain-vector AND full-context
    by >= 5 points disambiguation accuracy with non-overlapping CIs, OR if the
    accuracy-vs-cluster-size curve is flat for the baseline too.

Every clause here is a function of measured numbers. Nothing in this module
takes a threshold from the result it is judging, and nothing lets a caller
choose the interval afterwards:

* Both named baselines are REQUIRED. A comparison run without full-context is
  not the pre-registered criterion, so the outcome is `undecided`, never
  `survives`.
* The intervals must be CLUSTER bootstrap intervals. A seed-level t interval on
  a deterministic arm is zero-width, and feeding it to an overlap test would
  manufacture separation out of nothing, so it is refused rather than warned
  about.
* A fired kill criterion is a RESULT (spec, Build Plan). This module returns it
  as one; it does not soften it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from brain.eval.bootstrap import DEFAULT_RESAMPLES, paired_delta_ci
from brain.eval.stats import Interval, cohens_h, non_overlapping

__all__ = [
    "KILL_MARGIN",
    "REQUIRED_BASELINES",
    "ArmStats",
    "Comparison",
    "Verdict",
    "evaluate_kill_criterion",
]

#: 5 points of disambiguation accuracy, pre-registered before any measurement.
KILL_MARGIN = 0.05
#: The two arms H-A names. Both must be present for a decision to be the
#: pre-registered one.
REQUIRED_BASELINES: tuple[str, ...] = ("plain_vector", "full_context")

_CLUSTER_METHODS = ("cluster-bootstrap", "paired-cluster-bootstrap")
#: Float representation slack only (see brain.eval.stats).
_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class ArmStats:
    """One arm's headline number, its interval, and the clusters behind it."""

    name: str
    accuracy: float
    ci: Interval
    clusters: Mapping[str, Sequence[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.name, "accuracy": float(self.accuracy), "ci": self.ci.to_dict()}


@dataclass(frozen=True, slots=True)
class Comparison:
    """Candidate against one baseline, on both halves of the first kill clause."""

    baseline: str
    candidate_accuracy: float
    baseline_accuracy: float
    delta: float
    margin: float
    meets_margin: bool
    intervals_disjoint: bool
    passes: bool
    candidate_ci: Interval
    baseline_ci: Interval
    delta_ci: Interval | None = None
    effect_size_h: float = math.nan
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate_accuracy": float(self.candidate_accuracy),
            "baseline_accuracy": float(self.baseline_accuracy),
            "delta": float(self.delta),
            "delta_points": round(float(self.delta) * 100.0, 4),
            "margin": float(self.margin),
            "meets_margin": bool(self.meets_margin),
            "non_overlapping_cis": bool(self.intervals_disjoint),
            "passes": bool(self.passes),
            "candidate_ci": self.candidate_ci.to_dict(),
            "baseline_ci": self.baseline_ci.to_dict(),
            "paired_delta_ci": None if self.delta_ci is None else self.delta_ci.to_dict(),
            "cohens_h": float(self.effect_size_h),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """The mechanical outcome. `status` is survives | killed | undecided."""

    candidate: str
    status: str
    margin: float
    comparisons: tuple[Comparison, ...]
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    baseline_curve: dict[str, Any] | None = None
    curve_arm: str | None = None

    @property
    def survives(self) -> bool:
        return self.status == "survives"

    @property
    def killed(self) -> bool:
        return self.status == "killed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": (
                "H-A dies if the candidate fails to beat plain-vector AND full-context by "
                ">= 5 points disambiguation accuracy with non-overlapping cluster-bootstrap "
                "CIs, OR if the baseline accuracy-vs-cluster-size curve is flat too"
            ),
            "candidate": self.candidate,
            "status": self.status,
            "survives": self.survives,
            "killed": self.killed,
            "margin": float(self.margin),
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "baseline_curve_arm": self.curve_arm,
            "baseline_curve": self.baseline_curve,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def _check_interval(stats: ArmStats, *, require_cluster_ci: bool) -> None:
    if not require_cluster_ci:
        return
    method = str(stats.ci.method)
    if not method.startswith(_CLUSTER_METHODS):
        raise ValueError(
            f"arm {stats.name!r} carries a {method!r} interval; the kill criterion is defined "
            "on the cluster bootstrap (a seed-level interval is zero-width for a deterministic "
            "arm and would fake a separation)"
        )


def _compare(
    candidate: ArmStats,
    baseline: ArmStats,
    *,
    margin: float,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator | None,
) -> Comparison:
    delta = float(candidate.accuracy) - float(baseline.accuracy)
    meets_margin = bool(math.isfinite(delta) and delta >= float(margin) - _TOL)
    disjoint = non_overlapping(candidate.ci, baseline.ci)
    # Disjoint intervals alone do not say WHICH arm won, so the direction is
    # required explicitly: an arm that loses by 20 points also has disjoint CIs.
    passes = bool(meets_margin and disjoint)
    warnings: list[str] = []
    for stats in (candidate, baseline):
        if not stats.ci.defined:
            warnings.append(f"{stats.name}: interval undefined, overlap test cannot separate")
        elif stats.ci.degenerate:
            warnings.append(
                f"{stats.name}: zero-width interval (metric saturated or no cluster varies); "
                "non-overlap against it is fragile"
            )
    delta_ci: Interval | None = None
    if candidate.clusters is not None and baseline.clusters is not None:
        try:
            delta_ci = paired_delta_ci(
                candidate.clusters,
                baseline.clusters,
                alpha=alpha,
                n_resamples=n_resamples,
                rng=rng,
            )
        except ValueError as exc:  # different cluster sets, e.g. a partial run
            warnings.append(f"paired delta unavailable: {exc}")
    return Comparison(
        baseline=baseline.name,
        candidate_accuracy=float(candidate.accuracy),
        baseline_accuracy=float(baseline.accuracy),
        delta=delta,
        margin=float(margin),
        meets_margin=meets_margin,
        intervals_disjoint=disjoint,
        passes=passes,
        candidate_ci=candidate.ci,
        baseline_ci=baseline.ci,
        delta_ci=delta_ci,
        effect_size_h=(
            cohens_h(candidate.accuracy, baseline.accuracy)
            if all(0.0 <= float(v) <= 1.0 for v in (candidate.accuracy, baseline.accuracy))
            else math.nan
        ),
        warnings=tuple(warnings),
    )


def evaluate_kill_criterion(
    candidate: ArmStats,
    baselines: Sequence[ArmStats],
    *,
    margin: float = KILL_MARGIN,
    required: Sequence[str] = REQUIRED_BASELINES,
    baseline_curve: Mapping[str, Any] | None = None,
    curve_arm: str | None = None,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
    require_cluster_ci: bool = True,
) -> Verdict:
    """Decide H-A from the numbers. See the module docstring for the clauses.

    `baseline_curve` is the output of `brain.eval.curve.curve_decay` for the
    plain-vector arm. Omitting it leaves the second clause unevaluated, which
    makes the verdict `undecided` rather than `survives`.
    """
    _check_interval(candidate, require_cluster_ci=require_cluster_ci)
    by_name = {stats.name: stats for stats in baselines}
    comparisons: list[Comparison] = []
    reasons: list[str] = []
    warnings: list[str] = []
    missing = [name for name in required if name not in by_name]
    for name in sorted(by_name):
        _check_interval(by_name[name], require_cluster_ci=require_cluster_ci)
        comparison = _compare(
            candidate,
            by_name[name],
            margin=margin,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=rng,
        )
        comparisons.append(comparison)
        warnings.extend(comparison.warnings)

    required_comparisons = [c for c in comparisons if c.baseline in set(required)]
    failed = [c for c in required_comparisons if not c.passes]
    for comparison in failed:
        detail = []
        if not comparison.meets_margin:
            detail.append(f"delta {comparison.delta * 100:.2f} points < {margin * 100:.2f}")
        if not comparison.intervals_disjoint:
            detail.append("CIs overlap")
        reasons.append(f"does not beat {comparison.baseline}: {', '.join(detail)}")

    curve_flat = None if baseline_curve is None else bool(baseline_curve.get("flat", False))
    if curve_flat:
        reasons.append(
            f"baseline curve is flat for {curve_arm or 'the baseline'}: interference is not "
            "the binding constraint on this corpus"
        )

    if missing:
        status = "undecided"
        reasons.insert(0, f"missing required baseline(s): {missing}")
    elif baseline_curve is None:
        status = "undecided"
        reasons.insert(0, "baseline accuracy-vs-cluster-size curve not supplied")
    elif failed or curve_flat:
        status = "killed"
    else:
        status = "survives"
    return Verdict(
        candidate=candidate.name,
        status=status,
        margin=float(margin),
        comparisons=tuple(comparisons),
        reasons=tuple(reasons),
        warnings=tuple(dict.fromkeys(warnings)),
        baseline_curve=dict(baseline_curve) if baseline_curve is not None else None,
        curve_arm=curve_arm,
    )
