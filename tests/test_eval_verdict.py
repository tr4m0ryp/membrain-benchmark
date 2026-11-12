"""The kill criterion: does the verdict fall out of the numbers, at the boundary too.

H-A was pre-registered as "beats plain-vector AND full-context by >= 5 points with
non-overlapping CIs, and the baseline curve is not flat too". Every clause is
checked here on synthetic intervals, including the cases that decide whether the
implementation could manufacture or destroy a result: exactly 5.000 points,
touching intervals, an arm that loses with disjoint intervals, a missing named
baseline, and a seed-level interval smuggled in where a cluster interval belongs.
"""

from __future__ import annotations

import pytest

from brain.eval import ArmStats, Interval, evaluate_kill_criterion


# -- the kill criterion ----------------------------------------------------


def _arm(name: str, accuracy: float, half_width: float) -> ArmStats:
    return ArmStats(
        name=name,
        accuracy=accuracy,
        ci=Interval(accuracy, accuracy - half_width, accuracy + half_width, 33,
                    "cluster-bootstrap bca B=10000"),
    )


DECAYING = {"flat": False, "decays": True, "drop": 0.4}
FLAT = {"flat": True, "decays": False, "drop": 0.0}


def test_verdict_survives_only_when_every_clause_holds() -> None:
    verdict = evaluate_kill_criterion(
        _arm("separation", 0.72, 0.03),
        [_arm("plain_vector", 0.62, 0.03), _arm("full_context", 0.60, 0.03)],
        baseline_curve=DECAYING, curve_arm="plain_vector",
    )
    assert verdict.status == "survives" and verdict.survives and not verdict.killed
    assert all(comparison.passes for comparison in verdict.comparisons)


def test_verdict_at_the_margin_boundary() -> None:
    """Exactly 5.000 points passes; 4.999 does not. Float error must not decide it."""
    exact = evaluate_kill_criterion(
        _arm("separation", 0.67, 0.005),
        [_arm("plain_vector", 0.62, 0.005), _arm("full_context", 0.62, 0.005)],
        baseline_curve=DECAYING,
    )
    assert exact.comparisons[0].delta == pytest.approx(0.05)
    assert exact.status == "survives"
    just_under = evaluate_kill_criterion(
        _arm("separation", 0.66999, 0.005),
        [_arm("plain_vector", 0.62, 0.005), _arm("full_context", 0.62, 0.005)],
        baseline_curve=DECAYING,
    )
    assert just_under.status == "killed"
    assert any("< 5.00" in reason for reason in just_under.reasons)


def test_verdict_kills_on_overlapping_cis_and_on_a_flat_baseline_curve() -> None:
    overlapping = evaluate_kill_criterion(
        _arm("separation", 0.72, 0.09),
        [_arm("plain_vector", 0.62, 0.09), _arm("full_context", 0.60, 0.03)],
        baseline_curve=DECAYING,
    )
    assert overlapping.status == "killed"
    assert any("CIs overlap" in reason for reason in overlapping.reasons)
    flat = evaluate_kill_criterion(
        _arm("separation", 0.72, 0.03),
        [_arm("plain_vector", 0.62, 0.03), _arm("full_context", 0.60, 0.03)],
        baseline_curve=FLAT, curve_arm="plain_vector",
    )
    assert flat.status == "killed" and any("flat" in reason for reason in flat.reasons)


def test_a_losing_arm_with_disjoint_cis_does_not_pass() -> None:
    """Disjoint intervals do not say who won. Direction is required separately."""
    verdict = evaluate_kill_criterion(
        _arm("separation", 0.40, 0.03),
        [_arm("plain_vector", 0.62, 0.03), _arm("full_context", 0.60, 0.03)],
        baseline_curve=DECAYING,
    )
    comparison = next(c for c in verdict.comparisons if c.baseline == "plain_vector")
    assert comparison.intervals_disjoint and not comparison.meets_margin
    assert verdict.status == "killed"


def test_verdict_is_undecided_without_the_named_baselines_or_the_curve() -> None:
    partial = evaluate_kill_criterion(
        _arm("separation", 0.72, 0.03), [_arm("plain_vector", 0.62, 0.03)],
        baseline_curve=DECAYING,
    )
    assert partial.status == "undecided" and "full_context" in partial.reasons[0]
    no_curve = evaluate_kill_criterion(
        _arm("separation", 0.72, 0.03),
        [_arm("plain_vector", 0.62, 0.03), _arm("full_context", 0.60, 0.03)],
    )
    assert no_curve.status == "undecided"


def test_a_seed_interval_cannot_be_used_for_the_verdict() -> None:
    seed_ci = ArmStats("separation", 0.72, Interval(0.72, 0.72, 0.72, 5, "t"))
    with pytest.raises(ValueError, match="cluster bootstrap"):
        evaluate_kill_criterion(seed_ci, [_arm("plain_vector", 0.62, 0.03)])


def test_a_degenerate_interval_is_warned_about() -> None:
    verdict = evaluate_kill_criterion(
        _arm("separation", 1.0, 0.0),
        [_arm("plain_vector", 0.62, 0.03), _arm("full_context", 0.60, 0.03)],
        baseline_curve=DECAYING,
    )
    assert verdict.status == "survives"
    assert any("zero-width" in warning for warning in verdict.warnings)
