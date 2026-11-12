"""Choosing the candidate, judging it, and checking the harness did not drift.

Three decisions that must not be made ad hoc inside the run loop:

* **Which separation configuration is judged.** The argmax of the sweep, which
  is the reading most generous to the mechanism. Generous on purpose: a kill at
  the best-case setting is a stronger negative result than a kill at an
  arbitrary one. The price is a selection bias in the mechanism's favour, so it
  is written into the payload beside the number rather than left to the reader.
* **Which baselines the verdict sees.** All of them, with `plain_vector`
  supplying the curve for the criterion's second clause. Nothing is dropped for
  losing.
* **Whether the instrument is still the instrument.** s = 0 is an exact no-op in
  M-A, so the separation code path at s = 0 must reproduce the plain-vector
  control to the last bit. If it ever does not, every other number on the page
  is suspect, which is why the check ships with the result instead of living in
  a test that nobody reads at report time.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from brain.eval import evaluate_kill_criterion
from brain.experiments.gate_separation.specs import ArmSpec

__all__ = [
    "select_candidate",
    "verdict_for",
    "sweep_verdicts",
    "flat_curve_check",
    "harness_validation",
]

#: A no-op must be exactly a no-op; this only absorbs float representation error.
_TOL = 1e-12


def _is_control_clone(spec: ArmSpec) -> bool:
    """decorrelate at s = 0 stores the embedder's bytes: it IS the control arm."""
    return (
        spec.config.get("mode") == "decorrelate"
        and float(spec.config.get("s", -1.0)) == 0.0
    )


def select_candidate(
    results: Mapping[str, Any],
    specs: Mapping[str, ArmSpec],
    sweep: Sequence[str],
    control: str,
) -> dict[str, Any]:
    """Best separation arm by disambiguation accuracy, with the bias disclosed."""
    candidates = [name for name in sweep if not _is_control_clone(specs[name])]
    if not candidates:
        return {"selected_arm": None, "reason": "no separation arm produced a result"}
    # `max` returns the FIRST maximal element, so sorting first makes an exact
    # tie resolve by name rather than by whichever arm happened to run first.
    best = max(sorted(candidates), key=lambda name: results[name].disambiguation)
    baseline = results.get(control)
    return {
        "selected_arm": best,
        "selected_config": dict(specs[best].config),
        "selected_accuracy": float(results[best].disambiguation),
        "selection_rule": "argmax disambiguation accuracy over the (mode, s) sweep",
        "selection_bias": (
            "s and mode were selected on the SAME frozen corpus the verdict is read off, "
            "which biases the comparison IN FAVOUR of the mechanism. The full sweep is "
            "reported, and the pre-registered default s is evaluated alongside as the "
            "bias-free reference."
        ),
        "control_accuracy": None if baseline is None else float(baseline.disambiguation),
    }


def verdict_for(
    results: Mapping[str, Any],
    baselines: Sequence[str],
    candidate: str,
    control: str,
    *,
    n_resamples: int,
) -> Any:
    """The pre-registered criterion, computed over every baseline that ran."""
    reference = results.get(control)
    return evaluate_kill_criterion(
        results[candidate].stats(),
        [results[name].stats() for name in sorted(baselines) if name in results],
        baseline_curve=None if reference is None else reference.decay,
        curve_arm=None if reference is None else control,
        n_resamples=int(n_resamples),
    )


def sweep_verdicts(
    results: Mapping[str, Any],
    baselines: Sequence[str],
    sweep: Sequence[str],
    control: str,
    *,
    n_resamples: int,
) -> dict[str, Any]:
    """The criterion applied to EVERY swept configuration, not only the winner.

    "The best setting was killed" invites the reply that some other setting
    might not have been. Evaluating all of them answers it in advance, and the
    count of survivors is the number a reader actually wants.
    """
    rows: list[dict[str, Any]] = []
    for name in sweep:
        verdict = verdict_for(results, baselines, name, control, n_resamples=n_resamples)
        rows.append(
            {
                "arm": name,
                "status": verdict.status,
                "accuracy": float(results[name].disambiguation),
                "comparisons": {
                    comparison.baseline: {
                        "delta_points": round(comparison.delta * 100.0, 4),
                        "meets_margin": comparison.meets_margin,
                        "non_overlapping_cis": comparison.intervals_disjoint,
                        "passes": comparison.passes,
                    }
                    for comparison in verdict.comparisons
                },
            }
        )
    return {
        "rule": "the pre-registered H-A criterion, applied to every swept configuration",
        "n_configurations": len(rows),
        "n_survives": sum(1 for row in rows if row["status"] == "survives"),
        "n_killed": sum(1 for row in rows if row["status"] == "killed"),
        "n_undecided": sum(1 for row in rows if row["status"] == "undecided"),
        "survivors": sorted(row["arm"] for row in rows if row["status"] == "survives"),
        "arms": rows,
    }


def flat_curve_check(results: Mapping[str, Any], sweep: Sequence[str]) -> dict[str, Any]:
    """M-A predicts separation stays FLAT in cluster size while baselines decay.

    The decay test is already computed per arm; this only counts it, because the
    derived expectation is a statement about ALL separation arms and reading it
    off a table row by row is how a reader misses that none of them is flat.
    """
    # A corpus with a single cluster size makes decay unmeasurable, and
    # `curve_decay` reports that as flat. Counting it as a flat curve would turn
    # a missing measurement into evidence for the prediction.
    measurable = sorted(
        name for name in sweep if len(results[name].decay.get("sizes", []) or []) >= 2
    )
    flat = sorted(name for name in measurable if not results[name].decay.get("decays", False))
    return {
        "prediction": "M-A: separation accuracy does not degrade with cluster size m",
        "n_configurations": len(sweep),
        "n_measurable": len(measurable),
        "n_flat": len(flat),
        "flat_arms": flat,
        "drop_by_arm": {
            name: float(results[name].decay.get("drop", float("nan"))) for name in sorted(sweep)
        },
    }


def harness_validation(
    results: Mapping[str, Any],
    specs: Mapping[str, ArmSpec],
    sweep: Sequence[str],
    control: str,
) -> dict[str, Any]:
    """s = 0 in decorrelate mode must reproduce the control bit-for-bit."""
    name = next((arm for arm in sweep if _is_control_clone(specs[arm])), None)
    reference = results.get(control)
    if name is None or reference is None:
        return {"checked": False, "reason": "control or s = 0 arm absent"}
    delta = float(results[name].disambiguation) - float(reference.disambiguation)
    return {
        "checked": True,
        "arm": name,
        "control": control,
        "accuracy": float(results[name].disambiguation),
        "control_accuracy": float(reference.disambiguation),
        "delta": delta,
        "identical": bool(abs(delta) <= _TOL),
        "claim": "s = 0 is an exact no-op, so the separation code path must equal plain vector",
    }
