"""Which arms the gate runs, and how a fresh one is built per seed.

Every arm in the gate is described by one `ArmSpec`, so the harness call is
identical for all of them and no arm can pick up a private advantage in the
plumbing. The two knobs that differ are declared, not implied: `seed_independent`
(the LLM arm has no randomness, so it runs once and replays) and `prefetch` (only
the LLM arm has a cache to warm).

`SeededFactory` exists for one reason. `run_arm` wants a zero-argument factory,
but the expansion matrix of `expand_sparsify` is the only genuine randomness any
arm in this launch has, and pinning it to a constant would report a zero-width
across-seed interval for an arm that actually varies. The factory therefore hands
each successive build the seed the harness is about to run under, and records
what it handed out so the caller can VERIFY the pairing rather than trust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from brain.separation import AGGREGATIONS, DEFAULT_AGGREGATION, DEFAULT_S, DEFAULT_TAU
from brain.experiments.gate_separation.arm import SeparationArm, arm_label

__all__ = [
    "S_GRID",
    "SMOKE_S_GRID",
    "MODES",
    "BASELINE_ARMS",
    "PREREGISTERED_S",
    "ArmSpec",
    "SeededFactory",
    "baseline_specs",
    "separation_spec",
    "sweep_specs",
]

#: Separation strengths. Spans the frontier M-A predicts (no separation, through
#: an interior optimum, to a regime where separation has destroyed completion).
#: Fixed before this run and NOT chosen from the outcome; s = 0 is the exact
#: no-op that must reproduce the plain-vector control.
S_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.8)
SMOKE_S_GRID: tuple[float, ...] = (0.0, 0.05, 0.25)
#: F10 caution 1: lead with expansion + decorrelation, do not assume top-k
#: sparsity is the driver. Both substrates are run so the two can be read apart.
MODES: tuple[str, ...] = ("decorrelate", "expand_sparsify")
#: Task 005's module default, chosen before any gate measurement. Reported as the
#: bias-free reference next to the sweep-selected configuration.
PREREGISTERED_S: float = DEFAULT_S
#: Order fixed so a report is diffable. plain_vector and full_context are the two
#: arms H-A names; bm25 and write_time_mmr are the mandatory extra baselines.
BASELINE_ARMS: tuple[str, ...] = ("plain_vector", "bm25", "write_time_mmr", "full_context")


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One arm the gate runs, plus how the harness must be called for it."""

    name: str
    kind: str  # "baseline" or "separation"
    build: Callable[[int], Any]
    seed_independent: bool = False
    prefetch: bool = False
    config: dict[str, Any] = field(default_factory=dict)


class SeededFactory:
    """Adapts `build(seed)` to `run_arm`'s zero-argument factory contract.

    `run_arm` builds exactly one arm per seed, in ascending seed order
    (`brain.eval.runner._run_seed`). Popping the seed list in that order pairs
    each arm with the seed it is about to be run under; `used` lets the caller
    assert the pairing held, so the coupling is checked rather than assumed.
    """

    def __init__(self, build: Callable[[int], Any], seeds: Sequence[int]) -> None:
        self._build = build
        self._seeds = [int(seed) for seed in seeds]
        if not self._seeds:
            raise ValueError("SeededFactory needs at least one seed")
        self.used: list[int] = []

    def __call__(self) -> Any:
        index = min(len(self.used), len(self._seeds) - 1)
        seed = self._seeds[index]
        self.used.append(seed)
        return self._build(seed)


def _baseline_builder(name: str) -> Callable[[int], Any]:
    def build(seed: int) -> Any:  # the baselines carry no seedable randomness
        from brain.baselines import build_arm

        return build_arm(name)

    return build


def baseline_specs(names: Sequence[str] = BASELINE_ARMS) -> list[ArmSpec]:
    """The comparison arms, in a fixed order."""
    specs: list[ArmSpec] = []
    for name in names:
        llm = name == "full_context"
        specs.append(
            ArmSpec(
                name=name,
                kind="baseline",
                build=_baseline_builder(name),
                # No randomness in the LLM arm: one pass, replayed, not five
                # identical paid passes.
                seed_independent=llm,
                prefetch=llm,
                config={"arm": name},
            )
        )
    return specs


def separation_spec(
    mode: str,
    s: float,
    *,
    aggregation: str = DEFAULT_AGGREGATION,
    tau: float = DEFAULT_TAU,
    calibrate: bool = True,
    name: str | None = None,
) -> ArmSpec:
    """One separation arm at one point of the (mode, s, aggregation) grid."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}, expected {AGGREGATIONS}")
    label = name or arm_label(mode, s, aggregation)

    def build(seed: int) -> Any:
        return SeparationArm(
            mode=mode,
            s=float(s),
            tau=float(tau),
            aggregation=aggregation,
            calibrate=calibrate,
            # Only expand_sparsify reads this; decorrelate has no random part.
            projection_seed=int(seed),
            name=label,
        )

    return ArmSpec(
        name=label,
        kind="separation",
        build=build,
        config={
            "mode": mode,
            "s": float(s),
            "aggregation": aggregation,
            "raw_tau": float(tau),
            "tau_calibrated": bool(calibrate),
        },
    )


def sweep_specs(
    *,
    modes: Sequence[str] = MODES,
    grid: Sequence[float] = S_GRID,
    aggregation: str = DEFAULT_AGGREGATION,
    tau: float = DEFAULT_TAU,
) -> list[ArmSpec]:
    """The full (mode, s) sweep, mode-major then ascending s."""
    return [
        separation_spec(mode, s, aggregation=aggregation, tau=tau)
        for mode in modes
        for s in sorted(float(value) for value in grid)
    ]
