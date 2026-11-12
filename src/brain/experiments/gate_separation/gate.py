"""The gate: every arm on the same frozen corpus, then the H-A verdict.

Order of operations is fixed here so it cannot drift between runs:

1. Load the corpus THROUGH the loader, which verifies the content hash. A run
   against drifted bytes aborts instead of producing an unattributable number.
2. Run the baselines, then the (mode, s) sweep, all with the same seeds, the same
   k, the same query set and one shared embedding pass.
3. Select the best separation configuration, and record that the selection was
   made on the SAME frozen corpus the verdict is read off. That is an optimistic
   bias in favour of the mechanism and it is disclosed in the payload, not
   buried.
4. Compute the verdict mechanically with `evaluate_kill_criterion`, twice: once
   at the selected configuration and once at the pre-registered default s, which
   carries no selection bias. Nothing here hand-asserts an outcome.

A killed verdict is a valid outcome (spec, Build Plan). Nothing in this module
retunes anything after seeing a number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from brain.config import RESULTS_DIR
from brain.eval import ArmBlocked, run_arm
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.eval.runner import DEFAULT_K, DEFAULT_SEEDS
from brain.experiments.gate_separation.diagnostics import (
    geometry_sweep,
    m2_invariance,
    m2_within_mode,
)
from brain.experiments.gate_separation.selection import (
    flat_curve_check,
    harness_validation,
    select_candidate,
    sweep_verdicts,
    verdict_for,
)
from brain.experiments.gate_separation.specs import (
    MODES,
    PREREGISTERED_S,
    S_GRID,
    ArmSpec,
    SeededFactory,
    baseline_specs,
    separation_spec,
    sweep_specs,
)

__all__ = ["GateRun", "run_gate", "run_spec"]

_CONTROL = "plain_vector"


@dataclass(slots=True)
class GateRun:
    """Everything one gate execution produced."""

    corpus: Any
    k: int
    seeds: tuple[int, ...]
    results: dict[str, Any] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    specs: dict[str, ArmSpec] = field(default_factory=dict)
    sweep: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    verdicts: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    sweep_verdicts: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    wall_clock: float = 0.0

    @property
    def candidate(self) -> str | None:
        return self.selection.get("selected_arm")

    def cost_usd(self) -> float:
        """Reported LLM spend for THIS run. Zero once the response cache is warm."""
        return float(
            sum(
                float(run.arm_stats.get("cost_usd", 0.0) or 0.0)
                for result in self.results.values()
                for run in result.runs
            )
        )

    def llm_accounting(self) -> dict[str, Any]:
        """Live calls versus cache replays, so a zero cost cannot be misread.

        A warmed run costs nothing, which says the answers were replayed, not
        that the arm was free to produce.
        """
        fields = (
            "api_calls",
            "cache_hits",
            "cost_usd",
            "live_prompt_tokens",
            "live_cache_read_tokens",
            "live_cache_creation_tokens",
            "live_output_tokens",
            "replayed_prompt_tokens",
            "degraded_queries",
            "parse_failures",
            "padded_slots",
            "temperature_enforced",
        )
        out: dict[str, Any] = {}
        for name, result in sorted(self.results.items()):
            stats = dict(result.runs[0].arm_stats) if result.runs else {}
            if "api_calls" not in stats:
                continue
            out[name] = {
                field: stats.get(field) for field in fields if field in stats
            }
            out[name]["backend"] = result.describe.get("params", {}).get("backend")
            out[name]["model"] = result.describe.get("params", {}).get("model")
        return out


def run_spec(
    spec: ArmSpec,
    corpus: Any,
    *,
    seeds: Sequence[int],
    k: int,
    n_resamples: int,
    write: bool = False,
    results_dir: Path | str = RESULTS_DIR,
    progress: Callable[[str], None] | None = None,
) -> tuple[Any | None, str | None]:
    """Run one arm. Returns (result, blocked_reason); exactly one is None."""
    factory = SeededFactory(spec.build, seeds)
    try:
        result = run_arm(
            factory,
            corpus,
            seeds=seeds,
            k=k,
            name=spec.name,
            n_resamples=n_resamples,
            seed_independent=spec.seed_independent,
            prefetch=spec.prefetch,
            write=write,
            results_dir=results_dir,
            progress=None,
        )
    except ArmBlocked as exc:
        if progress:
            progress(f"{spec.name}: BLOCKED ({exc})")
        return None, str(exc)
    # The seed the arm was BUILT with must be the seed it was RUN under,
    # otherwise the sparse arm's projection would not track the reported seeds.
    if factory.used != list(result.seeds_run):
        raise RuntimeError(
            f"{spec.name}: built with seeds {factory.used} but ran seeds "
            f"{list(result.seeds_run)}; the per-seed pairing broke"
        )
    if progress:
        ci = result.bootstrap
        progress(
            f"{spec.name:<40} disambig={ci.mean:.4f} "
            f"cluster95=[{ci.lower:.4f}, {ci.upper:.4f}] {result.wall_clock:.1f}s"
        )
    return result, None


def _run_all(
    specs: Sequence[ArmSpec], corpus: Any, run: GateRun, **kwargs: Any
) -> list[str]:
    names: list[str] = []
    for spec in specs:
        result, reason = run_spec(spec, corpus, **kwargs)
        run.specs[spec.name] = spec
        if result is None:
            run.blocked[spec.name] = str(reason)
            continue
        run.results[spec.name] = result
        names.append(spec.name)
    return names


def _verdict_for(run: GateRun, candidate: str, n_resamples: int) -> Any:
    return verdict_for(
        run.results, run.baselines, candidate, _CONTROL, n_resamples=n_resamples
    )


def run_gate(
    corpus: Any,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    k: int = DEFAULT_K,
    n_resamples: int = DEFAULT_RESAMPLES,
    baselines: Sequence[str] | None = None,
    modes: Sequence[str] = MODES,
    grid: Sequence[float] | None = None,
    aggregation: str = "sequential",
    secondary_aggregation: str | None = "net",
    write_arms: bool = False,
    results_dir: Path | str = RESULTS_DIR,
    progress: Callable[[str], None] | None = None,
) -> GateRun:
    """Run every arm, sweep s, and evaluate the pre-registered H-A criterion."""
    started = time.perf_counter()
    seed_tuple = tuple(sorted({int(seed) for seed in seeds}))
    run = GateRun(corpus=corpus, k=int(k), seeds=seed_tuple)
    common = {
        "seeds": seed_tuple,
        "k": int(k),
        "n_resamples": int(n_resamples),
        "write": bool(write_arms),
        "results_dir": results_dir,
        "progress": progress,
    }
    specs = baseline_specs() if baselines is None else baseline_specs(baselines)
    run.baselines = _run_all(specs, corpus, run, **common)
    sweep = sweep_specs(
        modes=modes,
        grid=S_GRID if grid is None else grid,
        aggregation=aggregation,
    )
    run.sweep = _run_all(sweep, corpus, run, **common)
    run.selection = select_candidate(run.results, run.specs, run.sweep, _CONTROL)

    # Secondary aggregation, reported as a diagnostic only: "sequential" is the
    # literal reading of M-A and the recorded spec decision, so it stays primary
    # whatever "net" scores.
    secondary: list[str] = []
    if secondary_aggregation and run.candidate:
        chosen = run.specs[run.candidate].config
        secondary = _run_all(
            [
                separation_spec(mode, float(chosen["s"]), aggregation=secondary_aggregation)
                for mode in modes
            ],
            corpus,
            run,
            **common,
        )

    if run.candidate:
        run.verdicts["selected"] = _verdict_for(run, run.candidate, n_resamples)
    preregistered = f"separation-decorrelate-s{PREREGISTERED_S:g}-{aggregation}"
    if preregistered in run.results:
        run.verdicts["preregistered"] = _verdict_for(run, preregistered, n_resamples)
    run.diagnostics = {
        "primary_aggregation": aggregation,
        "secondary_aggregation_arms": secondary,
        "preregistered_arm": preregistered if preregistered in run.results else None,
        "m2_invariance": (
            m2_invariance({name: run.results[name] for name in run.sweep}, run.results[_CONTROL])
            if _CONTROL in run.results and run.sweep
            else None
        ),
        "m2_within_mode": (
            m2_within_mode({name: run.results[name] for name in run.sweep}, run.specs)
            if run.sweep
            else None
        ),
        "stored_geometry": geometry_sweep([run.specs[name] for name in run.sweep], corpus),
        "flat_curve_check": flat_curve_check(run.results, run.sweep),
        "harness_validation": harness_validation(
            run.results, run.specs, run.sweep, _CONTROL
        ),
    }
    # Every configuration judged, not only the winner, so "some other s might
    # have survived" is answered by the run rather than left open.
    run.sweep_verdicts = sweep_verdicts(
        run.results, run.baselines, run.sweep, _CONTROL, n_resamples=n_resamples
    )
    run.wall_clock = time.perf_counter() - started
    return run
