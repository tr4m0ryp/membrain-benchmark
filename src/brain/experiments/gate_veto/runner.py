"""Run the veto experiment: plain vector, llm-pick, verify-and-veto, both axes.

Fixed order so it cannot drift: control, arm A, arm B, over the same seeds, k and
query set, each on a fresh arm and therefore a fresh store. Every candidate list
is recorded per query so a caller can REFUSE to report when the decision arms
were not handed byte-identical candidates: a gain attributable to a different
retrieval front end is not a gain attributable to the veto.

**Both axes, always, together.** Disambiguation gain and correct-item rejection
come from one pass over the same records, so a caller cannot obtain one without
the other: the Lure-Discrimination-Index discipline, as a data structure. Every
arm here is deterministic, so the across-seed interval is legitimately
zero-width; the CLUSTER bootstrap carries the verdict, never the seed one."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from brain.config import get_rng, seed_everything
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci, paired_delta_ci
from brain.eval.metrics import metric_summary, score_query
from brain.eval.runner import DEFAULT_K, DEFAULT_SEEDS
from brain.eval.stats import Interval, mean_ci
__all__ = ["AXES", "CONTROL", "ARM_A", "ARM_B", "QueryRecord", "ArmRun", "VetoRun", "arm_groups",
           "run_veto_arm", "run_veto_gate", "safe_paired_delta", "cost_estimate"]

CONTROL, ARM_A, ARM_B = "plain_vector", "llm_pick", "verify_veto"
#: `disambiguation` is the gain side; the rest are harm, and never travel alone.
AXES = ("disambiguation", "rejection_rate", "demotion_rate", "harm_rate")
#: Properties of ONE pass over the query set, not counts of transport events.
#: Every seed replays the same deterministic decisions, so these are maxed over
#: seeds; summing reports 5x and reads as a rate error. Transport counters
#: (api_calls, cache_hits, cost_usd, live_*) accumulate and stay summed.
_PER_PASS = frozenset({"unique_requests", "n_indexed", "queries", "vetoes", "padded_slots",
                       "candidates_judged", "all_vetoed_queries", "parse_failures",
                       "ungrounded_vetoes_downgraded"})


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """What one arm did on one query, on both axes at once."""
    query_id: str
    cluster_id: str
    retrieved: bool  # the correct trace is in the shared candidate set
    disambiguated: bool | None
    rejected: bool | None  # the arm's OWN reject act fired on the correct trace
    correct_rank: int | None


@dataclass(slots=True)
class ArmRun:
    """One arm over N seeds: per-seed metrics, per-seed records, provenance."""
    name: str
    seeds: tuple[int, ...]
    wall_clock: float = 0.0
    metrics: list[dict[str, float]] = field(default_factory=list)
    records: list[list[QueryRecord]] = field(default_factory=list)
    candidates: dict[str, list[str]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    describe: dict[str, Any] = field(default_factory=dict)

    def cost_usd(self) -> float:
        """What this run actually spent. Zero once the response cache is warm."""
        return float(self.stats.get("cost_usd", 0.0) or 0.0)

    def cold_usd(self) -> float:
        """What the same calls cost when first made. A warm re-run spends
        nothing, which says the answers were replayed, not that the arm is free
        to produce, so the cold figure stays reportable."""
        return self.cost_usd() + float(self.stats.get("replayed_cost_usd", 0.0) or 0.0)


def _reject_indicator(arm_name: str, arm: Any, query: Any, outcome: Any) -> bool | None:
    """Did THIS arm's own reject act fire on the correct trace?

    Each arm rejects differently and the contract asks for each arm's own rate.
    `verify_veto` reads the BOOLEAN off the mechanism; `llm_pick` rejects by
    choosing otherwise, so it is a sibling placed ahead of the correct trace; the
    control rejects nothing and is scored on arm A's formula so a reader gets a
    reference point, not a structural zero. None when the correct trace was never
    a candidate: an arm cannot discard what it was not shown."""
    if not outcome.retrieved:
        return None
    if arm_name == ARM_B:
        seen = (getattr(arm, "decisions", {}) or {}).get(query.text, {})
        decision = seen.get(str(query.correct_trace_id))
        return None if decision is None else bool(decision.veto)
    return bool(outcome.eligible and not outcome.disambiguated)


def run_veto_arm(
    name: str, factory: Callable[[], Any], corpus: Any, *,
    seeds: Sequence[int] = DEFAULT_SEEDS, k: int = DEFAULT_K, prefetch: bool = True,
    prefetch_workers: int = 8, progress: Callable[[str], None] | None = None,
) -> ArmRun:
    """Run one arm over `seeds`, recording both axes per query."""
    episodes = sorted(corpus.episodes, key=lambda e: str(e.trace_id))  # ids drive the order
    queries = sorted(corpus.queries, key=lambda q: str(q.query_id))
    run, started = ArmRun(name, tuple(sorted({int(s) for s in seeds}))), time.perf_counter()
    for index, seed in enumerate(run.seeds):
        seed_everything(seed)
        arm = factory()
        try:
            reason = getattr(arm, "blocked_reason", None)
            if reason:
                raise RuntimeError(f"{name}: BLOCKED ({reason})")
            arm.index(episodes)
            # ONCE, before the first query loop; later seeds replay from disk.
            if prefetch and index == 0 and hasattr(arm, "prefetch"):
                arm.prefetch([q.text for q in queries], int(k), workers=int(prefetch_workers))
            records, outcomes = [], []
            for query in queries:
                outcomes.append(outcome := score_query(query, arm.retrieve(query.text, int(k))))
                records.append(QueryRecord(
                    query_id=outcome.query_id, cluster_id=outcome.cluster_id,
                    retrieved=bool(outcome.retrieved), disambiguated=outcome.disambiguated,
                    rejected=_reject_indicator(name, arm, query, outcome),
                    correct_rank=outcome.correct_rank))
            run.metrics.append(metric_summary(outcomes, int(k)))
            run.records.append(records)
            if index == 0:
                # The control is not a CandidateArm, so this is empty for it.
                seen = getattr(arm, "candidates", None) or {}
                run.candidates = {q: list(v) for q, v in seen.items()}
            for key, value in (getattr(arm, "stats", {}) or {}).items():  # never overwrite
                numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
                if key in _PER_PASS:  # identical every seed; max, never sum
                    run.stats[key] = max(run.stats.get(key, 0), value)
                else:
                    run.stats[key] = run.stats.get(key, 0) + value if numeric else value
            run.describe = arm.describe() if hasattr(arm, "describe") else {}
        finally:
            if hasattr(arm, "close"):
                arm.close()
        if progress:
            progress(f"{name} seed {seed}: disambiguation={run.metrics[-1]['disambiguation']:.4f}")
    run.wall_clock = time.perf_counter() - started
    return run


# -- axes ---------------------------------------------------------------------
def _axis(
    run: ArmRun, field_name: str, keep: Callable[[QueryRecord], bool], *, invert: bool = False
) -> dict[str, list[float]]:
    """Per-cluster indicators for one axis, averaging per query over seeds first.

    Averaging per query first keeps the CLUSTER as the resampling unit (coverage
    0.94 cluster versus 0.58 query-level) while folding seed variability in, as
    `brain.eval.runner._pool_clusters` does. Sorted ids drive the order, and a
    None indicator is dropped rather than counted as a zero.
    """
    per_query: dict[str, list[float]] = {}
    cluster_of: dict[str, str] = {}
    for seed_records in run.records:
        for record in seed_records:
            value = getattr(record, field_name)
            if value is None or not keep(record):
                continue
            per_query.setdefault(record.query_id, []).append(float(bool(value) != invert))
            cluster_of[record.query_id] = record.cluster_id
    groups: dict[str, list[float]] = {}
    for query_id in sorted(per_query):
        values = per_query[query_id]
        groups.setdefault(cluster_of[query_id], []).append(sum(values) / len(values))
    return {cid: groups[cid] for cid in sorted(groups)}


def _harm(run: ArmRun, control: ArmRun) -> dict[str, list[float]]:
    """Regression against the control: it was right, this arm broke it. Computed
    identically for every arm, so it is the apples-to-apples harm number."""
    right = {r.query_id for s in control.records for r in s if r.disambiguated}
    return _axis(run, "disambiguated", lambda r: r.query_id in right, invert=True)


def arm_groups(run: ArmRun, control: ArmRun) -> dict[str, dict[str, list[float]]]:
    """Per-cluster indicators for every reported axis, in one structure.
    Rejection and demotion are denominated on queries whose correct trace was a
    candidate: an arm cannot discard what it was never handed."""
    return {
        "disambiguation": _axis(run, "disambiguated", lambda r: True),
        "rejection_rate": _axis(run, "rejected", lambda r: r.retrieved),
        "demotion_rate": _axis(run, "disambiguated", lambda r: r.retrieved, invert=True),
        "harm_rate": _harm(run, control),
    }


def safe_paired_delta(
    a: dict[str, list[float]], b: dict[str, list[float]], **kwargs: Any
) -> Interval:
    """`paired_delta_ci`, but an empty axis returns NaN rather than raising: a
    metric with no observations must be undefined, never 0."""
    if not a or not b:
        return Interval(math.nan, math.nan, math.nan, 0, "paired-cluster-bootstrap (empty)",
                        float(kwargs.get("alpha", 0.05)))
    return paired_delta_ci(a, b, **kwargs)


@dataclass(slots=True)
class VetoRun:
    """Everything one execution of the veto experiment produced."""
    corpus: Any
    k: int
    seeds: tuple[int, ...]
    wall_clock: float = 0.0
    arms: dict[str, ArmRun] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    groups: dict[str, dict[str, dict[str, list[float]]]] = field(default_factory=dict)
    intervals: dict[str, dict[str, Interval]] = field(default_factory=dict)
    seed_intervals: dict[str, dict[str, Interval]] = field(default_factory=dict)
    deltas: dict[str, dict[str, Interval]] = field(default_factory=dict)
    pairing: dict[str, Any] = field(default_factory=dict)

    def cost_usd(self) -> float:
        return float(sum(arm.cost_usd() for arm in self.arms.values()))


def _pairing_check(arms: dict[str, ArmRun]) -> dict[str, Any]:  # paired, or not comparable
    a, b = arms.get(ARM_A), arms.get(ARM_B)
    if a is None or b is None or not a.candidates or not b.candidates:
        return {"checked": False, "identical": None, "reason": "an arm did not run"}
    shared = sorted(set(a.candidates) & set(b.candidates))
    differing = [q for q in shared if a.candidates[q] != b.candidates[q]]
    return {"checked": True, "n_queries_compared": len(shared),
            "n_differing": len(differing),
            "identical": not differing and set(a.candidates) == set(b.candidates)}


def run_veto_gate(
    corpus: Any, *, factories: dict[str, Callable[[], Any]],
    seeds: Sequence[int] = DEFAULT_SEEDS, k: int = DEFAULT_K,
    n_resamples: int = DEFAULT_RESAMPLES, alpha: float = 0.05, bootstrap_seed: int = 0,
    prefetch: bool = True, prefetch_workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> VetoRun:
    """Run every arm, compute both axes, and pair the deltas against the control."""
    started, run = time.perf_counter(), VetoRun(
        corpus=corpus, k=int(k), seeds=tuple(sorted({int(s) for s in seeds})))
    for name in (CONTROL, ARM_A, ARM_B):
        factory = factories.get(name)
        if factory is None:
            run.blocked[name] = "no factory supplied"
            continue
        try:  # a blocked arm is recorded, never scored as if it had answered
            run.arms[name] = run_veto_arm(
                name, factory, corpus, seeds=run.seeds, k=run.k, progress=progress,
                prefetch=prefetch and name != CONTROL, prefetch_workers=prefetch_workers)
        except RuntimeError as exc:
            run.blocked[name] = str(exc)
    if (control := run.arms.get(CONTROL)) is None:
        raise RuntimeError(f"control arm required, absent: {run.blocked}")
    for name, arm in run.arms.items():
        run.groups[name] = arm_groups(arm, control)
        # One bootstrap stream per AXIS, not per arm: two arms drawn from the same
        # seed see the same resampled corpora, so their intervals are comparable
        # rather than differing by draw luck.
        run.intervals[name] = {
            axis: cluster_bootstrap_ci(
                groups, alpha=alpha, n_resamples=n_resamples,
                rng=get_rng(bootstrap_seed, stream=f"veto.bootstrap.{axis}"))
            for axis, groups in run.groups[name].items()}
        run.seed_intervals[name] = {a: mean_ci([m[a] for m in arm.metrics], alpha)
                                    for a in ("disambiguation", "recall_at_k")}
    for name in (n for n in (ARM_A, ARM_B) if n in run.groups):
        run.deltas[name] = {
            axis: safe_paired_delta(
                run.groups[name][axis], run.groups[CONTROL][axis], alpha=alpha,
                n_resamples=n_resamples,
                rng=get_rng(bootstrap_seed, stream=f"veto.delta.{axis}"))
            for axis in AXES}
    run.pairing, run.wall_clock = _pairing_check(run.arms), time.perf_counter() - started
    return run


def cost_estimate(
    run: VetoRun, *, n_smoke: int, n_full: int, budget: float = 25.0
) -> dict[str, Any]:
    """Extrapolate a smoke pass to the full pass. Scaled on DISTINCT requests,
    not calls made: over N seeds plus a prefetch pass the same request is issued
    N+1 times, so scaling `api_calls` would over-state a cold pass by N+1."""
    scale, per_arm = float(n_full) / float(max(1, int(n_smoke))), {}
    for name, arm in sorted(run.arms.items()):
        if (unique := int(arm.stats.get("unique_requests", 0) or 0)) <= 0:
            continue
        # Cost per REQUEST MADE, priced onto the distinct requests a cold pass issues.
        made = sum(int(arm.stats.get(k, 0) or 0) for k in ("api_calls", "cache_hits"))
        per_call = arm.cold_usd() / max(1, made)
        per_arm[name] = {
            "smoke_unique_requests": unique, "smoke_cost_usd": round(arm.cost_usd(), 4),
            "cold_usd_for_unique_requests": round(per_call * unique, 4),
            "requests_made": made, "cold_usd_per_request": round(per_call, 6),
            "smoke_api_calls": int(arm.stats.get("api_calls", 0) or 0),
            "projected_requests": int(round(unique * scale)),
            "projected_cost_usd": round(per_call * unique * scale, 4)}
    total = round(sum(float(v["projected_cost_usd"]) for v in per_arm.values()), 4)
    return {"n_smoke_queries": int(n_smoke), "n_full_queries": int(n_full), "per_arm": per_arm,
            "scale": round(scale, 4), "budget_usd": float(budget),
            "projected_full_pass_usd": total, "over_budget": bool(total > float(budget))}
