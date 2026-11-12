"""The multi-seed harness: one arm, N seeds, one result file with its provenance.

`run_arm` is the only entry point Task 008 needs. Per seed it reseeds globally,
builds a FRESH arm (and therefore a fresh store, no leakage between runs),
indexes the corpus, answers every query, and scores the outcomes. It then reports
two uncertainties that must not be confused:

* the across-seed t interval per metric (how much the arm's own randomness moves
  the number), which is legitimately zero-width for the deterministic arms; and
* the cluster bootstrap interval on disambiguation accuracy (how much another
  draw of interference clusters would move it), which is the one the kill
  criterion consumes.

Cost control the LLM arm needs: `prefetch` is called ONCE, before the first
seed's query loop, and never again. The full-context arm has no randomness, so
its answers are seed-independent by construction; pass `seed_independent=True`
to run it once and say so in the result rather than paying for five identical
passes. Embeddings are not recomputed either: every vector arm shares the
process-wide `VectorCache`, so the corpus is encoded once no matter how many
seeds or arms run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from brain.config import RESULTS_DIR, SEED, get_rng, seed_everything
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci
from brain.eval.curve import accuracy_curve, curve_decay
from brain.eval.metrics import METRIC_NAMES, metric_summary, score_query
from brain.eval.results import ArmResult, SeedRun, corpus_provenance, write_result
from brain.eval.stats import mean_ci

__all__ = ["DEFAULT_SEEDS", "DEFAULT_K", "ArmBlocked", "SeedRun", "ArmResult", "run_arm"]

#: Five seeds is the floor the spec sets ("a single run is not a result").
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
#: Retrieval depth. Disambiguation is stable in k for every local arm above the
#: largest cluster (measured: identical at k = 5, 10 and 120), and k = 5 is what
#: the full-context cache was warmed with, so a rerun stays free.
DEFAULT_K = 5


class ArmBlocked(RuntimeError):
    """The arm cannot answer (no backend, no cache). Task 008 reports BLOCKED, not 0."""


def _factory(source: Any) -> Callable[[], Any]:
    if isinstance(source, str):
        from brain.baselines import build_arm

        return lambda: build_arm(source)
    if not callable(source):
        raise TypeError(f"retriever_factory must be a callable or an arm name, got {source!r}")
    return source


def _check_corpus(episodes: Sequence[Any], queries: Sequence[Any]) -> None:
    """A query whose answer is not in the store would score 0 for the wrong reason."""
    known = {str(episode.trace_id) for episode in episodes}
    missing = sorted({str(q.correct_trace_id) for q in queries} - known)
    if missing:
        raise ValueError(f"{len(missing)} queries name a correct trace absent from the corpus: {missing[:5]}")


def _pool_clusters(runs: Sequence[SeedRun]) -> tuple[dict[str, list[float]], dict[int, dict[str, list[float]]]]:
    """Per-cluster indicators averaged over seeds, whole-corpus and by cluster size.

    Averaging per query first keeps the cluster as the resampling unit while
    folding in seed variability: a query answered in 3 of 5 seeds contributes 0.6.
    The pooled mean equals the mean of the per-seed accuracies exactly, because
    every seed answers the same query set.
    """
    per_query: dict[str, list[float]] = {}
    cluster_of: dict[str, str] = {}
    size_of: dict[str, int] = {}
    for run in runs:
        for outcome in run.outcomes:
            if not outcome.eligible:
                continue
            per_query.setdefault(outcome.query_id, []).append(float(bool(outcome.disambiguated)))
            cluster_of[outcome.query_id] = outcome.cluster_id
            size_of[outcome.query_id] = int(outcome.cluster_size)
    groups: dict[str, list[float]] = {}
    strata: dict[int, dict[str, list[float]]] = {}
    for query_id in sorted(per_query):  # sorted: ids drive every downstream order
        value = float(np.mean(per_query[query_id]))
        cluster = cluster_of[query_id]
        groups.setdefault(cluster, []).append(value)
        strata.setdefault(size_of[query_id], {}).setdefault(cluster, []).append(value)
    return (
        {cid: groups[cid] for cid in sorted(groups)},
        {size: dict(sorted(strata[size].items())) for size in sorted(strata)},
    )


def _run_seed(
    factory: Callable[[], Any],
    seed: int,
    episodes: Sequence[Any],
    queries: Sequence[Any],
    k: int,
    *,
    prefetch: bool,
    prefetch_workers: int,
) -> SeedRun:
    seed_everything(seed)
    arm = factory()
    try:
        reason = getattr(arm, "blocked_reason", None)
        if reason:
            raise ArmBlocked(f"{getattr(arm, 'name', 'arm')}: {reason}")
        started = time.perf_counter()
        arm.index(episodes)
        index_seconds = time.perf_counter() - started
        warmed = 0
        if prefetch and hasattr(arm, "prefetch"):
            texts = [q.text for q in queries]
            try:
                warmed = int(arm.prefetch(texts, k, workers=prefetch_workers))
            except TypeError:  # an arm whose prefetch takes no worker count
                warmed = int(arm.prefetch(texts, k))
        started = time.perf_counter()
        outcomes = [score_query(query, arm.retrieve(query.text, k)) for query in queries]
        retrieve_seconds = time.perf_counter() - started
        stats = dict(getattr(arm, "stats", {}) or {})
        describe = arm.describe() if hasattr(arm, "describe") else {}
    finally:
        if hasattr(arm, "close"):
            arm.close()
    return SeedRun(
        seed=int(seed),
        metrics=metric_summary(outcomes, k),
        outcomes=outcomes,
        index_seconds=index_seconds,
        retrieve_seconds=retrieve_seconds,
        prefetched=warmed,
        arm_stats=stats,
        describe=describe,
    )


def run_arm(
    retriever_factory: Any,
    corpus: Any,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    k: int = DEFAULT_K,
    name: str | None = None,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int | None = None,
    seed_independent: bool = False,
    prefetch: bool = True,
    prefetch_workers: int = 6,
    write: bool = True,
    results_dir: Path | str = RESULTS_DIR,
    progress: Callable[[str], None] | None = None,
) -> ArmResult:
    """Run one arm over `seeds` and return per-seed values plus aggregates.

    `retriever_factory` is a zero-argument callable returning a fresh arm (the
    shape `ARM_BUILDERS` already has) or an arm name. A fresh arm per seed means
    a fresh store per seed: nothing carries over between runs.
    """
    episodes = sorted(corpus.episodes, key=lambda e: str(e.trace_id))
    queries = sorted(corpus.queries, key=lambda q: str(q.query_id))
    _check_corpus(episodes, queries)
    if int(k) <= 0:
        raise ValueError(f"k must be positive, got {k!r}")
    wanted = tuple(sorted({int(seed) for seed in seeds}))
    if not wanted:
        raise ValueError("at least one seed is required")
    running = wanted[:1] if seed_independent else wanted

    factory = _factory(retriever_factory)
    started = time.perf_counter()
    runs: list[SeedRun] = []
    for index, seed in enumerate(running):
        if progress:
            progress(f"seed {seed} ({index + 1}/{len(running)})")
        runs.append(
            _run_seed(
                factory,
                seed,
                episodes,
                queries,
                int(k),
                # Once, before the first query loop: the cache is on disk and
                # every later seed replays from it for free.
                prefetch=bool(prefetch) and index == 0,
                prefetch_workers=int(prefetch_workers),
            )
        )
    wall_clock = time.perf_counter() - started

    describe = dict(runs[0].describe)
    arm_name = str(name or describe.get("name") or "arm")
    aggregate = {
        metric: mean_ci([run.metrics[metric] for run in runs], alpha) for metric in METRIC_NAMES
    }
    groups, strata = _pool_clusters(runs)
    base = SEED if bootstrap_seed is None else int(bootstrap_seed)
    # One stream per purpose and NOT per arm: two arms bootstrapped from the same
    # seed then see the same sequence of resampled corpora, so their intervals
    # are directly comparable rather than differing by draw luck.
    bootstrap = cluster_bootstrap_ci(
        groups, alpha=alpha, n_resamples=n_resamples, rng=get_rng(base, stream="eval.bootstrap")
    )
    curve = accuracy_curve(
        strata, alpha=alpha, n_resamples=n_resamples, rng=get_rng(base, stream="eval.curve")
    )
    decay = curve_decay(
        strata, alpha=alpha, n_resamples=n_resamples, rng=get_rng(base, stream="eval.decay")
    )
    result = ArmResult(
        arm=arm_name,
        k=int(k),
        seeds=wanted,
        seeds_run=tuple(run.seed for run in runs),
        seed_independent=bool(seed_independent),
        runs=tuple(runs),
        aggregate=aggregate,
        bootstrap=bootstrap,
        clusters=groups,
        curve=curve,
        decay=decay,
        describe=describe,
        corpus=corpus_provenance(corpus),
        config={
            "alpha": float(alpha),
            "bootstrap_resamples": int(n_resamples),
            "bootstrap_seed": int(base),
            "resampling_unit": "interference cluster",
            "seed_interval_method": "student-t over seeds",
            # Stated because it bounds what the cluster interval covers: seed
            # noise is averaged out per query before resampling, so that
            # interval is about corpus sampling, and seed variability is
            # reported separately by the t interval above.
            "seed_pooling": "mean over seeds per query, then resample clusters",
            "prefetch_workers": int(prefetch_workers),
        },
        wall_clock=wall_clock,
    )
    if write:
        path = write_result(result.to_dict(), arm=arm_name, directory=results_dir)
        object.__setattr__(result, "path", path)
        if progress:
            progress(f"wrote {path}")
    return result
