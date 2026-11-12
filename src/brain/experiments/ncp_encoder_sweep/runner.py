"""Run the whole cell grid for ONE encoder, at every swept k_neighbours.

Reuses `brain.experiments.gate_ncp.channels` (the pure per-cell math) and the
generic parts of `brain.experiments.gate_ncp.runner` (ranking, reordering,
cluster pooling, the bootstrap wrappers) unmodified. The only new plumbing is:
a PRIVATE `TokenCache` bound to a chosen `sentence-transformers` model instead
of the process-wide MiniLM one, and a spec list that varies k_neighbours
instead of holding it fixed.

**The free correctness test.** At gamma = 0 every local/global weight is
exactly 1.0 (`x ** 0 == 1` for every finite x), so `dist_channel` must return
`state.full` byte-for-byte and every alpha's score matrix must equal the plain
mean-pooled score matrix, for EVERY encoder. `assert_gamma_zero_identity`
checks this per cell and the caller raises if it ever fails, rather than
trusting that an architecture-agnostic pooling formula stays true across five
different token embedding spaces.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from brain.config import get_rng, seed_everything
from brain.embed.embedder import Embedder
from brain.embed.pooling import TokenCache, pool_weighted
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.eval.metrics import score_query
from brain.experiments.gate_ncp.channels import blended_geometry, corpus_state, dist_channel, geometry, query_channel
from brain.experiments.gate_ncp.runner import (
    ALPHA_GRID,
    cluster_indicators,
    label_for,
    paired_delta,
    rank_ids,
    reordering,
)
from brain.eval.stats import Interval, mean_ci
from brain.experiments.ncp_encoder_sweep.specs import all_cell_specs

__all__ = ["RETRIEVAL_K", "EncoderRun", "run_encoder", "paired_delta"]

#: Retrieval depth for ranking (not the swept neighbourhood size). Fixed at
#: F26's value; this task does not reopen it.
RETRIEVAL_K = 5


def _evaluate_local(per_seed: Sequence[Sequence[Any]], *, k: int, alpha: float,
                    n_resamples: int, bootstrap_seed: int) -> dict[str, Any]:
    """Metrics plus cluster-BCa, without the subsample machinery `gate_ncp` needs.

    This task never compares against `llm_pick`'s cache-warmed subsample, so
    there is no subsample to score; a minimal re-implementation avoids passing
    a meaningless empty subset through `gate_ncp.runner.evaluate`.
    """
    from brain.eval.metrics import metric_summary

    summaries = [metric_summary(o, k) for o in per_seed]
    groups = cluster_indicators(per_seed)
    return {
        "metrics": {name: float(np.mean([s[name] for s in summaries])) for name in summaries[0]},
        "seed_interval": mean_ci([s["disambiguation"] for s in summaries], alpha),
        "bootstrap": cluster_bootstrap_ci_(groups, alpha, n_resamples, bootstrap_seed),
        "clusters": groups,
    }


def cluster_bootstrap_ci_(groups: dict[str, list[float]], alpha: float, n_resamples: int,
                          bootstrap_seed: int) -> Interval:
    from brain.eval.bootstrap import cluster_bootstrap_ci

    return cluster_bootstrap_ci(groups, alpha=alpha, n_resamples=n_resamples,
                                rng=get_rng(bootstrap_seed, stream="ncp_encoder_sweep.bootstrap"))


@dataclass(slots=True)
class EncoderRun:
    """Everything one encoder's grid execution produced."""

    encoder: str
    dim: int
    k_grid: tuple[int, ...]
    seeds: tuple[int, ...]
    cells: list[dict[str, Any]] = field(default_factory=list)
    groups: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    wall_clock: float = 0.0


def run_encoder(model_name: str, corpus: Any, *, seeds: Sequence[int], k_grid: Sequence[int],
                n_resamples: int = DEFAULT_RESAMPLES, alpha_level: float = 0.05,
                bootstrap_seed: int = 0,
                progress: Callable[[str], None] | None = None) -> EncoderRun:
    """Run the pre-registered surface for one encoder. No LLM call is reachable from here."""
    started = time.perf_counter()
    embedder = Embedder(model_name=model_name, device="cpu")
    tokens = TokenCache(embedder=embedder)  # PRIVATE cache: never the shared MiniLM one.

    episodes = sorted(corpus.episodes, key=lambda e: str(e.trace_id))
    queries = sorted(corpus.queries, key=lambda q: str(q.query_id))
    ids = [str(e.trace_id) for e in episodes]
    if ids != sorted(ids):
        raise ValueError("episodes must be id-sorted for the vectorised tie-break")

    encode_started = time.perf_counter()
    state = corpus_state([e.text for e in episodes], tokens)
    query_docs = tokens.docs([q.text for q in queries])
    encode_seconds = time.perf_counter() - encode_started
    q_full = np.asarray(pool_weighted(query_docs, None)[0], dtype=np.float64)
    reference = q_full @ state.full.T  # the gamma = 0 / plain-vector score matrix, THIS encoder

    position = {trace_id: index for index, trace_id in enumerate(ids)}
    members = [np.array(sorted(position[t] for t in
                               [str(q.correct_trace_id), *map(str, q.distractor_ids)]
                               if t in position), dtype=np.int64) for q in queries]
    cluster_of = [str(e.cluster_id) for e in episodes]

    run = EncoderRun(encoder=model_name, dim=int(embedder.dim), k_grid=tuple(int(k) for k in k_grid),
                     seeds=tuple(sorted({int(s) for s in seeds})))
    gamma_zero_failures: list[str] = []
    sibling_floor = math.inf
    sibling_floor_cell = ""

    specs = all_cell_specs(k_grid, k_label=int(k_grid[0]) if k_grid else 0)
    for spec in specs:
        dist, stats = dist_channel(state, spec)
        base = geometry(state.full, dist, cluster_of)
        # Restricted to the LOCAL family: F26's 0.5153 floor was measured on the
        # local primary surface (gamma sweep at k=8), and `sibling_cosine_dist`
        # does not depend on alpha, so the label omits it.
        if spec["weights"] == "local" and float(base["sibling_cosine_dist"]) < sibling_floor:
            sibling_floor = float(base["sibling_cosine_dist"])
            sibling_floor_cell = (f"local-g{float(spec['gamma']):g}-k{int(spec['k_neighbours'])}"
                                  f"-q{spec['query_side']}")
        is_gamma_zero_identity_cell = float(spec["gamma"]) == 0.0 and spec["weights"] in ("local", "global")
        if is_gamma_zero_identity_cell and not np.array_equal(dist, state.full):
            gamma_zero_failures.append(f"{label_for(spec, 0.0)} (dist_channel != mean pooling)")
        s_dist = query_channel(state, query_docs, q_full, spec) @ dist.T
        for blend in ALPHA_GRID:
            scores = blend * reference + (1.0 - blend) * s_dist
            if is_gamma_zero_identity_cell and not np.array_equal(scores, reference):
                gamma_zero_failures.append(f"{label_for(spec, blend)} (scores != plain-vector scores)")
            per_seed = []
            for seed in run.seeds:
                seed_everything(seed)
                per_seed.append([score_query(q, r) for q, r
                                 in zip(queries, rank_ids(scores, ids, RETRIEVAL_K))])
            record = {
                "encoder": model_name, "arm": label_for(spec, blend), **spec, "alpha": float(blend),
                **_evaluate_local(per_seed, k=RETRIEVAL_K, alpha=alpha_level,
                                  n_resamples=n_resamples, bootstrap_seed=bootstrap_seed),
                **blended_geometry(base, blend), **base,
                **reordering(scores, reference, members), **stats,
            }
            run.groups[record["arm"]] = record.pop("clusters")
            run.cells.append(record)
        if progress:
            progress(f"{model_name} {spec['weights']} gamma={spec['gamma']:g} "
                     f"k={spec['k_neighbours']} q={spec['query_side']}: {len(run.cells)} cells")

    run.diagnostics["encode_seconds"] = round(encode_seconds, 3)
    run.diagnostics["n_cells"] = len(run.cells)
    run.diagnostics["gamma_zero_identity_ok"] = not gamma_zero_failures
    run.diagnostics["gamma_zero_identity_failures"] = gamma_zero_failures
    run.diagnostics["sibling_cosine_floor"] = sibling_floor if math.isfinite(sibling_floor) else None
    run.diagnostics["sibling_cosine_floor_cell"] = sibling_floor_cell
    run.wall_clock = time.perf_counter() - started
    return run
