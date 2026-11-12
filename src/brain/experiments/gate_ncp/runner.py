"""The NCP sweep: the pre-registered (gamma, alpha) surface, its arms, its diagnostics.

The corpus-side math lives in `channels`; this module is the orchestration and
the measurement discipline. No LLM call is reachable from here, by construction
rather than by policy.

Three diagnostics travel with every cell and none of them can be dropped:

* **Sibling cosine**, in the geometry retrieval actually runs in (see `channels`).
* **Disambiguation**, because C-A cut sibling cosine from 0.824 to 0.704 and
  moved disambiguation by nothing. A cosine drop on its own is not a win.
* **Sibling-pair reordering**, counted against the gamma = 0 scores. C-A's
  signature was that no pair could reorder at any strength: 0.8056 at every
  setting, identical to four decimals. If NCP shows the same flatness it is the
  same dead mechanism under a new name, and that is the result.

At gamma = 0 every weight is exactly 1, so `v_dist == v_full` and the blended
score is the plain-vector score for EVERY alpha. That identity is a free check on
the whole grid, and `tests/test_ncp.py` asserts it rather than trusting it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from brain.baselines.base import as_traces
from brain.config import get_rng, seed_everything
from brain.embed.pooling import pool_weighted, shared_token_cache
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci, paired_delta_ci
from brain.eval.metrics import metric_summary, score_query
from brain.eval.runner import DEFAULT_K, DEFAULT_SEEDS
from brain.eval.stats import Interval, mean_ci
from brain.experiments.gate_ncp.channels import (
    blended_geometry, corpus_state, dist_channel, geometry, query_channel,
)

__all__ = [
    "GAMMA_GRID", "ALPHA_GRID", "K_GRID", "PRIMARY_K", "PRIMARY_QUERY_SIDE",
    "PREREGISTERED_CELL", "SIF_A_GRID", "SIF_PC_GRID", "SUBSAMPLE_N", "preregistration",
    "subsample_ids", "rank_ids", "reordering", "cluster_indicators", "evaluate",
    "cell_specs", "label_for", "GridRun", "run_grid", "paired_delta",
]

# -- PRE-REGISTRATION. Fixed before any number from this task was seen. --------
GAMMA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
K_GRID: tuple[int, ...] = (4, 8, 16)
#: The largest cluster on the frozen corpus is 8, so k = 8 is the smallest
#: neighbourhood that can hold every sibling of every episode.
PRIMARY_K = 8
PRIMARY_QUERY_SIDE = "mean"
#: A neutral centre of the grid, named in advance, so the best cell can be read
#: against a cell that was NOT chosen with hindsight.
PREREGISTERED_CELL = {"weights": "local", "gamma": 1.0, "alpha": 0.5,
                      "k_neighbours": PRIMARY_K, "query_side": PRIMARY_QUERY_SIDE}
SIF_A_GRID: tuple[float, ...] = (1e-3, 1e-4)
SIF_PC_GRID: tuple[bool, ...] = (False, True)
#: The query set llm_pick's response cache was warmed on (F24). Reported as a
#: second column so the paid arm is compared on queries it actually answered.
SUBSAMPLE_N = 180


def preregistration() -> dict[str, Any]:
    """The grid, printed BEFORE the run and stored with the result."""
    return {
        "gamma_grid": list(GAMMA_GRID), "alpha_grid": list(ALPHA_GRID), "k_grid": list(K_GRID),
        "primary_k": PRIMARY_K, "primary_query_side": PRIMARY_QUERY_SIDE,
        "query_side_variants": ["mean", "idf"], "preregistered_cell": dict(PREREGISTERED_CELL),
        "sif_a_grid": list(SIF_A_GRID), "sif_remove_pc_grid": list(SIF_PC_GRID),
        "baseline_arms": ["plain_vector", "bm25", "sif/global_idf", "llm_pick", "full_context"],
        "note": ("the whole surface is reported, not the best cell; the preregistered cell is "
                 "reported beside it so a hindsight pick is visible"),
    }


def subsample_ids(queries: Sequence[Any], n: int) -> list[str]:
    """`n` evenly spaced query ids over the id-sorted list (gate_veto's rule, verbatim)."""
    ordered = sorted(queries, key=lambda q: str(q.query_id))
    want = max(1, min(int(n), len(ordered)))
    picked = sorted({int(i * len(ordered) / want) for i in range(want)})
    return [str(ordered[i].query_id) for i in picked]


def rank_ids(scores: np.ndarray, ids: Sequence[str], k: int) -> list[list[str]]:
    """Top-k per row by descending score, ties by ascending id.

    `ids` must be ascending, which the caller guarantees by sorting episodes on
    `trace_id`. A stable argsort then breaks ties by ascending index, which is
    ascending id: exactly `brain.baselines.base.rank_hits`, vectorised.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="stable")
    width = max(0, min(int(k), len(ids)))
    return [[str(ids[j]) for j in row[:width]] for row in order]


def reordering(scores: np.ndarray, reference: np.ndarray,
               members: Sequence[np.ndarray]) -> dict[str, Any]:
    """Sibling pairs whose ORDER under a query changed against the gamma = 0 scores.

    This is the C-A test. A symmetric transform on finished vectors cannot move
    any of these counts off zero, whatever the knob; NCP has to show that it can.
    """
    total = flipped = 0
    for row, index in enumerate(members):
        if index.size < 2:
            continue
        now = np.sign(scores[row, index][:, None] - scores[row, index][None, :])
        before = np.sign(reference[row, index][:, None] - reference[row, index][None, :])
        upper = np.triu(np.ones_like(now, dtype=bool), k=1)
        total += int(upper.sum())
        flipped += int(((now != before) & upper).sum())
    return {"sibling_pairs_compared": total, "sibling_pairs_reordered": flipped,
            "reorder_rate": (flipped / total) if total else float("nan")}


def cluster_indicators(per_seed: Sequence[Sequence[Any]],
                       keep: set[str] | None = None) -> dict[str, list[float]]:
    """Per-cluster disambiguation indicators, averaged per query over seeds first.

    Mirrors `brain.eval.runner._pool_clusters`: the cluster stays the resampling
    unit (measured coverage 0.94 cluster against 0.58 query-level) while seed
    variability is folded in per query.
    """
    per_query: dict[str, list[float]] = {}
    cluster_of: dict[str, str] = {}
    for outcomes in per_seed:
        for outcome in outcomes:
            if not outcome.eligible or (keep is not None and outcome.query_id not in keep):
                continue
            per_query.setdefault(outcome.query_id, []).append(float(bool(outcome.disambiguated)))
            cluster_of[outcome.query_id] = outcome.cluster_id
    groups: dict[str, list[float]] = {}
    for query_id in sorted(per_query):
        groups.setdefault(cluster_of[query_id], []).append(float(np.mean(per_query[query_id])))
    return {cid: groups[cid] for cid in sorted(groups)}


def _ci(groups: dict[str, list[float]], alpha: float, n_resamples: int, seed: int) -> Interval:
    return cluster_bootstrap_ci(groups, alpha=alpha, n_resamples=n_resamples,
                                rng=get_rng(seed, stream="ncp.bootstrap"))


def evaluate(per_seed: Sequence[Sequence[Any]], *, k: int, subset: set[str], alpha: float,
             n_resamples: int, bootstrap_seed: int) -> dict[str, Any]:
    """Metrics plus cluster-BCa intervals on the full query set and on the subsample."""
    summaries = [metric_summary(o, k) for o in per_seed]
    sub = [metric_summary([x for x in o if x.query_id in subset], k) for o in per_seed]
    groups, sub_groups = cluster_indicators(per_seed), cluster_indicators(per_seed, keep=subset)
    return {
        "metrics": {name: float(np.mean([s[name] for s in summaries])) for name in summaries[0]},
        "subsample_metrics": {name: float(np.mean([s[name] for s in sub])) for name in sub[0]},
        "seed_interval": mean_ci([s["disambiguation"] for s in summaries], alpha),
        "bootstrap": _ci(groups, alpha, n_resamples, bootstrap_seed),
        "subsample_bootstrap": _ci(sub_groups, alpha, n_resamples, bootstrap_seed),
        # Carried out of the cell record by the caller and kept in memory only:
        # two overlapping marginal intervals are a far weaker statement than one
        # paired interval, and the claim NCP has to earn (local df beats global
        # df) is exactly the kind that marginals cannot settle.
        "clusters": groups, "subsample_clusters": sub_groups,
    }


def paired_delta(a: dict[str, list[float]], b: dict[str, list[float]], *, alpha: float = 0.05,
                 n_resamples: int = DEFAULT_RESAMPLES, bootstrap_seed: int = 0) -> Interval:
    """`mean(a) - mean(b)` with both arms scored on the SAME resampled clusters."""
    return paired_delta_ci(a, b, alpha=alpha, n_resamples=n_resamples,
                           rng=get_rng(bootstrap_seed, stream="ncp.delta"))


def cell_specs() -> list[dict[str, Any]]:
    """The pre-registered cell list, expanded. Order is fixed and data-independent."""
    specs: list[dict[str, Any]] = []
    for k in K_GRID:
        for gamma in GAMMA_GRID:
            for side in ("mean", "idf"):
                specs.append({"weights": "local", "gamma": gamma, "k_neighbours": k,
                              "query_side": side, "sif_a": 0.0, "remove_pc": False})
    for gamma in GAMMA_GRID:
        for side in ("mean", "idf"):
            specs.append({"weights": "global", "gamma": gamma, "k_neighbours": PRIMARY_K,
                          "query_side": side, "sif_a": 0.0, "remove_pc": False})
    for a in SIF_A_GRID:
        for pc in SIF_PC_GRID:
            for side in ("mean", "idf"):
                specs.append({"weights": "sif", "gamma": 0.0, "k_neighbours": PRIMARY_K,
                              "query_side": side, "sif_a": a, "remove_pc": pc})
    return specs


def label_for(spec: dict[str, Any], alpha: float) -> str:
    parts = [str(spec["weights"]), f"g{float(spec['gamma']):g}"]
    if spec["weights"] == "local":
        parts.append(f"k{int(spec['k_neighbours'])}")
    if spec["weights"] == "sif":
        parts.append(f"a{float(spec['sif_a']):g}" + ("-pc" if spec["remove_pc"] else ""))
    parts += [f"al{float(alpha):g}", f"q{spec['query_side']}"]
    return "-".join(parts)


@dataclass(slots=True)
class GridRun:
    """Everything one execution of the sweep produced."""

    corpus: Any
    k: int
    seeds: tuple[int, ...]
    subsample: tuple[str, ...]
    preregistration: dict[str, Any] = field(default_factory=preregistration)
    cells: list[dict[str, Any]] = field(default_factory=list)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    #: arm label -> per-cluster indicators. Kept in memory for the paired
    #: deltas and deliberately NOT serialised: it is 320 x 33 lists of noise.
    groups: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    subsample_groups: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    wall_clock: float = 0.0


def run_grid(corpus: Any, *, seeds: Sequence[int] = DEFAULT_SEEDS, k: int = DEFAULT_K,
             n_resamples: int = DEFAULT_RESAMPLES, alpha_level: float = 0.05,
             bootstrap_seed: int = 0,
             progress: Callable[[str], None] | None = None) -> GridRun:
    """Run the whole pre-registered surface. No LLM call is reachable from here."""
    started = time.perf_counter()
    episodes = sorted(corpus.episodes, key=lambda e: str(e.trace_id))
    queries = sorted(corpus.queries, key=lambda q: str(q.query_id))
    ids = [str(e.trace_id) for e in episodes]
    if ids != sorted(ids):
        raise ValueError("episodes must be id-sorted for the vectorised tie-break")
    tokens = shared_token_cache()
    state = corpus_state([e.text for e in episodes], tokens)
    query_docs = tokens.docs([q.text for q in queries])
    q_full = np.asarray(pool_weighted(query_docs, None)[0], dtype=np.float64)
    run = GridRun(corpus=corpus, k=int(k), seeds=tuple(sorted({int(s) for s in seeds})),
                  subsample=tuple(subsample_ids(queries, SUBSAMPLE_N)))
    subset = set(run.subsample)
    position = {trace_id: index for index, trace_id in enumerate(ids)}
    members = [np.array(sorted(position[t] for t in
                               [str(q.correct_trace_id), *map(str, q.distractor_ids)]
                               if t in position), dtype=np.int64) for q in queries]
    cluster_of = [str(e.cluster_id) for e in episodes]
    reference = q_full @ state.full.T  # the gamma = 0 / plain-vector score matrix
    for spec in cell_specs():
        dist, stats = dist_channel(state, spec)
        base = geometry(state.full, dist, cluster_of)
        s_dist = query_channel(state, query_docs, q_full, spec) @ dist.T
        for blend in ALPHA_GRID:
            scores = blend * reference + (1.0 - blend) * s_dist
            clock, per_seed = time.perf_counter(), []
            for seed in run.seeds:
                seed_everything(seed)
                per_seed.append([score_query(q, r) for q, r
                                 in zip(queries, rank_ids(scores, ids, run.k))])
            record = {
                "arm": label_for(spec, blend), **spec, "alpha": float(blend),
                **evaluate(per_seed, k=run.k, subset=subset, alpha=alpha_level,
                           n_resamples=n_resamples, bootstrap_seed=bootstrap_seed),
                **blended_geometry(base, blend), **base,
                **reordering(scores, reference, members), **stats,
                "retrieve_seconds_per_pass": (time.perf_counter() - clock) / len(run.seeds),
            }
            run.groups[record["arm"]] = record.pop("clusters")
            run.subsample_groups[record["arm"]] = record.pop("subsample_clusters")
            run.cells.append(record)
        if progress:
            progress(f"{spec['weights']} gamma={spec['gamma']:g} k={spec['k_neighbours']} "
                     f"q={spec['query_side']}: {len(run.cells)} cells")
    run.diagnostics["encode_seconds"] = round(state.encode_seconds, 3)
    run.diagnostics["n_cells"] = len(run.cells)
    run.wall_clock = time.perf_counter() - started
    return run
