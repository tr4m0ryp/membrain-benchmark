"""One (arm, k) cell: every pre-registered metric, all cluster-BCa.

Reuses `eval.runner.run_arm` for the heavy lifting (indexing, 5-seed retrieval,
scoring, the disambiguation cluster bootstrap, and the accuracy-by-cluster-size
curve with its own per-size cluster bootstrap) and adds the three cluster-BCa
intervals `run_arm` does not compute (recall@k, precision@k, MRR), via
`pool.pool_metric_clusters`. Nothing here re-implements retrieval or scoring;
this module is assembly plus the extra bootstrap calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brain.eval.bootstrap import cluster_bootstrap_ci
from brain.eval.results import ArmResult
from brain.eval.runner import run_arm
from brain.eval.stats import Interval

from brain.experiments.gate_k_sweep.pool import POOLED_METRICS, pool_metric_clusters
from brain.experiments.gate_k_sweep.registration import (
    ALPHA,
    BOOTSTRAP_SEED,
    N_RESAMPLES,
    SEEDS,
)

__all__ = ["KPoint", "evaluate_point"]


@dataclass(frozen=True, slots=True)
class KPoint:
    """Every headline metric for one (arm, k), each with a cluster-BCa interval."""

    arm: str
    k: int
    recall_at_k: Interval
    disambiguation: Interval
    precision_at_k: Interval
    mrr: Interval
    curve: dict[int, Any]  # accuracy-by-cluster-size, cluster-BCa per point
    decay: dict[str, Any]
    result: ArmResult

    def to_dict(self) -> dict[str, Any]:
        from brain.eval.curve import curve_to_dict

        return {
            "arm": self.arm,
            "k": int(self.k),
            "metrics": {
                "recall_at_k": self.recall_at_k.to_dict(),
                "disambiguation": self.disambiguation.to_dict(),
                "precision_at_k": self.precision_at_k.to_dict(),
                "mrr": self.mrr.to_dict(),
            },
            "accuracy_by_cluster_size": curve_to_dict(self.curve),
            "cluster_size_decay": self.decay,
            "n_clusters": len(self.result.clusters),
            "n_queries": sum(len(v) for v in self.result.clusters.values()),
        }


def evaluate_point(
    arm_name: str,
    corpus: Any,
    k: int,
    *,
    seeds: tuple[int, ...] = SEEDS,
    alpha: float = ALPHA,
    n_resamples: int = N_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    progress: Any = None,
) -> KPoint:
    """Run one arm at one k over the pre-registered seed set, cluster-BCa everywhere."""
    from brain.config import get_rng

    result = run_arm(
        arm_name,
        corpus,
        seeds=seeds,
        k=int(k),
        alpha=alpha,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        write=False,
        prefetch=True,
        progress=progress,
    )
    extra: dict[str, Interval] = {}
    for metric in POOLED_METRICS:
        groups = pool_metric_clusters(result.runs, metric, int(k))
        # One stream per metric, shared across every (arm, k): the resampled
        # cluster INDEX draws are then identical wherever the cluster count is
        # identical (always 33 here), matching `run_arm`'s own convention that
        # arms bootstrapped from the same seed see the same resampled corpora.
        extra[metric] = cluster_bootstrap_ci(
            groups,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=get_rng(bootstrap_seed, stream=f"gate_k_sweep.{metric}"),
        )
    return KPoint(
        arm=arm_name,
        k=int(k),
        recall_at_k=extra["recall_at_k"],
        disambiguation=result.bootstrap,
        precision_at_k=extra["precision_at_k"],
        mrr=extra["mrr"],
        curve=result.curve,
        decay=result.decay,
        result=result,
    )
