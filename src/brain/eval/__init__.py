"""Metrics and the multi-seed evaluation harness (Task 007).

The measurement instrument the gate rests on. Three layers, each usable alone:

* `metrics` -- disambiguation accuracy (the metric that tests separation) plus
  precision/recall/MRR, scored per query against the corpus's own sibling labels.
* `stats` / `bootstrap` / `curve` -- intervals. The resampling unit is the
  interference CLUSTER, never the query: three queries share an episode and m
  episodes share a cluster, so query-level resampling reports an interval about
  half the honest width (F11).
* `runner` / `verdict` -- `run_arm` produces per-seed values, aggregates and a
  provenance-carrying result file; `evaluate_kill_criterion` turns the arms'
  numbers into the pre-registered H-A verdict mechanically.

    from brain.eval import run_arm, evaluate_kill_criterion
    plain = run_arm("plain_vector", corpus, seeds=range(5), k=5)
    verdict = evaluate_kill_criterion(sep.stats(), [plain.stats(), full.stats()],
                                      baseline_curve=plain.decay, curve_arm="plain_vector")
"""

from brain.eval.bootstrap import (
    DEFAULT_RESAMPLES,
    ClusterGroups,
    cluster_bootstrap_ci,
    paired_delta_ci,
    stratified_delta_ci,
)
from brain.eval.curve import CurvePoint, accuracy_curve, curve_decay, curve_to_dict
from brain.eval.metrics import (
    METRIC_NAMES,
    QueryOutcome,
    accuracy_by_cluster_size,
    disambiguation_accuracy,
    indicators_by_cluster,
    indicators_by_cluster_size,
    metric_summary,
    mrr,
    precision_at_k,
    recall_at_k,
    score_queries,
    score_query,
)
from brain.eval.results import (
    RESULT_SCHEMA,
    ArmResult,
    SeedRun,
    corpus_provenance,
    write_result,
)
from brain.eval.runner import DEFAULT_K, DEFAULT_SEEDS, ArmBlocked, run_arm
from brain.eval.stats import Interval, cohens_d, cohens_h, mean_ci, non_overlapping
from brain.eval.verdict import (
    KILL_MARGIN,
    REQUIRED_BASELINES,
    ArmStats,
    Comparison,
    Verdict,
    evaluate_kill_criterion,
)

__all__ = [
    "METRIC_NAMES",
    "QueryOutcome",
    "score_query",
    "score_queries",
    "disambiguation_accuracy",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "accuracy_by_cluster_size",
    "indicators_by_cluster",
    "indicators_by_cluster_size",
    "metric_summary",
    "Interval",
    "mean_ci",
    "cohens_d",
    "cohens_h",
    "non_overlapping",
    "ClusterGroups",
    "DEFAULT_RESAMPLES",
    "cluster_bootstrap_ci",
    "paired_delta_ci",
    "stratified_delta_ci",
    "CurvePoint",
    "accuracy_curve",
    "curve_decay",
    "curve_to_dict",
    "ArmStats",
    "Comparison",
    "Verdict",
    "KILL_MARGIN",
    "REQUIRED_BASELINES",
    "evaluate_kill_criterion",
    "ArmResult",
    "ArmBlocked",
    "SeedRun",
    "DEFAULT_SEEDS",
    "DEFAULT_K",
    "run_arm",
    "RESULT_SCHEMA",
    "corpus_provenance",
    "write_result",
]
