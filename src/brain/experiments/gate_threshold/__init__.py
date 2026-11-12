"""Task 006: the threshold-ceiling gate (K1), the minimal falsifying experiment.

This is the gate the whole post-write correctness cluster is conditional on.
It measures the similarity distributions of the four relational classes, computes
the best achievable single-threshold rule separating SUPERSESSION (invalidate)
from INTERFERENCE (keep both), and reports it as the ceiling every shipped
merge-above-threshold policy sits under.

    K1 fires if that ceiling exceeds 0.90 balanced accuracy. Then a threshold is
    sufficient, I1 is unnecessary, and the plan collapses to I2 and I3.

Two ceilings are reported and K1 is evaluated against both. The
SAME-TOPIC-STRATIFIED one is PRIMARY: task 001 measured interference pairs at
100 percent same-workstream and supersession pairs at 53 percent, and both
effects push toward overlap, which is the direction that favours our own claim.
Stratifying removes that help, so the stratified ceiling is the adversarial
number and the marginal one is a companion rather than the headline.

Long and short regimes are never collapsed into the headline: task 001 predicted
the short regime would separate cleanly and called the long regime open, so a
pooled number would hide the finding rather than report it.

    from brain.experiments.gate_threshold import run_experiment
    from brain.corpus.relations import load_relations
    result = run_experiment(load_relations())

Run it:

    .venv/bin/python -m brain.experiments.gate_threshold.report --seeds 5
"""

from brain.experiments.gate_threshold.ceiling import (
    COST_RATIOS,
    HARM_TAUS,
    K1_TEXT,
    K1_THRESHOLD,
    Ceiling,
    best_threshold,
    ceiling_interval,
    expected_loss,
    harm_table,
    loco_balanced_accuracy,
    overlap_bayes_error,
)
from brain.experiments.gate_threshold.markdown import render_markdown
from brain.experiments.gate_threshold.distributions import (
    POPULATIONS,
    PairRow,
    all_pair_rows,
    by_class,
    direction_report,
    distribution_table,
    encode_corpus,
    gram_matrix,
    population,
    sample_rows,
    stratify,
    subsample,
    summarize,
    support_overlap,
)
from brain.experiments.gate_threshold.report import (
    CELLS,
    RELATIONS_SHA,
    SCHEMA,
    STRATA,
    build_payload,
    k1_verdict,
    run_experiment,
    write_report,
)

__all__ = [
    "K1_THRESHOLD",
    "K1_TEXT",
    "HARM_TAUS",
    "COST_RATIOS",
    "POPULATIONS",
    "STRATA",
    "CELLS",
    "SCHEMA",
    "RELATIONS_SHA",
    "PairRow",
    "Ceiling",
    "encode_corpus",
    "gram_matrix",
    "all_pair_rows",
    "sample_rows",
    "population",
    "stratify",
    "subsample",
    "by_class",
    "summarize",
    "distribution_table",
    "support_overlap",
    "direction_report",
    "best_threshold",
    "overlap_bayes_error",
    "ceiling_interval",
    "loco_balanced_accuracy",
    "k1_verdict",
    "harm_table",
    "expected_loss",
    "run_experiment",
    "build_payload",
    "render_markdown",
    "write_report",
]
