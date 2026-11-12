"""Task 008: neighbourhood-conditional pooling (NCP), swept and reported.

Attacks the measured interference failure at the POOLING layer, where mean
pooling averages every token equally and the distinctive tokens of an episode
drown in the two thirds of its text that is boilerplate. Tokens are reweighted by
how distinctive they are relative to that episode's neighbours, before pooling:

    df_t^(i) = |{ j in N(i) : t in x_j }| / |N(i)|        w_t^(i) = (1 - df_t^(i)) ** gamma
    score(q, i) = alpha * cos(q_full, v_full^(i)) + (1 - alpha) * cos(q_dist, v_dist^(i))

No training, no new model weights, and ZERO LLM calls on the query path. The
target is not to beat `llm_pick` (0.9444) but to approach it at zero per-query
cost, since `llm_pick` bills about $0.014 on every query.

    python -m brain.experiments.gate_ncp.report --seeds 5 --k 5

    from brain.experiments.gate_ncp import run_grid, write_report
    run = run_grid(load_corpus())

`gamma = 0` sends every weight to exactly 1, so the encoder reduces to mean
pooling and the stored vectors are byte-identical to today's. That is asserted in
`tests/test_ncp.py` and is the free correctness test on the whole implementation.
"""

from brain.experiments.gate_ncp.channels import (
    CorpusState,
    blended_geometry,
    corpus_state,
    dist_channel,
    geometry,
    query_channel,
    weight_rows,
)
from brain.experiments.gate_ncp.arms import cost_probe, run_baselines, run_retriever_arm
from brain.experiments.gate_ncp.markdown import render_markdown
from brain.experiments.gate_ncp.runner import (
    ALPHA_GRID,
    GAMMA_GRID,
    K_GRID,
    PREREGISTERED_CELL,
    PRIMARY_K,
    PRIMARY_QUERY_SIDE,
    SUBSAMPLE_N,
    GridRun,
    cell_specs,
    cluster_indicators,
    evaluate,
    paired_delta,
    preregistration,
    rank_ids,
    reordering,
    run_grid,
    subsample_ids,
)
from brain.experiments.gate_ncp.verdict import comparisons, verdict

__all__ = [
    "ALPHA_GRID", "GAMMA_GRID", "K_GRID", "PREREGISTERED_CELL", "PRIMARY_K",
    "PRIMARY_QUERY_SIDE", "SUBSAMPLE_N", "CorpusState", "GridRun", "blended_geometry",
    "cell_specs", "cluster_indicators", "corpus_state", "cost_probe", "dist_channel",
    "comparisons", "evaluate", "geometry", "paired_delta", "preregistration",
    "query_channel", "rank_ids", "render_markdown", "reordering", "run_baselines",
    "run_grid", "run_retriever_arm", "subsample_ids", "verdict", "weight_rows",
]
