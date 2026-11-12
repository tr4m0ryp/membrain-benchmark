"""Pre-registration for the k sweep: the grid and the metric list, fixed BEFORE
any number is read.

The gap this experiment closes: the 0.958 ceiling is `recall@5`, and k was
never swept. This module is the pre-registration
artifact the hard constraint asks for -- it is written into the result JSON
first, verbatim, so a reader can check the grid and metric list were fixed
in advance rather than chosen after seeing which k flattered the headline claim.

Everything here is a constant. No corpus read, no computation.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "K_GRID",
    "FREE_ARMS",
    "OUT_OF_SCOPE_ARMS",
    "METRIC_NAMES",
    "SEEDS",
    "N_RESAMPLES",
    "ALPHA",
    "BOOTSTRAP_SEED",
    "EXPECTED_CONTENT_SHA256",
    "pre_registration",
    "ADDENDUM_K",
    "ADDENDUM_REASON",
    "PUBLISHED_F24",
]

#: The k grid, pre-registered. 5 is the value the headline diagnostic (0.958
#: recall / 0.619 disambiguation) was measured at and never swept elsewhere.
K_GRID: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)

#: Arms that bill nothing: local embeddings (plain_vector) and a lexical index
#: (bm25). Both are already indexed on CPU in the existing harness.
FREE_ARMS: tuple[str, ...] = ("plain_vector", "bm25")

#: Named explicitly and never run. The response cache is keyed by the exact
#: request including k, so a rerun of these arms at k != 5 has zero cache hits,
#: and there is no API key in this environment to pay for a cold pass.
OUT_OF_SCOPE_ARMS: tuple[str, ...] = ("llm_pick", "verify_veto", "full_context")

#: Every metric this sweep reports, for every (arm, k) cell.
METRIC_NAMES: tuple[str, ...] = ("recall_at_k", "disambiguation", "precision_at_k", "mrr")

#: The project floor: "a single run is not a result" (eval/runner.py DEFAULT_SEEDS).
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Bootstrap resamples, matching the hard constraint (B=10000).
N_RESAMPLES: int = 10_000

ALPHA: float = 0.05

#: One bootstrap RNG base for the whole sweep so every (arm, k, metric) cell
#: sees the same sequence of resampled cluster draws where the cluster set is
#: identical, exactly the convention `eval/runner.py:run_arm` already documents
#: for cross-arm comparability. Streams are still separated by metric name so
#: the four metrics do not share one draw sequence.
BOOTSTRAP_SEED: int = 0

#: The frozen corpus this sweep must run on and must not modify (task hard
#: constraint). Asserted against the loaded manifest at the end of the run.
EXPECTED_CONTENT_SHA256 = (
    "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb"
)

#: NOT part of the original pre-registration above. Added as a labeled ADDENDUM
#: after the first run, at the coordinator's explicit request: max cluster size
#: is 8 (7 siblings), so full sibling coverage requires k >= 7, and the first
#: run bracketed the coverage/purity crossover between k=5 (coverage .61) and
#: k=10 (coverage 1.0, purity already down to .40) without ever measuring the
#: one or two k values where the F15/I4 precondition (bounded coverage AND
#: one-to-one purity) could plausibly hold simultaneously. This closes that
#: window by measurement instead of leaving it as an inference from the bracket.
ADDENDUM_K: tuple[int, ...] = (7, 8)
ADDENDUM_REASON = (
    "original grid bracketed the coverage/purity crossover at k=5 (coverage "
    "not yet complete) and k=10 (coverage complete, purity collapsed) without "
    "measuring k in {7, 8} = the largest cluster's sibling count; added to "
    "measure rather than infer whether the precondition is satisfiable at all"
)

#: Published F24 numbers (the veto experiment report), reused
#: VERBATIM for the corrected headroom recomputation -- NOT rerun here
#: (llm_pick and verify_veto are out of scope, no API key, cache keyed on the
#: exact request). These are on the 180-query stratified subsample, which is
#: why the control here (0.6000) differs from this sweep's full-360-query
#: plain_vector control (0.6194): mixing them was the bug the coordinator
#: flagged. Every headroom figure in `headroom.py:corrected_headroom` holds
#: this control fixed and varies only the ceiling.
PUBLISHED_F24: dict[str, float] = {
    "n_subsample": 180.0,
    "a_ctrl_subsample": 0.6000,        # plain_vector disambiguation, 180-query subsample
    "a_llm_pick_subsample": 0.9444,    # llm_pick disambiguation, same subsample
    "a_verify_veto_subsample": 0.8833,  # verify_veto disambiguation, same subsample
    "r_at_5_paper": 0.958,             # the paper's quoted ceiling at k=5
}


def pre_registration() -> dict[str, Any]:
    """The registration block, written into the result file before any number
    computed from data appears next to it."""
    return {
        "k_grid": list(K_GRID),
        "free_arms": list(FREE_ARMS),
        "out_of_scope_arms": {
            name: "requires a model call; no API key here and the response cache "
            "is keyed by the exact request including k, so a rerun at k != 5 "
            "would miss on every entry"
            for name in OUT_OF_SCOPE_ARMS
        },
        "metrics": list(METRIC_NAMES) + ["accuracy_by_cluster_size"],
        "seeds": list(SEEDS),
        "bootstrap": {
            "unit": "interference cluster (never query)",
            "method": "BCa, brain.eval.bootstrap.cluster_bootstrap_ci",
            "n_resamples": N_RESAMPLES,
            "alpha": ALPHA,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "corpus": {
            "expected_content_sha256": EXPECTED_CONTENT_SHA256,
            "frozen": True,
            "modification_policy": "read-only; SHA asserted unchanged at the end of the run",
        },
        "questions": [
            "Q1: where does recall@k saturate -- is 0.958 near a plateau or on a slope?",
            "Q2: what does disambiguation accuracy do as k grows -- does it fall?",
            "Q3: at which k does the recollection-veto precondition (F15/I4, a bounded "
            "enumerable competitor set) stop being satisfiable, and what does headroom "
            "H = (A - A_ctrl) / (R@k - A_ctrl) do across k?",
        ],
        # Explicitly separated from the fields above: NOT decided before the first
        # run's numbers were read, and labeled as such rather than folded in as if
        # it always belonged to the original pre-registration.
        "addendum": {
            "k": list(ADDENDUM_K),
            "reason": ADDENDUM_REASON,
            "published_f24_basis_for_headroom": dict(PUBLISHED_F24),
        },
    }
