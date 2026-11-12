"""Q3: the recollection-veto precondition and the headroom formula, across k.

The MemBrain paper defines headroom captured by an arm as

    H = (A - A_ctrl) / (R@5 - A_ctrl)

with A_ctrl = plain_vector's own disambiguation accuracy (the control) and R@5
= plain_vector's own recall@5 (the ceiling): the fraction of the gap between
"the answer is retrievable" and "the answer, unaided, outranks its siblings"
that a downstream operator closes. It was measured for llm_pick (96.2%) and
verify_veto (79.1%) at k=5 only.

Those two arms are OUT OF SCOPE here (no API key, cache keyed on k, task hard
constraint), so H cannot be recomputed for either at any k on this grid. What
CAN be computed honestly from the two free arms:

* the **denominator alone**, gap(k) = R@k - A_ctrl(k), using only
  plain_vector -- this is "how much headroom is available for ANY downstream
  operator to capture" and needs no downstream arm at all;
* an **illustrative, explicitly non-canonical** H using bm25's own
  disambiguation as a stand-in numerator. This is flagged because bm25 is a
  different top-level retrieval mechanism, not a reranker operating on
  plain_vector's candidate set the way llm_pick/verify_veto are, so its H is
  not the quantity the paper's table reports -- it only shows what a free
  second opinion buys, which is a different (still relevant) question.
"""

from __future__ import annotations

import math
from typing import Any

from brain.experiments.gate_k_sweep.coverage import CoveragePoint
from brain.experiments.gate_k_sweep.point import KPoint
from brain.experiments.gate_k_sweep.registration import PUBLISHED_F24

__all__ = ["headroom_row", "headroom_table", "corrected_headroom"]


def headroom_row(
    ctrl: KPoint, illustrative_arm: KPoint, coverage: CoveragePoint | None
) -> dict[str, Any]:
    """One k's headroom row: the available gap, the illustrative H, and the
    F15/I4 precondition state at that k."""
    a_ctrl = ctrl.disambiguation.mean
    r_ctrl = ctrl.recall_at_k.mean
    gap = r_ctrl - a_ctrl
    a_arm = illustrative_arm.disambiguation.mean
    h = (a_arm - a_ctrl) / gap if math.isfinite(gap) and abs(gap) > 1e-12 else math.nan
    row: dict[str, Any] = {
        "k": int(ctrl.k),
        "A_ctrl_plain_vector_disambiguation": a_ctrl,
        "R_at_k_plain_vector_recall": r_ctrl,
        "headroom_available_gap": gap,
        "illustrative_arm": illustrative_arm.arm,
        "A_illustrative_arm_disambiguation": a_arm,
        "H_illustrative_non_canonical": h,
        "H_canonical_llm_pick_verify_veto": (
            "NOT COMPUTABLE at this k: out of scope, no model call available"
        ),
    }
    if coverage is not None:
        row["precondition"] = {
            "sibling_coverage": coverage.coverage.mean,
            "competitor_purity": coverage.purity.mean,
            "bounded_enumerable_satisfied": coverage.coverage.mean >= 1.0 - 1e-9,
        }
    return row


def headroom_table(
    ctrl_points: dict[int, KPoint],
    illustrative_points: dict[int, KPoint],
    coverage_points: dict[int, CoveragePoint],
) -> list[dict[str, Any]]:
    """The headroom row for every k both arms were measured at."""
    ks = sorted(set(ctrl_points) & set(illustrative_points))
    return [
        headroom_row(ctrl_points[k], illustrative_points[k], coverage_points.get(k))
        for k in ks
    ]


def corrected_headroom(ceiling_by_k: dict[int, float]) -> dict[str, Any]:
    """H for llm_pick and verify_veto, ONE stated basis, ceiling varied only.

    Fixes the bug the coordinator flagged: the first pass divided the published
    llm_pick/verify_veto numerators (measured with A_ctrl = 0.6000 on the
    180-query stratified subsample, F24) by a denominator built from THIS
    sweep's full-360-query control (0.6194). Different corpus subset, different
    control value -- not a valid pairing.

    The basis held fixed here, throughout, is F24's own: A_ctrl = 0.6000 and
    the two published numerators (0.9444, 0.8833), all three from the SAME
    180-query subsample. The only thing that changes across rows is the
    ceiling `R`, taken from `ceiling_by_k` (this sweep's measured full-corpus
    plain_vector recall@k). Recall@k was never measured on the 180-query
    subsample, so every ceiling except k=5 (which reproduces the paper's own
    0.958 to three decimals) is an explicitly labeled cross-basis assumption:
    "if the subsample's own recall ceiling moves with k the way the full
    corpus's does, THEN H at that k would be this."
    """
    a_ctrl = PUBLISHED_F24["a_ctrl_subsample"]
    numerators = {
        "llm_pick": PUBLISHED_F24["a_llm_pick_subsample"],
        "verify_veto": PUBLISHED_F24["a_verify_veto_subsample"],
    }
    rows: list[dict[str, Any]] = []
    for k in sorted(ceiling_by_k):
        r = ceiling_by_k[k]
        gap = r - a_ctrl
        row: dict[str, Any] = {
            "k": int(k),
            "ceiling_source": (
                "paper-quoted (0.958), same subsample-adjacent basis"
                if abs(r - PUBLISHED_F24["r_at_5_paper"]) < 5e-3
                else "full-corpus plain_vector recall@k, measured this sweep -- "
                "NOT measured on the 180-query subsample; combining it with the "
                "subsample control/numerators below is a stated assumption, not a "
                "second measurement"
            ),
            "R_ceiling": r,
            "A_ctrl_subsample_180q": a_ctrl,
            "gap": gap,
        }
        for name, numerator in numerators.items():
            row[f"H_{name}"] = (
                (numerator - a_ctrl) / gap if math.isfinite(gap) and abs(gap) > 1e-12 else math.nan
            )
            row[f"A_{name}_subsample_180q"] = numerator
        rows.append(row)
    return {
        "basis": (
            "control and both numerators fixed at F24's 180-query subsample values "
            "(A_ctrl=0.6000, llm_pick=0.9444, verify_veto=0.8833); only the ceiling R "
            "varies across rows"
        ),
        "rows": rows,
    }
