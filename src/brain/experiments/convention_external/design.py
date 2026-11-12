"""The pre-registration. Written and committed BEFORE any accuracy was computed.

Every choice a reader could suspect of being made after seeing a number is fixed
here: the subset, the sample size, the threshold grid, the three conventions,
the scoring rule, the resampling unit, the primary cell, and the rule that
decides what the outcome is called. `PRE_REGISTRATION` is copied verbatim into
the result JSON, so the file carries its own design.

**What is deliberately NOT free here.** The distractor definition is the one
choice with real leverage, because the crediting convention can only bite when a
merged record absorbs the queried event together with another event some
question distinguishes it from. It is therefore taken from LoCoMo's own evidence
annotation and never from similarity: distractors are the evidence turns of
OTHER included questions, which is an annotator's judgement that those turns are
distinct answerable events. Two scopes are registered in advance and BOTH are
reported, so the narrow one cannot be presented as the result if the wide one
disagrees.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["CONVENTIONS", "MODES", "PRE_REGISTRATION", "PRIMARY", "SCOPES", "THRESHOLDS",
           "classify_outcome"]

#: Same grid as the synthetic run (`brain.eval.consolidation_delta.THRESHOLDS`).
THRESHOLDS: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.95)
MODES: tuple[str, ...] = ("destructive", "non_destructive", "placebo_shared", "placebo_self")
#: The three readings under test, in the order the paper brackets them.
CONVENTIONS: tuple[str, ...] = ("credited", "expected", "strict")
SCOPES: tuple[str, ...] = ("session", "conversation")

#: The single cell the sign-flip verdict is read off. Fixed in advance so no
#: cell can be selected after the fact; the whole grid is reported anyway.
PRIMARY: dict[str, Any] = {
    "mode": "destructive",
    "threshold": 0.85,
    "depth": "5",
    "scope": "session",
    "resampling_unit": "conversation",
}

PRE_REGISTRATION: dict[str, Any] = {
    "registered_before_any_result": True,
    "question": (
        "On a public benchmark, does the reported SIGN of the consolidation "
        "delta depend on the convention for crediting a merged record that "
        "absorbed several events?"
    ),
    "dataset": {
        "name": "LoCoMo-10",
        "citation": "Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang. ACL 2024.",
        "source": "https://github.com/snap-research/locomo.git",
        "commit": "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
        "file_sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "subset": "ALL ten conversations, ALL usable annotated questions. No subsampling.",
        "verified_shape": {
            "conversations": 10, "raw_turns": 5882, "raw_qa": 1986,
            "episodes_after_collapsing_identical_turns": 5880,
            "sessions": 272, "usable_queries": 1527,
        },
        "exclusions": {
            "category_5_adversarial": "unanswerable by construction, carries adversarial_answer",
            "empty_evidence": "no annotated event to be the answer",
            "unresolved_evidence": "an evidence dia_id with no matching turn (9 of 2815 refs)",
        },
    },
    "mapping": {
        "episode": "one dialogue turn, rendered as the benchmark's own evaluation renders it",
        "store": "one per conversation; a cross-conversation merge is a pooling artifact",
        "cluster": "the LoCoMo session",
        "correct_trace": "the earliest evidence turn of the question",
        "distractors": (
            "evidence turns of OTHER included questions, from the annotation only, "
            "never from similarity; scope `session` is primary, `conversation` is "
            "reported alongside it"
        ),
    },
    "consolidation": {
        "machinery": "brain.consolidate, unmodified",
        "linkage": "complete (every pair inside a group is itself above tau), not transitive closure",
        "thresholds": list(THRESHOLDS),
        "modes": list(MODES),
        "passes_per_condition": 1,
        "model": "claude-haiku-4-5 via the claude CLI (no ANTHROPIC_API_KEY on this machine)",
        "temperature": 0.0,
        "temperature_enforced": False,
        "note": "the CLI exposes no temperature flag; determinism rests on the merge cache",
    },
    "conventions": {
        "credited": "the merged record is credited in full for the queried event (generous)",
        "expected": "1/(1+a) over the a+1 events the record conflates (the reader who must pick one)",
        "strict": "the merged record earns nothing once a >= 1 (the binding is gone)",
        "implementation": "brain.eval.consolidation_delta.query_indicators, unmodified",
    },
    "metrics": [
        "credited@5", "expected@5", "strict@5",
        "credited@full", "expected@full", "strict@full",
        "answer_recall@1",
    ],
    "statistics": {
        "estimator": "paired BCa cluster bootstrap, brain.eval.bootstrap.paired_delta_ci",
        "resamples": 10000,
        "resampling_unit_primary": "conversation (n = 10)",
        "resampling_unit_secondary": "session (n = 272)",
        "query_level_bootstrap": "FORBIDDEN; the design effect on this project's own data was 2.37",
        "pairing": "same clusters, same queries, eligibility pinned to the before store",
    },
    "seeds": {
        "seeds": [0],
        "justification": (
            "the pass is deterministic given the embedder, complete linkage and a "
            "temperature-0 merge cache; determinism is asserted by re-running one "
            "conversation under a second seed rather than paid for on all ten"
        ),
    },
    "primary_cell": dict(PRIMARY),
    "decision_rule": {
        "SIGN_FLIPS": "at the primary cell, delta under `credited` and under `strict` have opposite signs",
        "STRONG_FLIP": "SIGN_FLIPS and both BCa intervals exclude zero, in opposite directions",
        "IMMUNE": (
            "at the primary cell, `credited`, `expected` and `strict` return the SAME "
            "value on at least 95 percent of queries, i.e. the winning record absorbed "
            "none of that query's own distractors (a = 0) or the query was lost "
            "outright; the lever then has almost nothing to act on"
        ),
        "NO_FLIP": "neither of the above",
    },
    "budget_usd": 25.0,
}


def classify_outcome(credited: float, strict: float, *, credited_ci: Mapping[str, float],
                     strict_ci: Mapping[str, float], zero_absorbed_fraction: float) -> str:
    """Apply `decision_rule` to the primary cell. No other reading is the verdict."""
    if zero_absorbed_fraction >= 0.95:
        return "IMMUNE"
    if not (credited > 0.0) == (strict > 0.0) and credited != strict:
        excludes_zero = (
            (credited_ci["lower"] > 0.0 or credited_ci["upper"] < 0.0)
            and (strict_ci["lower"] > 0.0 or strict_ci["upper"] < 0.0)
        )
        return "STRONG_FLIP" if excludes_zero else "SIGN_FLIPS"
    return "NO_FLIP"
