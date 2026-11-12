"""The two arms of the veto experiment (I4), on one shared candidate set.

    from brain.baselines.veto import LlmPickRetriever, VerifyVetoRetriever

* `LlmPickRetriever` (arm A) sees all k candidates at once and returns a
  ranking. The strong naive baseline, and the arm to beat.
* `VerifyVetoRetriever` (arm B) sees ONE candidate per call and returns a
  BOOLEAN veto driven by positive evidence recalled out of that candidate.

Both subclass `CandidateArm`, which owns the plain-vector base they draw their
top-k from, so the two arms are handed byte-identical candidate lists and the
comparison is paired rather than confounded by two retrieval front ends.
"""

from brain.baselines.veto.llm_pick import (
    PICK_CACHE_DIR,
    PICK_SYSTEM_HEADER,
    CandidateArm,
    LlmCandidateArm,
    LlmPickRetriever,
)
from brain.baselines.veto.verify_veto import (
    VETO_CACHE_DIR,
    VETO_SYSTEM_HEADER,
    VerifyVetoRetriever,
    VetoDecision,
)

__all__ = [
    "CandidateArm",
    "LlmCandidateArm",
    "LlmPickRetriever",
    "PICK_CACHE_DIR",
    "PICK_SYSTEM_HEADER",
    "VerifyVetoRetriever",
    "VetoDecision",
    "VETO_CACHE_DIR",
    "VETO_SYSTEM_HEADER",
]
