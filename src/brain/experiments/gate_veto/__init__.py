"""Task 004: the veto experiment (I4), judged by the pre-registered K-I4.

Two arms on top of the SAME plain-vector top-k, where recall is 0.958 and
disambiguation is 0.619, so the correct memory is almost always retrieved and
then out-ranked:

* **arm A, `llm_pick`** -- all k candidates in one prompt, a ranking out. The
  strong naive baseline and the arm to beat.
* **arm B, `verify_veto`** -- each candidate judged alone, a BOOLEAN veto out,
  fired only on positive evidence recalled from that candidate.

Both are scored on BOTH axes in the same pass: disambiguation gain AND
correct-item rejection rate. A gain reported without its rejection rate would
hide the harm the veto can do, which is the failure the Mnemonic Similarity
Task's Lure Discrimination Index is designed to prevent.

    python -m brain.experiments.gate_veto.report --seeds 5 --k 5 --smoke 10
    python -m brain.experiments.gate_veto.report --seeds 5 --k 5
"""

from brain.experiments.gate_veto.report import (
    FROZEN_SHA,
    K_I4,
    RECALL_AT_5_CEILING,
    build_payload,
    directory_sha,
    evaluate_k_i4,
    headroom,
    main,
    render_markdown,
    write_report,
)
from brain.experiments.gate_veto.runner import (
    ARM_A,
    ARM_B,
    AXES,
    CONTROL,
    ArmRun,
    QueryRecord,
    VetoRun,
    arm_groups,
    cost_estimate,
    run_veto_arm,
    run_veto_gate,
)

__all__ = [
    "AXES",
    "CONTROL",
    "ARM_A",
    "ARM_B",
    "QueryRecord",
    "ArmRun",
    "VetoRun",
    "arm_groups",
    "run_veto_arm",
    "run_veto_gate",
    "cost_estimate",
    "K_I4",
    "FROZEN_SHA",
    "RECALL_AT_5_CEILING",
    "directory_sha",
    "evaluate_k_i4",
    "headroom",
    "build_payload",
    "render_markdown",
    "write_report",
    "main",
]
