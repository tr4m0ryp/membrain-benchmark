"""Task 008: the minimal falsifying experiment for H-A (THE GATE).

Does write-time separation (Invention C-A, model M-A) beat plain vector, BM25,
write-time MMR and full context at disambiguating similar-but-distinct episodes?
The pre-registered kill criterion decides, computed from the numbers:

    H-A dies if separation fails to beat plain-vector AND full-context by >= 5
    points disambiguation accuracy with non-overlapping CIs, OR if the
    accuracy-vs-cluster-size curve is flat for the baseline too.

    from brain.experiments.gate_separation import run_gate, write_gate_report
    run = run_gate(load_corpus())
    json_path, md_path, payload = write_gate_report(run, command="...", directory="results")

`SeparationArm` is the piece Task 005 did not ship: its write path behind Task
006's uniform `Retriever` interface, with the cue passed through the same
transform as the store and tau calibrated into the stored geometry.
"""

from brain.experiments.gate_separation.arm import CacheEmbedder, SeparationArm, arm_label
from brain.experiments.gate_separation.diagnostics import (
    geometry_sweep,
    m2_invariance,
    m2_within_mode,
    stored_geometry,
)
from brain.experiments.gate_separation.gate import GateRun, run_gate, run_spec
from brain.experiments.gate_separation.markdown import render_markdown
from brain.experiments.gate_separation.report import (
    build_payload,
    verdict_line,
    write_gate_report,
)
from brain.experiments.gate_separation.selection import (
    flat_curve_check,
    harness_validation,
    select_candidate,
    sweep_verdicts,
    verdict_for,
)
from brain.experiments.gate_separation.specs import (
    BASELINE_ARMS,
    MODES,
    PREREGISTERED_S,
    S_GRID,
    SMOKE_S_GRID,
    ArmSpec,
    SeededFactory,
    baseline_specs,
    separation_spec,
    sweep_specs,
)

__all__ = [
    "SeparationArm",
    "CacheEmbedder",
    "arm_label",
    "ArmSpec",
    "SeededFactory",
    "BASELINE_ARMS",
    "MODES",
    "S_GRID",
    "SMOKE_S_GRID",
    "PREREGISTERED_S",
    "baseline_specs",
    "separation_spec",
    "sweep_specs",
    "GateRun",
    "run_gate",
    "run_spec",
    "m2_invariance",
    "m2_within_mode",
    "stored_geometry",
    "geometry_sweep",
    "select_candidate",
    "verdict_for",
    "sweep_verdicts",
    "flat_curve_check",
    "harness_validation",
    "build_payload",
    "verdict_line",
    "write_gate_report",
    "render_markdown",
]
