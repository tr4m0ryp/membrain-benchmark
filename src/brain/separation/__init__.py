"""Write-time separation transform, Invention C-A / model M-A (Task 005).

`SeparationPolicy` is the write path: probe, decorrelate, store. `stored_dim`
tells a caller how wide to open the store before it can hold a given mode's
representation, and `calibrate_tau` translates a collision threshold between
those representations. `python -m brain.separation` prints the separation and
retrievability numbers for both arms.
"""

from brain.separation.calibrate import calibrate_tau
from brain.separation.geometry import (
    AGGREGATIONS,
    DEFAULT_AGGREGATION,
    cosine_after_push,
    difference_directions,
    hard_threshold,
    push_apart,
    tiebreak_direction,
    unit_rows,
)
from brain.separation.policy import SeparationPolicy, build_policy
from brain.separation.result import (
    DEFAULT_K_PROBE,
    DEFAULT_S,
    DEFAULT_TAU,
    DistinctHint,
    SeparationConfigError,
    WriteResult,
)
from brain.separation.transform import (
    DEFAULT_EXPANSION,
    DEFAULT_FAN_IN_FRACTION,
    DEFAULT_SPARSENESS,
    MODES,
    Decorrelate,
    ExpandSparsify,
    SeparationTransform,
    make_transform,
    stored_dim,
)

__all__ = [
    "SeparationPolicy",
    "build_policy",
    "WriteResult",
    "DistinctHint",
    "SeparationConfigError",
    "calibrate_tau",
    "DEFAULT_TAU",
    "DEFAULT_S",
    "DEFAULT_K_PROBE",
    "SeparationTransform",
    "Decorrelate",
    "ExpandSparsify",
    "make_transform",
    "stored_dim",
    "MODES",
    "AGGREGATIONS",
    "DEFAULT_AGGREGATION",
    "DEFAULT_EXPANSION",
    "DEFAULT_SPARSENESS",
    "DEFAULT_FAN_IN_FRACTION",
    "push_apart",
    "difference_directions",
    "tiebreak_direction",
    "hard_threshold",
    "unit_rows",
    "cosine_after_push",
]
