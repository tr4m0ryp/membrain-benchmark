"""What one separating write reports, and the knobs it reports against.

One question: what does the experiment get to observe about a write. Kept apart
from the write path so that Tasks 007 and 008 can depend on the observables
without importing the mechanism, and so that adding a diagnostic never means
touching the code that moves vectors.

Every field here exists because some downstream claim needs it: `collisions` is
the empirical collision rate rho that M-A's Acc ~ 1/(1 + rho*(m-1)) prediction is
stated in terms of, `updated_ids` is the write amplification the cost claim rests
on, and `probe_saturated` is the honesty flag that says a top-k probe may have
missed a colliding neighbour rather than found none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from brain.store import Trace

__all__ = [
    "DEFAULT_TAU",
    "DEFAULT_S",
    "DEFAULT_K_PROBE",
    "DistinctHint",
    "SeparationConfigError",
    "WriteResult",
]

#: Collision threshold tau. Read off the frozen corpus manifest rather than
#: tuned: sibling similarity there is 0.824 +/- 0.060 and the largest non-sibling
#: pair is 0.715, so 0.75 sits above every non-sibling pair and below the sibling
#: mean. Quoted in RAW embedding units; see `calibrate_tau` for the sparse arm.
DEFAULT_TAU = 0.75
#: Separation strength s. The disambiguation-versus-completion frontier is swept
#: over this, so the default is a starting point and explicitly not a tuned value.
DEFAULT_S = 0.25
#: Neighbours pulled per probe. sqlite-vec has no radius query, so tau is applied
#: to a top-k; 32 covers the largest cluster in the corpus (8) with room to spare,
#: and `probe_saturated` reports the case where it did not.
DEFAULT_K_PROBE = 32

#: Either a blanket assertion or a predicate over (new trace, neighbour trace).
#: None means "every non-identical neighbour is a distinct episode", which is the
#: only safe default: inferring sameness from similarity is the failure mode.
DistinctHint = bool | Callable[[Trace, Trace], bool] | None


class SeparationConfigError(ValueError):
    """Policy and store disagree about the stored representation."""


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What one write did. Every field is an observable the experiment reads."""

    trace_id: str
    inserted: bool
    deduped: bool
    collisions: tuple[tuple[str, float], ...] = ()
    updated_ids: tuple[str, ...] = ()
    probe_saturated: bool = False
    skipped_ids: tuple[str, ...] = ()

    @property
    def separated(self) -> bool:
        """True when the write actually moved stored vectors."""
        return bool(self.updated_ids)

    @property
    def n_collisions(self) -> int:
        return len(self.collisions)
