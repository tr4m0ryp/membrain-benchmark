"""Write-time MMR: the sharp alternative flagged in F10, a first-class arm.

F10 warns that a trivial write-time MMR might match the separation transform.
If it does, the C-A claim collapses from "write-time separation" to "apply MMR
at write", so this arm is built to have its best shot, not to lose.

**Derivation (not an invented transform).** MMR ranks a candidate d against a
query q and an already-selected set S by
`lambda * sim(d, q) - (1 - lambda) * max_{s in S} sim(d, s)`.
At write time there is no query, so the item's own content vector x plays the
role of q and the already-stored set plays the role of S, with `x_nn` the
nearest stored neighbour (exactly the `max` term of the MMR formula). Two
readings of "reduce redundancy" follow, and both are implemented:

    x' = normalize(x - (1 - lambda) * <x, x_nn> * x_nn)     (projection, default)
    x' = argmax_{||u||=1} [ lambda*<u, x> - (1 - lambda)*<u, x_nn> ]
       = normalize(lambda * x - (1 - lambda) * x_nn)        (objective)

Both are the identity at lambda = 1 and both run at lambda = 0.5, the standard
default (Carbonell and Goldstein 1998; the value that ships as `lambda_mult` in
current RAG libraries).

**Why projection is the default.** Measured on the frozen corpus at the standard
lambda = 0.5, projection scores 0.514 disambiguation against 0.431 for the
objective form, so the default is the stronger of the two readings; a baseline
implemented at its weaker setting would be a correctness bug here. The objective
form also degenerates below lambda = 0.5, where `-x_nn` dominates and the stored
code points away from its own content (measured: top-1 accuracy collapses to
0.106 at lambda = 0.3 while pairwise order partly survives). The choice of form
was made on the frozen corpus and in the conservative direction, which is
disclosed rather than hidden; lambda itself is NOT tuned, it stays at 0.5.

**The reading not implemented.** Write-time MMR could instead mean admission
control: refuse to store an item whose redundancy exceeds a threshold. That is
dedup-on-write, the anti-separation policy this project exists to refute, and it
cannot compete on a task whose answer is the episode it would have dropped.

**How this differs from M-A separation (Task 005), which is the point.** MMR is
greedy and one-sided: only the arriving item moves, the already-stored codes are
frozen, and every write is adjusted whether or not it collides. M-A is
symmetric (both members of a colliding pair move) and gated by the collision
threshold tau. Read time is identical to the plain-vector arm: this arm is about
WRITE time only, so no read-time MMR is applied anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from brain.baselines.vectors import VectorCache
from brain.baselines.plain_vector import VectorStoreArm
from brain.store import EpisodicStore, Trace

__all__ = ["WriteTimeMmrRetriever", "DEFAULT_LAMBDA", "DEFAULT_MODE", "OBJECTIVE", "PROJECTION"]

#: Standard MMR trade-off value: relevance (self-fidelity) versus redundancy.
DEFAULT_LAMBDA = 0.5

OBJECTIVE = "objective"
PROJECTION = "projection"
_MODES = (PROJECTION, OBJECTIVE)
#: The stronger of the two readings at the standard lambda (see module docstring).
DEFAULT_MODE = PROJECTION

#: Below this the adjusted vector is numerically all rounding error: two codes
#: that near-coincide would otherwise be normalized up from a 1e-8 residual into
#: an arbitrary direction. Such a write keeps its original vector and is counted.
_MIN_NORM = 1e-6


class WriteTimeMmrRetriever(VectorStoreArm):
    """MMR-style redundancy reduction applied to the STORED representation.

    One k-NN probe plus one rank-1 update per write, the same cost profile as
    the separation policy, so a difference in the result is a difference in the
    transform and not in the budget.
    """

    name = "write_time_mmr"

    def __init__(
        self,
        *,
        lambda_: float = DEFAULT_LAMBDA,
        mode: str = DEFAULT_MODE,
        store: EpisodicStore | None = None,
        db_path: str | Path = ":memory:",
        cache: VectorCache | None = None,
        name: str | None = None,
    ) -> None:
        if not 0.0 <= float(lambda_) <= 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {lambda_!r}")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        super().__init__(
            store=store,
            db_path=db_path,
            cache=cache,
            name=name,
            params={
                "lambda": float(lambda_),
                "mode": mode,
                "applied_at": "write",
                "read_time_rerank": None,
                "symmetric": False,
                "metric": "cosine",
            },
        )
        self._lambda = float(lambda_)
        self._mode = mode

    def _write(self, traces: Sequence[Trace], vectors: np.ndarray) -> None:
        """Write one at a time: each item is diversified against what is already in.

        Sequential on purpose. A batch write would diversify every item against
        the same empty store, which is not MMR, it is a no-op.
        """
        adjusted_count = 0
        degenerate = 0
        shifts: list[float] = []
        for trace, vector in zip(traces, vectors):
            stored, moved, degraded = self._diversify(np.asarray(vector, dtype=np.float32))
            if moved:
                adjusted_count += 1
                shifts.append(float(np.dot(stored.astype(np.float64), vector.astype(np.float64))))
            degenerate += int(degraded)
            self._store.add(trace, stored)
        self.stats["adjusted_writes"] = adjusted_count
        self.stats["degenerate_writes"] = degenerate
        # Mean cosine between the original embedding and what was actually
        # stored: 1.0 would mean this arm silently equals plain vector.
        self.stats["mean_self_similarity"] = float(np.mean(shifts)) if shifts else 1.0

    def _diversify(self, vector: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        """Return (stored vector, was adjusted, hit the degenerate guard)."""
        hits = self._store.knn(vector, 1)
        if not hits:
            return vector, False, False  # first write, nothing to be redundant with
        neighbour_id, distance = hits[0]
        similarity = 1.0 - float(distance)
        if similarity <= 0.0:
            # Orthogonal or opposed already: MMR has no redundancy to remove.
            return vector, False, False
        neighbour = self._store.get_vector(neighbour_id)
        if neighbour is None:  # pragma: no cover - knn returned it a line ago
            return vector, False, False
        x = vector.astype(np.float64, copy=False)
        n = neighbour.astype(np.float64, copy=False)
        if self._mode == OBJECTIVE:
            candidate = self._lambda * x - (1.0 - self._lambda) * n
        else:
            candidate = x - (1.0 - self._lambda) * float(np.dot(x, n)) * n
        norm = float(np.linalg.norm(candidate))
        if not np.isfinite(norm) or norm < _MIN_NORM:
            return vector, False, True
        return (candidate / norm).astype(np.float32), True, False


def _smoke() -> None:
    """Tiny CPU path: python -m brain.baselines.write_time_mmr."""
    from brain.baselines.smoke import run_smoke

    with WriteTimeMmrRetriever() as arm:
        run_smoke(arm)


if __name__ == "__main__":
    _smoke()
