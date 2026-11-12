"""The two write-time separation transforms behind one interface (model M-A).

A transform answers two questions and nothing else:

* `encode` -- what representation goes into the store, applied identically to a
  written trace and to a read cue, so the two live in the same space.
* `separate` -- given a new vector and the stored vectors it collides with, what
  do all of them become. Ids never reach this layer; `updates` comes back
  row-aligned with the input so the policy can zip them back onto ids.

**Ordering follows F10 (verified DR5), not intuition.** Sparseness per se is not
the separation driver; EXPANSION and explicit DECORRELATION predict separation
better than top-k sparsity alone. So the default arm is pure decorrelation with
no sparsity at all, and the sparse arm is built as expansion + decorrelation +
winner-take-all, in that order, with the top-k step last and optional. Reading
the two arms against each other is what isolates how much of any effect the
sparsity is actually responsible for.

`s = 0` is a strict no-op in both arms: the returned vector is the input array's
bytes, not a renormalized copy of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from brain.config import SEED, get_rng
from brain.separation.geometry import (
    AGGREGATIONS,
    DEFAULT_AGGREGATION,
    hard_threshold,
    push_apart,
    unit_rows,
)

__all__ = [
    "MODES",
    "DEFAULT_EXPANSION",
    "DEFAULT_SPARSENESS",
    "DEFAULT_FAN_IN_FRACTION",
    "SeparationTransform",
    "Decorrelate",
    "ExpandSparsify",
    "make_transform",
    "stored_dim",
]

MODES: tuple[str, ...] = ("decorrelate", "expand_sparsify")

#: Expansion ratio D/d. The dentate gyrus expands entorhinal input roughly
#: tenfold (EC ~50k -> DG ~500k, F6); 10 keeps that ratio at d = 384 -> D = 3840.
DEFAULT_EXPANSION = 10
#: Code sparseness a, the fraction of active dims. DG theory says 2-5% (F1).
DEFAULT_SPARSENESS = 0.05
#: Fan-in per expanded unit as a fraction of d. 0.1 is the FlyHash sampling rate
#: and lands on 38 inputs at d = 384, next to the ~46 mossy inputs per CA3 cell.
DEFAULT_FAN_IN_FRACTION = 0.1


class SeparationTransform(ABC):
    """Representation map plus collision response. Stateless in s."""

    name: str = "abstract"

    def __init__(self, in_dim: int, *, aggregation: str = DEFAULT_AGGREGATION) -> None:
        if not isinstance(in_dim, int) or isinstance(in_dim, bool) or in_dim <= 1:
            raise ValueError(f"in_dim must be an int > 1, got {in_dim!r}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {aggregation!r}, expected {AGGREGATIONS}")
        self.in_dim = int(in_dim)
        self.aggregation = str(aggregation)

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Width of the stored vector. The store must be opened at this dim."""

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Map a raw embedding (1-D or 2-D) into the stored space."""

    @abstractmethod
    def separate(
        self, x_new: np.ndarray, x_collisions: np.ndarray, s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply T_s. Returns (x_new', updates) in the stored space."""

    def params(self) -> dict[str, object]:
        """Everything needed to rebuild this transform, for a run manifest."""
        return {
            "mode": self.name,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "aggregation": self.aggregation,
        }

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params().items()))
        return f"{type(self).__name__}({items})"


class Decorrelate(SeparationTransform):
    """Push the colliding pair apart along their difference direction (default).

    `encode` is the identity on purpose. The embedder already emits float32 unit
    rows, and renormalizing them here would perturb the last ulp of every stored
    vector, which is precisely what the s = 0 control arm must not do.
    """

    name = "decorrelate"

    @property
    def out_dim(self) -> int:
        return self.in_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        if arr.shape[-1] != self.in_dim:
            raise ValueError(f"expected dim {self.in_dim}, got {arr.shape[-1]}")
        return arr

    def separate(
        self, x_new: np.ndarray, x_collisions: np.ndarray, s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return push_apart(x_new, x_collisions, s, aggregation=self.aggregation)


class ExpandSparsify(SeparationTransform):
    """Expansion recoding, then decorrelation, then winner-take-all (DG analog).

    The projection is a structurally sparse random matrix: each expanded unit
    samples `fan_in` input dims with +/-1 weights, the sparse-random fan-in of
    F6. Signed weights rather than FlyHash's 0/1 because the embedding is signed;
    a 0/1 matrix would give every unit the same DC term and waste the expansion.
    Scaling by 1/sqrt(fan_in) makes the projection inner-product preserving in
    expectation, so expansion alone leaves cosine roughly intact and decorrelates
    only through projection variance. The overlap reduction comes from the
    winner-take-all step and from the explicit push, which is exactly the
    decomposition F10 says to test rather than assume.

    The matrix is materialized dense (d x D floats, 5.9 MB at 384 x 3840) because
    at these sizes one GEMM beats any sparse gather; the sparsity is structural,
    not a storage claim.
    """

    name = "expand_sparsify"

    def __init__(
        self,
        in_dim: int,
        *,
        expansion: int = DEFAULT_EXPANSION,
        sparseness: float = DEFAULT_SPARSENESS,
        fan_in_fraction: float = DEFAULT_FAN_IN_FRACTION,
        seed: int | None = None,
        resparsify: bool = True,
        aggregation: str = DEFAULT_AGGREGATION,
    ) -> None:
        super().__init__(in_dim, aggregation=aggregation)
        if not isinstance(expansion, int) or isinstance(expansion, bool) or expansion < 1:
            raise ValueError(f"expansion must be an int >= 1, got {expansion!r}")
        if not 0.0 < float(sparseness) <= 1.0:
            raise ValueError(f"sparseness a must be in (0, 1], got {sparseness!r}")
        if not 0.0 < float(fan_in_fraction) <= 1.0:
            raise ValueError(f"fan_in_fraction must be in (0, 1], got {fan_in_fraction!r}")
        self.expansion = int(expansion)
        self.sparseness = float(sparseness)
        self.fan_in_fraction = float(fan_in_fraction)
        self.seed = SEED if seed is None else int(seed)
        self.resparsify = bool(resparsify)
        self._out_dim = self.in_dim * self.expansion
        self.fan_in = max(1, min(self.in_dim, round(self.fan_in_fraction * self.in_dim)))
        # k >= 1 keeps the code from collapsing to the zero vector, which the
        # store would reject as directionless rather than accept as NaN.
        self.active = max(1, min(self._out_dim, round(self.sparseness * self._out_dim)))
        self._matrix: np.ndarray | None = None

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def params(self) -> dict[str, object]:
        return {
            **super().params(),
            "expansion": self.expansion,
            "sparseness": self.sparseness,
            "fan_in": self.fan_in,
            "active": self.active,
            "resparsify": self.resparsify,
            "seed": self.seed,
        }

    @property
    def matrix(self) -> np.ndarray:
        """The (d, D) projection, built once from the seed and then reused."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def _build_matrix(self) -> np.ndarray:
        rng = get_rng(
            self.seed,
            stream=f"separation.expand.{self.in_dim}.{self._out_dim}.{self.fan_in}",
        )
        # argsort of one uniform draw per (unit, input) picks fan_in distinct
        # inputs per expanded unit in a single vectorized pass; a stable sort
        # keeps the choice a pure function of the seed.
        keys = rng.random((self._out_dim, self.in_dim))
        picks = np.argsort(keys, axis=1, kind="stable")[:, : self.fan_in]
        signs = rng.integers(0, 2, size=(self._out_dim, self.fan_in)) * 2.0 - 1.0
        matrix = np.zeros((self._out_dim, self.in_dim), dtype=np.float64)
        np.put_along_axis(matrix, picks, signs, axis=1)
        matrix /= np.sqrt(float(self.fan_in))
        return np.ascontiguousarray(matrix.T, dtype=np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.in_dim:
            raise ValueError(f"expected dim {self.in_dim}, got {arr.shape[-1]}")
        rows = arr.reshape(1, -1) if arr.ndim == 1 else arr
        projected = hard_threshold(rows @ self.matrix, self.active)
        coded = unit_rows(projected)
        return coded[0] if arr.ndim == 1 else coded

    def separate(
        self, x_new: np.ndarray, x_collisions: np.ndarray, s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        moved_new, moved_others = push_apart(
            x_new, x_collisions, s, aggregation=self.aggregation
        )
        if not self.resparsify or float(s) == 0.0 or moved_others.shape[0] == 0:
            return moved_new, moved_others
        # The push lands on the union of two k-sparse supports, so re-projecting
        # onto the k-sparse set is what keeps a an invariant of the stored code
        # rather than a property of the first write only.
        return (
            unit_rows(hard_threshold(moved_new, self.active))[0],
            unit_rows(hard_threshold(moved_others, self.active)),
        )


def make_transform(mode: str = "decorrelate", *, dim: int, **kwargs: object) -> SeparationTransform:
    """Build the transform for `mode`, rejecting an unknown one loudly."""
    name = str(mode).strip().lower()
    if name == "decorrelate":
        extra = set(kwargs) - {"aggregation"}
        if extra:
            raise TypeError(f"decorrelate takes no parameter(s) {sorted(extra)}")
        return Decorrelate(dim, **kwargs)  # type: ignore[arg-type]
    if name == "expand_sparsify":
        return ExpandSparsify(dim, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown separation mode {mode!r}, expected one of {MODES}")


def stored_dim(dim: int, *, mode: str = "decorrelate", expansion: int = DEFAULT_EXPANSION) -> int:
    """Width the store must be opened at for `mode`, without building anything.

    The store fixes its dim at construction, so a caller has to know D before it
    can open the file the expanded code will live in.
    """
    name = str(mode).strip().lower()
    if name == "decorrelate":
        return int(dim)
    if name == "expand_sparsify":
        return int(dim) * int(expansion)
    raise ValueError(f"unknown separation mode {mode!r}, expected one of {MODES}")
