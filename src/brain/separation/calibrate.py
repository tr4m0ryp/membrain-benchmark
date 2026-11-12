"""Translating the collision threshold tau between representation scales.

One question: two transforms store the same episodes in geometries with totally
different cosine ranges, so what does one arm's tau mean in the other's space.

M-A defines tau on the STORED representation, and an expanded sparse code is a
different stored representation from a frozen embedding: a pair sitting at 0.75
raw lands near 0.41 once expanded and sparsified. Handing the raw threshold to
the sparse arm therefore leaves the collision detector permanently silent, which
presents as an arm that runs cleanly and separates nothing. Comparing arms at
the same NUMBER instead of the same CRITERION would be a measurement error, not
a modelling choice, which is why this lives in code rather than in a constant.
"""

from __future__ import annotations

import numpy as np

from brain.config import get_rng
from brain.separation.geometry import unit_rows
from brain.separation.transform import SeparationTransform

__all__ = ["calibrate_tau"]


def calibrate_tau(
    transform: SeparationTransform,
    raw_tau: float,
    *,
    n_pairs: int = 256,
    seed: int | None = None,
) -> float:
    """The stored-space threshold corresponding to a raw-space `raw_tau`.

    Synthetic pairs are held at exactly `raw_tau`, pushed through the transform,
    and the resulting cosine is averaged. The answer is a property of the
    TRANSFORM alone: calibrating against the frozen corpus instead would be
    tuning the mechanism on the instrument it is about to be judged by.

    Approximate on purpose. Random unit vectors spread out more than real
    embeddings, which concentrate on a low-dimensional cone, so this is the
    centre of a sweep rather than a tuned value. It is exact for any transform
    whose `encode` is the identity.
    """
    target = min(1.0, max(-1.0, float(raw_tau)))
    pairs = int(n_pairs)
    if pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs!r}")
    rng = get_rng(seed, stream=f"separation.calibrate.{transform.name}.{transform.in_dim}")
    base = unit_rows(rng.standard_normal((pairs, transform.in_dim))).astype(np.float64)
    other = rng.standard_normal((pairs, transform.in_dim))
    # Gram-Schmidt against base, so the partner sits at exactly `target` and the
    # calibration is not confounded by an uncontrolled starting similarity.
    other -= (np.einsum("ij,ij->i", other, base) / np.einsum("ij,ij->i", base, base))[:, None] * base
    other = unit_rows(other).astype(np.float64)
    partner = unit_rows(target * base + np.sqrt(max(0.0, 1.0 - target * target)) * other)
    left = np.asarray(transform.encode(unit_rows(base)), dtype=np.float64)
    right = np.asarray(transform.encode(partner), dtype=np.float64)
    return float(np.einsum("ij,ij->i", left, right).mean())
