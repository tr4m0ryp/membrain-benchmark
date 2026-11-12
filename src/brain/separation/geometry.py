"""The separation geometry: model M-A's transform T_s and the k-sparse code.

One question: given unit vectors, how do you move a colliding pair apart without
leaving the sphere, without dividing by anything near zero, and without
disturbing the directions the pair does not span.

**T_s in closed form.** Write m = (x_i + x_j)/2 and u = (x_i - x_j)/||x_i - x_j||.
For unit x_i, x_j the two are orthogonal (<x_i + x_j, x_i - x_j> = ||x_i||^2 -
||x_j||^2 = 0), and the pair decomposes as x_i = m + h*u, x_j = m - h*u with
h = sqrt((1 - c)/2), ||m||^2 = 1 - h^2, c = <x_i, x_j>. T_s sends h to h + s and
renormalizes, which gives

    c' = (1 - h^2 - (h + s)^2) / (1 - h^2 + (h + s)^2)

Three properties follow and each one is load-bearing:

1. c' = c at s = 0, exactly. This is what makes s = 0 a true no-op rather than
   an approximate one, so the plain-vector control arm is bit-identical.
2. dc'/ds = -4(h + s)(1 - h^2) / (1 - h^2 + (h + s)^2)^2 <= 0, so separation is
   monotone in s and, since c' -> -1 as s grows, it saturates instead of
   diverging. No value of s can produce a non-finite result.
3. T_s is a rotation inside span{m, u} and fixes the orthogonal complement
   pointwise. A cue's component outside the pair plane survives untouched, which
   is why separation costs completion gradually rather than catastrophically
   (the DG-vs-CA3 tradeoff of F6, measured rather than assumed).

Arithmetic is float64 throughout and only the result is cast to float32, so the
stored vector carries a float32-exact unit norm instead of 1 - 6e-8.
"""

from __future__ import annotations

import hashlib

import numpy as np

from brain.config import get_rng
from brain.embed.vectors import l2_normalize

__all__ = [
    "DIRECTION_EPS",
    "AGGREGATIONS",
    "DEFAULT_AGGREGATION",
    "unit_rows",
    "difference_directions",
    "tiebreak_direction",
    "push_apart",
    "hard_threshold",
    "cosine_after_push",
]

#: How m > 1 simultaneous collisions compose. M-A defines T_s on a PAIR and then
#: says "for each colliding distinct trace x_j, apply T_s", which does not settle
#: what happens when several fire on one write, so both readings are implemented
#: and the ambiguity is a flagged spec question rather than a silent choice.
#:
#: * "sequential" -- the literal reading: apply the pairwise T_s once per
#:   neighbour, in the store's deterministic (distance, id) probe order, each
#:   step seeing the newcomer as the previous step left it. Every pair is
#:   provably farther apart at the moment its own push lands, at the cost of the
#:   newcomer accumulating up to m*s of displacement while each neighbour gets s.
#: * "net" -- simultaneous: every neighbour moves s away from the pre-write
#:   newcomer, and the newcomer moves s along the unit NET repulsion direction.
#:   Order-independent and keeps the newcomer's displacement at s regardless of
#:   cluster size, at the cost of the per-pair guarantee.
#:
#: Both reduce to the pairwise T_s exactly at m = 1.
AGGREGATIONS: tuple[str, ...] = ("sequential", "net")
DEFAULT_AGGREGATION = "sequential"

#: Below this a difference vector carries no direction. Chosen well above the
#: float64 noise floor of a difference of two float32 unit vectors (~1e-7 per
#: component) so the tie-break path fires on genuinely identical inputs rather
#: than on rounding.
DIRECTION_EPS = 1e-9


def _as_rows(values: np.ndarray) -> np.ndarray:
    """View a 1-D or 2-D array as float64 rows."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {arr.shape}")
    return arr


def unit_rows(values: np.ndarray) -> np.ndarray:
    """Row-wise unit normalization to float32, raising on a zero-norm row."""
    return l2_normalize(_as_rows(values))


def tiebreak_direction(x: np.ndarray, *, salt: str = "") -> np.ndarray:
    """A unit direction orthogonal to `x`, derived from the content of `x`.

    Reached only when two DISTINCT traces embed to the same vector, where the
    difference direction does not exist. Declining to separate there would leave
    standing exactly the collision this mechanism exists to remove, so the tie is
    broken the way the dentate gyrus breaks it: with fresh random units
    (neurogenesis, F1). Keying the draw on the bytes of `x` makes it reproducible
    across runs and machines and identical for both sides of the pair, so no
    RNG call order can leak into a stored vector.
    """
    vec = np.asarray(x, dtype=np.float64).ravel()
    dim = vec.size
    if dim < 2:
        raise ValueError("cannot build an orthogonal direction in fewer than 2 dims")
    digest = hashlib.blake2b(
        np.asarray(vec, dtype=np.float32).tobytes(), digest_size=8
    ).hexdigest()
    rng = get_rng(stream=f"separation.tiebreak.{salt}.{digest}")
    sq = float(vec @ vec)
    for _ in range(4):
        draw = rng.standard_normal(dim)
        if sq > 0.0:
            draw -= (float(draw @ vec) / sq) * vec
        norm = float(np.linalg.norm(draw))
        if norm > 1e-6:
            return draw / norm
    # Deterministic fallback: the axis carrying the least of x, orthogonalized.
    axis = int(np.argmin(np.abs(vec)))
    basis = np.zeros(dim, dtype=np.float64)
    basis[axis] = 1.0
    if sq > 0.0:
        basis -= (vec[axis] / sq) * vec
    norm = float(np.linalg.norm(basis))
    if norm <= 1e-12:
        raise ValueError("no direction orthogonal to x could be constructed")
    return basis / norm


def difference_directions(
    x_new: np.ndarray, x_collisions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions u_j pointing from each x_j toward x_new, plus a tie mask.

    Returned in float64 and in the row order of `x_collisions`, so the caller's
    id order survives the transform unchanged.
    """
    new = np.asarray(x_new, dtype=np.float64).ravel()
    others = _as_rows(x_collisions)
    if others.shape[1] != new.size:
        raise ValueError(
            f"collision vectors have dim {others.shape[1]}, x_new has {new.size}"
        )
    deltas = new[None, :] - others
    norms = np.linalg.norm(deltas, axis=1)
    tied = norms <= DIRECTION_EPS
    directions = np.empty_like(deltas)
    live = ~tied
    if live.any():
        directions[live] = deltas[live] / norms[live, None]
    # Every tied row is identical to x_new and therefore to every other tied row,
    # so they share one tie-break direction and stay coincident with each other
    # under "net". They cannot stay coincident under "sequential", which moves
    # x_new between rows. Reaching two simultaneous ties at all needs two traces
    # that already embed identically and were written without separation.
    for index in np.flatnonzero(tied):
        directions[index] = tiebreak_direction(new)
    return directions, tied


def _unit64(values: np.ndarray) -> np.ndarray:
    """Normalize a 1-D float64 vector in place-free fashion, refusing zero norm."""
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("separation produced a zero-norm vector, no direction")
    return values / norm


def push_apart(
    x_new: np.ndarray,
    x_collisions: np.ndarray,
    s: float,
    *,
    aggregation: str = DEFAULT_AGGREGATION,
) -> tuple[np.ndarray, np.ndarray]:
    """M-A's T_s over one new vector and its m colliding neighbours.

    Each neighbour is pushed a distance s along -u_j, the direction away from the
    newcomer, exactly as the pairwise T_s prescribes. How the newcomer's m pushes
    compose is the one thing M-A leaves open; see `AGGREGATIONS`. Sibling-to-
    sibling geometry is never re-litigated here: it was settled by the write that
    introduced the sibling, which is what makes this a write-time policy rather
    than a periodic re-layout of the store.

    Under "net", a newcomer sitting at the centroid of its collisions has zero
    net repulsion and does not move; its neighbours still spread away from it,
    which raises the same pairwise distances without inventing a direction.

    Returns (x_new', updates) as float32 unit rows, updates aligned row-for-row
    with `x_collisions`. Intermediate arithmetic stays in float64 so that a long
    sequential chain does not accumulate float32 rounding.
    """
    strength = float(s)
    if not np.isfinite(strength):
        raise ValueError(f"separation strength must be finite, got {s!r}")
    if strength < 0.0:
        raise ValueError(f"separation strength must be >= 0, got {s!r}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}, expected {AGGREGATIONS}")
    new = np.asarray(x_new, dtype=np.float64).ravel()
    others = _as_rows(x_collisions)
    if strength == 0.0 or others.shape[0] == 0:
        # Nothing happened, so nothing is touched: every input is returned
        # unchanged rather than renormalized, which is what keeps the s = 0
        # control arm bit-identical to a plain insert instead of merely close to
        # it (a float32 input round-trips through float64 exactly). Returning an
        # uninitialized buffer here would hand a direct caller NaN.
        return (
            np.ascontiguousarray(new, dtype=np.float32),
            np.ascontiguousarray(others, dtype=np.float32),
        )
    if aggregation == "net":
        directions, _ = difference_directions(new, others)
        net = directions.sum(axis=0)
        net_norm = float(np.linalg.norm(net))
        net = net / net_norm if net_norm > DIRECTION_EPS else net * 0.0
        return unit_rows(new + strength * net)[0], unit_rows(others - strength * directions)
    current = new
    moved = np.empty_like(others)
    for row in range(others.shape[0]):
        direction = difference_directions(current, others[row][None, :])[0][0]
        current = _unit64(current + strength * direction)
        moved[row] = _unit64(others[row] - strength * direction)
    return unit_rows(current)[0], unit_rows(moved)


def cosine_after_push(cosine: float, s: float) -> float:
    """Closed-form c' for a single pair at similarity `cosine` and strength `s`.

    The module docstring derives this. Kept as code because it is the cheapest
    possible check that the vectorized path implements the model it claims to,
    and because a caller sweeping s can read the frontier off it without
    touching the store.
    """
    c = min(1.0, max(-1.0, float(cosine)))
    h = float(np.sqrt(max(0.0, (1.0 - c) / 2.0))) + float(s)
    m_sq = (1.0 + c) / 2.0
    total = m_sq + h * h
    if total <= 0.0:
        return -1.0
    return float((m_sq - h * h) / total)


def hard_threshold(values: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest-magnitude entries per row, zero the rest.

    This is the best k-term L2 approximation of the input, which is the right
    winner-take-all for a signed cosine store: a rate-coded dentate granule cell
    cannot represent a sign, but the stored code can, and discarding a large
    negative coordinate would throw away exactly the evidence that tells two
    near-duplicates apart. Selection is a stable argsort rather than a partition
    so that exactly tied magnitudes resolve by index and the code is a pure
    function of the input.
    """
    arr = np.asarray(values, dtype=np.float64)
    width = arr.shape[-1]
    keep = int(k)
    if keep <= 0:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if keep >= width:
        return arr.copy()
    order = np.argsort(-np.abs(arr), axis=-1, kind="stable")
    take = order[..., :keep]
    out = np.zeros_like(arr)
    np.put_along_axis(out, take, np.take_along_axis(arr, take, axis=-1), axis=-1)
    return out
