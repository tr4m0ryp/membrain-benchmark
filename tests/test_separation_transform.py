"""The transform math: does the code implement model M-A, and does it stay finite.

The vectorized push is compared against the closed form derived in `geometry`,
so a silent divergence from M-A fails here rather than surfacing as a puzzling
number in the gate experiment. The degenerate cases get the same treatment as
the normal ones: identical vectors, a zero net repulsion, and a strength large
enough to send a pair antipodal must all produce finite unit vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.separation import (
    Decorrelate,
    ExpandSparsify,
    calibrate_tau,
    cosine_after_push,
    hard_threshold,
    make_transform,
    push_apart,
    stored_dim,
    tiebreak_direction,
)

DIM = 16


def unit(*values: float) -> np.ndarray:
    """A unit vector padded out to DIM, so cosines are readable by hand."""
    vec = np.zeros(DIM, dtype=np.float32)
    vec[: len(values)] = np.asarray(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def near(base: np.ndarray, tilt: float, axis: int = 2) -> np.ndarray:
    """A unit vector at a controlled small angle from `base`."""
    other = np.zeros(DIM, dtype=np.float32)
    other[axis] = 1.0
    mixed = (1.0 - tilt) * base.astype(np.float64) + tilt * other
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


# -- the transform: does it match model M-A ---------------------------------

def test_push_matches_the_closed_form_of_m_a() -> None:
    """The vectorized push must reproduce the derived c'(c, s) exactly."""
    base = unit(1, 0.2, 0.05, -0.1)
    for tilt in (0.02, 0.1, 0.3, 0.6):
        other = near(base, tilt)
        before = cosine(base, other)
        for s in (0.0, 0.05, 0.25, 1.0, 4.0):
            moved_new, updates = push_apart(base, other[None, :], s)
            after = cosine(moved_new, updates[0])
            assert after == pytest.approx(cosine_after_push(before, s), abs=1e-6)


def test_separation_is_monotone_in_s_and_saturates() -> None:
    base = unit(1, 0.2)
    other = near(base, 0.05)
    before = cosine(base, other)
    sims = []
    for s in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 8.0):
        moved_new, updates = push_apart(base, other[None, :], s)
        sims.append(cosine(moved_new, updates[0]))
    assert sims[0] == pytest.approx(before, abs=1e-6)
    assert all(later < earlier for earlier, later in zip(sims, sims[1:]))
    assert sims[-1] > -1.0000001 and np.isfinite(sims).all()


def test_push_is_a_rotation_in_the_pair_plane() -> None:
    """Everything orthogonal to span{m, u} is untouched: that is what a partial
    cue outside the plane can still complete from."""
    base, other = unit(1, 0.3), near(unit(1, 0.3), 0.1)
    moved_new, updates = push_apart(base, other[None, :], 0.4)
    plane = np.stack([base + other, base - other]).astype(np.float64)
    basis = np.linalg.qr(plane.T)[0]
    for before, after in ((base, moved_new), (other, updates[0])):
        residual_before = before - basis @ (basis.T @ before.astype(np.float64))
        residual_after = after - basis @ (basis.T @ after.astype(np.float64))
        assert np.allclose(residual_before, residual_after, atol=1e-6)


def test_zero_strength_is_bit_identical() -> None:
    """s = 0 returns the inputs verbatim, including the untouched neighbours.

    Regression: returning an uninitialized buffer for the neighbours here read
    as NaN to any direct caller, and the policy's own s = 0 short circuit hid it.
    """
    base, other = unit(1, 0.3), near(unit(1, 0.3), 0.1)
    moved_new, updates = push_apart(base, other[None, :], 0.0)
    assert moved_new.tobytes() == base.tobytes()
    assert updates.shape == (1, DIM)
    assert updates[0].tobytes() == other.tobytes()
    assert np.isfinite(updates).all()


def test_multi_collision_reduces_to_the_pair_at_m_equals_one() -> None:
    base, other = unit(1, 0.2), near(unit(1, 0.2), 0.08)
    single_new, single_updates = push_apart(base, other[None, :], 0.3)
    stacked_new, stacked_updates = push_apart(base, np.stack([other]), 0.3)
    assert np.allclose(single_new, stacked_new, atol=1e-7)
    assert np.allclose(single_updates, stacked_updates, atol=1e-7)


def test_centred_newcomer_does_not_move_under_net_aggregation() -> None:
    """Net repulsion of zero is a defined outcome, not a divide-by-zero."""
    base = unit(1, 0)
    left, right = near(base, 0.1, axis=2), near(base, -0.1, axis=2)
    moved_new, updates = push_apart(base, np.stack([left, right]), 0.3, aggregation="net")
    assert np.isfinite(moved_new).all() and np.isfinite(updates).all()
    assert cosine(moved_new, base) == pytest.approx(1.0, abs=1e-6)
    assert cosine(updates[0], moved_new) < cosine(left, base)
    assert cosine(updates[1], moved_new) < cosine(right, base)


def test_both_aggregations_reduce_to_the_pair_and_separate_every_neighbour() -> None:
    """The literal per-neighbour reading and the simultaneous one must agree at
    m = 1 and must both raise every colliding pair's distance."""
    base = unit(1, 0.2)
    single = near(base, 0.08)
    for aggregation in ("sequential", "net"):
        one_new, one_updates = push_apart(base, single[None, :], 0.3, aggregation=aggregation)
        assert cosine(one_new, one_updates[0]) == pytest.approx(
            cosine_after_push(cosine(base, single), 0.3), abs=1e-6
        )
        others = np.stack([near(base, t, axis=a) for t, a in ((0.06, 2), (0.09, 3), (0.12, 4))])
        moved_new, updates = push_apart(base, others, 0.3, aggregation=aggregation)
        assert np.isfinite(moved_new).all() and np.isfinite(updates).all()
        for row in range(len(others)):
            assert cosine(moved_new, updates[row]) < cosine(base, others[row])
    with pytest.raises(ValueError):
        push_apart(base, single[None, :], 0.3, aggregation="teleport")


# -- degenerate inputs -------------------------------------------------------

def test_identical_vectors_are_separated_not_left_colliding() -> None:
    base = unit(1, 0.4, -0.2)
    moved_new, updates = push_apart(base, base[None, :].copy(), 0.3)
    assert np.isfinite(moved_new).all() and np.isfinite(updates).all()
    assert cosine(moved_new, updates[0]) < 0.99
    assert cosine(moved_new, updates[0]) == pytest.approx(cosine_after_push(1.0, 0.3), abs=1e-6)


def test_tiebreak_direction_is_orthogonal_and_reproducible() -> None:
    base = unit(1, 0.4, -0.2)
    first, second = tiebreak_direction(base), tiebreak_direction(base)
    assert np.allclose(first, second)
    assert abs(float(first @ base.astype(np.float64))) < 1e-9
    assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(first, tiebreak_direction(unit(0, 1, 0.3)))


def test_no_collisions_and_bad_strength() -> None:
    base = unit(1, 0.2)
    moved_new, updates = push_apart(base, np.zeros((0, DIM), dtype=np.float32), 0.5)
    assert moved_new.tobytes() == base.tobytes() and updates.shape == (0, DIM)
    with pytest.raises(ValueError):
        push_apart(base, base[None, :], -0.1)
    with pytest.raises(ValueError):
        push_apart(base, base[None, :], float("nan"))


def test_outputs_are_unit_norm_over_a_random_sweep() -> None:
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((64, DIM))
    vectors = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
    for s in (0.01, 0.25, 1.0, 5.0):
        moved_new, updates = push_apart(vectors[0], vectors[1:], s)
        stacked = np.vstack([moved_new[None, :], updates]).astype(np.float64)
        assert np.isfinite(stacked).all()
        assert np.allclose(np.linalg.norm(stacked, axis=1), 1.0, atol=1e-6)


# -- the sparse arm ----------------------------------------------------------

def test_hard_threshold_keeps_the_largest_magnitudes() -> None:
    values = np.asarray([[0.1, -0.9, 0.4, 0.0, -0.5]], dtype=np.float64)
    kept = hard_threshold(values, 2)
    assert np.allclose(kept, [[0.0, -0.9, 0.0, 0.0, -0.5]])
    assert np.allclose(hard_threshold(values, 99), values)
    with pytest.raises(ValueError):
        hard_threshold(values, 0)


def test_expansion_is_deterministic_and_sparse() -> None:
    left = ExpandSparsify(DIM, expansion=8, sparseness=0.1, seed=3)
    right = ExpandSparsify(DIM, expansion=8, sparseness=0.1, seed=3)
    assert left.out_dim == DIM * 8 == stored_dim(DIM, mode="expand_sparsify", expansion=8)
    assert np.array_equal(left.matrix, right.matrix)
    assert (np.count_nonzero(left.matrix, axis=0) == left.fan_in).all()
    code = left.encode(unit(1, 0.3, -0.2))
    assert np.count_nonzero(code) == left.active
    assert float(np.linalg.norm(code.astype(np.float64))) == pytest.approx(1.0, abs=1e-6)
    assert np.array_equal(code, right.encode(unit(1, 0.3, -0.2)))


def test_expansion_separates_and_keeps_the_code_sparse() -> None:
    transform = ExpandSparsify(DIM, expansion=8, sparseness=0.1, seed=3)
    base, other = unit(1, 0.2), near(unit(1, 0.2), 0.08)
    coded_new, coded_other = transform.encode(base), transform.encode(other)
    before = cosine(coded_new, coded_other)
    moved_new, updates = transform.separate(coded_new, coded_other[None, :], 0.3)
    assert cosine(moved_new, updates[0]) < before
    assert np.count_nonzero(moved_new) == transform.active
    assert np.count_nonzero(updates[0]) == transform.active
    assert np.isfinite(moved_new).all() and np.isfinite(updates).all()


def test_calibrate_tau_maps_scales_and_is_identity_for_decorrelate() -> None:
    assert calibrate_tau(Decorrelate(DIM), 0.75) == pytest.approx(0.75, abs=1e-6)
    sparse = ExpandSparsify(DIM, expansion=8, sparseness=0.1, seed=3)
    mapped = calibrate_tau(sparse, 0.75)
    assert 0.0 < mapped < 0.75
    assert calibrate_tau(sparse, 0.9) > mapped


def test_make_transform_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        make_transform("magic", dim=DIM)
    with pytest.raises(TypeError):
        make_transform("decorrelate", dim=DIM, expansion=4)
    with pytest.raises(ValueError):
        ExpandSparsify(DIM, sparseness=0.0)
