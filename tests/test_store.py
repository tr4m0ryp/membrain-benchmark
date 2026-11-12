"""Store tests: round-trip, byte-identical dedup, k-NN, bi-temporal validity.

Determinism is checked as hard as behaviour: a retrieval that reorders between
runs would silently poison every experiment downstream.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from brain.store import (
    DimensionMismatchError, DuplicateTraceIdError, EpisodicStore,
    InvalidVectorError, StoreError, Trace, sha256_text,
)

DIM = 8


def unit(*values: float) -> np.ndarray:
    """A unit vector padded out to DIM, so cosine distances are readable."""
    vec = np.zeros(DIM, dtype=np.float32)
    vec[: len(values)] = np.asarray(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture()
def store(tmp_path: Path):
    with EpisodicStore(tmp_path / "brain.db", DIM) as episodic:
        yield episodic

def make(text: str, **kwargs) -> Trace:
    return Trace(text=text, **kwargs)

def test_trace_derives_id_sha_and_valid_at() -> None:
    trace = make("fixed the timeout in auth.py", ts=12.5)
    assert trace.content_sha == sha256_text("fixed the timeout in auth.py")
    assert trace.id == f"t_{trace.content_sha[:16]}"
    assert trace.valid_at == 12.5
    assert trace.is_valid
    assert make("fixed the timeout in auth.py").id == trace.id  # stable id


def test_insert_get_round_trip(store: EpisodicStore) -> None:
    vec = np.asarray([0.13, -0.9, 0.4, 0.0, 0.5, -0.25, 0.1, 0.02], dtype=np.float32)
    trace = make("retry loop added to fetch_user", id="e1", session="s1", ts=100.0,
                 project="brain", salience=0.75, cluster_id="c3")
    assert store.add(trace, vec) == "e1"
    got = store.get("e1")
    assert got is not None
    assert (got.text, got.session, got.ts, got.project, got.salience) == (
        trace.text, "s1", 100.0, "brain", 0.75)
    assert (got.valid_at, got.invalid_at, got.cluster_id) == (100.0, None, "c3")
    assert got.content_sha == trace.content_sha
    # Stored bit for bit: Task 005 asserts s=0 leaves vectors untouched.
    assert store.get_vector("e1").tobytes() == vec.tobytes()
    assert store.count() == 1
    assert store.get("missing") is None
    assert store.get_vector("missing") is None


# -- dedup: byte-identical only --------------------------------------------

def test_byte_identical_dedup_returns_existing_id(store: EpisodicStore) -> None:
    first = store.add(make("same capture", id="e1"), unit(1, 0, 0))
    second = store.add(make("same capture", id="e2"), unit(0, 1, 0))
    assert first == "e1"
    assert second == "e1", "a repeat must resolve to the stored trace"
    assert store.count() == 1
    assert store.get("e2") is None
    assert np.allclose(store.get_vector("e1"), unit(1, 0, 0))  # winner kept


def test_similar_but_distinct_traces_are_never_merged(store: EpisodicStore) -> None:
    text = "timeout in auth.py fixed by raising the deadline"
    a = store.add(make(text), unit(1, 0))
    b = store.add(make(text + "."), unit(1, 0.001))
    assert a != b
    assert store.count() == 2


def test_duplicate_id_with_different_text_raises(store: EpisodicStore) -> None:
    store.add(make("first", id="e1"), unit(1, 0))
    with pytest.raises(DuplicateTraceIdError):
        store.add(make("second", id="e1"), unit(0, 1))
    assert store.count() == 1


def test_add_many_batches_and_dedups(store: EpisodicStore) -> None:
    traces = [make("a", id="e1"), make("b", id="e2"), make("a", id="e3")]
    vectors = np.stack([unit(1, 0), unit(0, 1), unit(0, 0, 1)])
    assert store.add_many(traces, vectors) == ["e1", "e2", "e1"]
    assert store.count() == 2
    assert store.add_many([], np.zeros((0, DIM))) == []


# -- k-NN ------------------------------------------------------------------

def _plant(store: EpisodicStore) -> None:
    store.add(make("near", id="near", project="p1"), unit(1, 0.05))
    store.add(make("mid", id="mid", project="p1"), unit(1, 1))
    store.add(make("far", id="far", project="p2"), unit(0, 1))


def test_knn_returns_nearest_planted_vector(store: EpisodicStore) -> None:
    _plant(store)
    hits = store.knn(unit(1, 0), 3)
    assert [tid for tid, _ in hits] == ["near", "mid", "far"]
    # Cosine distance, so 1 - distance is the similarity numpy computes.
    sims = [1.0 - dist for _, dist in hits]
    assert sims == sorted(sims, reverse=True)
    assert sims[0] == pytest.approx(float(unit(1, 0) @ unit(1, 0.05)), abs=1e-6)
    assert sims[2] == pytest.approx(0.0, abs=1e-6)


def test_knn_filters_by_project_before_the_search(store: EpisodicStore) -> None:
    _plant(store)
    # The global top 2 for this probe is far (p2) then mid (p1). Getting two
    # p1 rows back proves the filter runs before the top-k cut, not after it.
    assert [tid for tid, _ in store.knn(unit(0, 1), 2, project="p1")] == ["mid", "near"]
    assert [tid for tid, _ in store.knn(unit(1, 0), 5, project="p2")] == ["far"]
    assert store.knn(unit(1, 0), 5, project="nope") == []


def test_knn_degenerate_inputs(store: EpisodicStore) -> None:
    # Empty store and non-positive k are the paths Task 005 hits first.
    assert store.knn(unit(1, 0), 5) == []
    assert store.all_traces() == []
    _plant(store)
    assert len(store.knn(unit(1, 0), 2)) == 2
    assert len(store.knn(unit(1, 0), 99)) == 3
    assert store.knn(unit(1, 0), 0) == []
    assert store.knn(unit(1, 0), -1) == []


def test_knn_ties_break_by_id(store: EpisodicStore) -> None:
    # Inserted in reverse id order and equidistant from the probe, so only the
    # explicit tie-break can produce ascending ids.
    store.add(make("second", id="b"), unit(1, 1, 0))
    store.add(make("first", id="a"), unit(1, 0, 1))
    hits = store.knn(unit(1, 0, 0), 2)
    assert hits[0][1] == pytest.approx(hits[1][1])
    assert [tid for tid, _ in hits] == ["a", "b"]


def test_knn_sees_updated_vectors(store: EpisodicStore) -> None:
    _plant(store)
    store.update_vector("far", unit(1, 0.01))
    assert store.knn(unit(1, 0), 1)[0][0] == "far"
    store.update_vectors([("far", unit(0, 1)), ("mid", unit(1, 0.02))])
    assert store.knn(unit(1, 0), 1)[0][0] == "mid"
    with pytest.raises(KeyError):
        store.update_vector("ghost", unit(1, 0))
    with pytest.raises(KeyError):
        store.update_vectors([("ghost", unit(1, 0))])


def test_get_vectors_keeps_requested_order(store: EpisodicStore) -> None:
    _plant(store)
    block = store.get_vectors(["mid", "near"])
    assert block.shape == (2, DIM)
    assert np.allclose(block, np.stack([unit(1, 1), unit(1, 0.05)]))
    assert store.get_vectors([]).shape == (0, DIM)
    with pytest.raises(KeyError):
        store.get_vectors(["near", "ghost"])


# -- validation ------------------------------------------------------------

def test_dimension_mismatch_raises(store: EpisodicStore) -> None:
    with pytest.raises(DimensionMismatchError):
        store.add(make("short"), np.ones(DIM - 1, dtype=np.float32))
    with pytest.raises(DimensionMismatchError):
        store.knn(np.ones(DIM + 3, dtype=np.float32), 1)
    store.add(make("ok", id="e1"), unit(1, 0))
    with pytest.raises(DimensionMismatchError):
        store.update_vector("e1", np.ones(2, dtype=np.float32))
    with pytest.raises(DimensionMismatchError):
        store.add_many([make("x")], np.ones((1, DIM + 1), dtype=np.float32))
    assert store.count() == 1


def test_degenerate_vectors_raise_not_store_nan(store: EpisodicStore) -> None:
    for bad in (np.full(DIM, np.nan, dtype=np.float32),
                np.full(DIM, np.inf, dtype=np.float32),
                np.zeros(DIM, dtype=np.float32),
                np.full(DIM, 1e-30)):  # squared norm underflows float32
        with pytest.raises(InvalidVectorError):
            store.add(make(f"bad {bad[0]}"), bad)
    with pytest.raises(InvalidVectorError):
        store.add(make("2d"), np.ones((2, DIM), dtype=np.float32))
    with pytest.raises(InvalidVectorError):
        store.add_many([make("a"), make("b")], np.ones((1, DIM), dtype=np.float32))
    assert store.count() == 0
    for bad_dim in (0, -4, 3.5, True):
        with pytest.raises(ValueError):
            EpisodicStore(":memory:", bad_dim)  # type: ignore[arg-type]


# -- bi-temporal validity --------------------------------------------------

def test_invalidate_hides_but_never_deletes(store: EpisodicStore) -> None:
    _plant(store)
    store.invalidate("near", 500.0)
    # Still k rows after the nearest one is hidden: validity is a pre-filter.
    assert [tid for tid, _ in store.knn(unit(1, 0), 2)] == ["mid", "far"]
    assert [tid for tid, _ in store.knn(unit(1, 0), 3, valid_only=False)][0] == "near"
    assert [t.id for t in store.all_traces()] == ["far", "mid"]
    assert [t.id for t in store.all_traces(valid_only=False)] == ["far", "mid", "near"]
    kept = store.get("near")
    assert kept is not None and kept.invalid_at == 500.0 and not kept.is_valid
    assert store.count() == 3
    assert store.count(valid_only=True) == 2
    with pytest.raises(KeyError):
        store.invalidate("ghost", 1.0)


def test_as_of_reads_the_store_at_a_past_instant(store: EpisodicStore) -> None:
    store.add(make("old", id="old", ts=10.0), unit(1, 0))
    store.add(make("new", id="new", ts=200.0), unit(1, 0.02))
    store.invalidate("old", 100.0)
    assert [tid for tid, _ in store.knn(unit(1, 0), 5, as_of=50.0)] == ["old"]
    assert [tid for tid, _ in store.knn(unit(1, 0), 5, as_of=150.0)] == []
    assert [tid for tid, _ in store.knn(unit(1, 0), 5, as_of=300.0)] == ["new"]
    assert [t.id for t in store.all_traces(as_of=50.0)] == ["old"]
    assert store.count(as_of=300.0) == 1


def test_all_traces_orders_by_id_and_filters(store: EpisodicStore) -> None:
    _plant(store)
    assert [t.id for t in store.all_traces()] == ["far", "mid", "near"]
    assert [t.id for t in store.all_traces(project="p1")] == ["mid", "near"]
    assert [t.id for t in store.all_traces(limit=2)] == ["far", "mid"]
    assert store.count(project="p1") == 2
    assert len(store) == 3


# -- persistence and determinism -------------------------------------------

def test_store_reopens_and_refuses_a_different_dim(tmp_path: Path) -> None:
    path = tmp_path / "brain.db"
    with EpisodicStore(path, DIM) as first:
        first.add(make("persisted", id="e1"), unit(1, 0))
    with EpisodicStore(path, DIM) as second:
        assert second.count() == 1
        assert second.knn(unit(1, 0), 1)[0][0] == "e1"
        second.close()  # close is idempotent
    with pytest.raises(StoreError):
        EpisodicStore(path, DIM + 1)


DETERMINISM_PROBE = """
import numpy as np
from brain.store import EpisodicStore, Trace
from brain.config import get_rng

rng = get_rng(11, stream="test-store")
vectors = rng.normal(size=(40, 16))
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
traces = [Trace(text="trace %d" % i, id="e%02d" % i) for i in range(40)]
order = list(range(40))
if {reverse}:
    order.reverse()
store = EpisodicStore({path!r}, 16)
store.add_many([traces[i] for i in order], vectors[order])
probe = get_rng(11, stream="test-store-probe").normal(size=16)
probe /= np.linalg.norm(probe)
print(store.knn(probe, 10))
store.close()
"""


def test_knn_ordering_is_stable_across_processes_and_write_order(
    tmp_path: Path,
) -> None:
    """A fresh store built in another interpreter must rank identically.

    The third arm writes the same traces in reverse order: a ranking that
    depended on insert order would make every reported number unreproducible.
    """
    arms = (("0", False), ("12345", False), ("999", True))
    outputs = []
    for i, (hash_seed, reverse) in enumerate(arms):
        script = DETERMINISM_PROBE.format(
            path=str(tmp_path / f"det{i}.db"), reverse=reverse
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONHASHSEED=hash_seed),
            capture_output=True, text=True, check=True,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0].count("(") == 10
