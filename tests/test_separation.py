"""The write path: does `SeparationPolicy` do the right thing to the store.

The transform's own math is checked in `test_separation_transform.py`. What is
checked here is everything the store can observe: near-duplicates end up farther
apart, s = 0 is bit-identical to a plain insert, a byte-identical capture dedups
instead of separating, a replay does not drift the store, and the cost is the one
M-A claims. The last test MEASURES the separation-versus-retrievability tradeoff
on real embeddings rather than asserting that vectors moved, because a transform
that separates and loses the cue has bought nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from brain.embed import get_embedder
from brain.separation import SeparationConfigError, SeparationPolicy, build_policy
from brain.store import EpisodicStore, Trace

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


class VectorEmbedder:
    """Fixed text-to-vector map, so the policy is testable without a model."""

    def __init__(self, table: dict[str, np.ndarray], dim: int = DIM) -> None:
        self.table = table
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        return np.asarray(self.table[text], dtype=np.float32)

    def encode(self, texts) -> np.ndarray:
        return np.stack([self.encode_one(t) for t in texts])


@pytest.fixture()
def near_duplicates() -> tuple[dict[str, np.ndarray], list[Trace]]:
    base = unit(1, 0.2, 0.0, 0.05)
    table = {
        "episode one": base,
        "episode two": near(base, 0.06),
        "episode three": near(base, 0.12),
        "something else entirely": unit(0, 0, 1, 0.2),
    }
    traces = [Trace(text=text, id=f"e{i}", ts=float(i)) for i, text in enumerate(table)]
    return table, traces


def make_policy(table: dict[str, np.ndarray], tmp_path: Path, **kwargs) -> SeparationPolicy:
    store = EpisodicStore(tmp_path / f"sep-{len(kwargs)}-{kwargs.get('s', 0)}.db", DIM)
    return SeparationPolicy(store, VectorEmbedder(table), **kwargs)


def test_write_pushes_near_duplicates_apart(near_duplicates, tmp_path: Path) -> None:
    table, traces = near_duplicates
    before = cosine(table["episode one"], table["episode two"])
    policy = make_policy(table, tmp_path, tau=0.8, s=0.3)
    with policy.store as store:
        results = policy.write_many(traces)
        after = cosine(store.get_vector("e0"), store.get_vector("e1"))
        assert before > 0.8, "the fixture must actually collide"
        assert after < before
        assert results[1].separated and results[1].collisions[0][0] == "e0"
        assert results[0].collisions == () and not results[3].separated
        stored = store.get_vectors([t.id for t in traces]).astype(np.float64)
        assert np.isfinite(stored).all()
        assert np.allclose(np.linalg.norm(stored, axis=1), 1.0, atol=1e-6)


def test_zero_strength_stores_the_plain_embedding_bit_for_bit(
    near_duplicates, tmp_path: Path
) -> None:
    table, traces = near_duplicates
    policy = make_policy(table, tmp_path, tau=0.8, s=0.0)
    with policy.store as store:
        results = policy.write_many(traces)
        for trace in traces:
            assert store.get_vector(trace.id).tobytes() == table[trace.text].tobytes()
        assert not any(r.separated for r in results)
        assert not any(r.collisions for r in results)


def test_byte_identical_capture_dedups_and_never_separates(
    near_duplicates, tmp_path: Path
) -> None:
    table, traces = near_duplicates
    policy = make_policy(table, tmp_path, tau=0.8, s=0.3)
    with policy.store as store:
        policy.write_many(traces)
        snapshot = store.get_vectors([t.id for t in traces]).copy()
        repeat = policy.write(Trace(text="episode two", id="e1", ts=1.0))
        assert repeat.deduped and not repeat.inserted and not repeat.separated
        # Same text under a different id is still one capture, not a sibling.
        other_id = policy.write(Trace(text="episode two", id="e99", ts=9.0))
        assert other_id.deduped and other_id.trace_id == "e1"
        assert store.count() == len(traces)
        assert np.array_equal(store.get_vectors([t.id for t in traces]), snapshot)


def test_replaying_the_corpus_does_not_drift_the_store(
    near_duplicates, tmp_path: Path
) -> None:
    """A re-run must be a no-op, otherwise stored geometry depends on run count."""
    table, traces = near_duplicates
    policy = make_policy(table, tmp_path, tau=0.8, s=0.3)
    with policy.store as store:
        policy.write_many(traces)
        first = store.get_vectors([t.id for t in traces]).copy()
        policy.write_many(traces)
        policy.write_many(traces)
        assert np.array_equal(store.get_vectors([t.id for t in traces]), first)
        assert store.count() == len(traces)


def test_empty_and_single_item_stores(near_duplicates, tmp_path: Path) -> None:
    table, traces = near_duplicates
    policy = make_policy(table, tmp_path, tau=0.8, s=0.3)
    with policy.store as store:
        assert policy.write_many([]) == []
        first = policy.write(traces[0])
        assert first.inserted and not first.separated and store.count() == 1
        assert len(policy.write(traces[1]).collisions) == 1


def test_distinct_hint_can_veto_and_defaults_to_separating(
    near_duplicates, tmp_path: Path
) -> None:
    table, traces = near_duplicates
    policy = make_policy(table, tmp_path, tau=0.8, s=0.3)
    with policy.store as store:
        policy.write(traces[0])
        vetoed = policy.write(traces[1], distinct_hint=False)
        assert not vetoed.separated and vetoed.skipped_ids == ("e0",)
        assert store.get_vector("e0").tobytes() == table["episode one"].tobytes()
        seen: list[tuple[str, str]] = []

        def hint(new: Trace, other: Trace) -> bool:
            seen.append((new.id, other.id))
            return True

        allowed = policy.write(traces[2], distinct_hint=hint)
        assert allowed.separated and seen and seen[0][1] in {"e0", "e1"}


def test_probe_saturation_is_reported(tmp_path: Path) -> None:
    base = unit(1, 0.1)
    table = {f"t{i}": near(base, 0.001 * (i + 1)) for i in range(6)}
    traces = [Trace(text=text, id=f"e{i}", ts=float(i)) for i, text in enumerate(table)]
    policy = make_policy(table, tmp_path, tau=0.5, s=0.05, k_probe=2)
    with policy.store as store:
        results = policy.write_many(traces)
        assert store.count() == len(traces)
        assert any(r.probe_saturated for r in results), "truncated probes must be visible"


def test_cost_is_one_probe_per_write_and_zero_at_s_zero(
    near_duplicates, tmp_path: Path
) -> None:
    """M-A's cost claim, made testable: one k-NN probe per write, none at s = 0."""
    table, traces = near_duplicates

    def counted(policy: SeparationPolicy) -> dict[str, int]:
        counts = {name: 0 for name in ("knn", "get_vectors", "update_vectors")}

        def wrap(name: str):
            func = getattr(policy.store, name)

            def inner(*a, **kw):
                counts[name] += 1
                return func(*a, **kw)

            setattr(policy.store, name, inner)

        for name in counts:
            wrap(name)
        return counts

    quiet = make_policy(table, tmp_path, tau=0.8, s=0.0)
    with quiet.store:
        counts = counted(quiet)
        quiet.write_many(traces)
        assert set(counts.values()) == {0}, "s = 0 must touch the store only to insert"

    active = make_policy(table, tmp_path, tau=0.8, s=0.3)
    with active.store:
        counts = counted(active)
        results = active.write_many(traces)
        assert counts["knn"] == len(traces), "exactly one probe per write"
        assert counts["get_vectors"] == counts["update_vectors"] == sum(r.separated for r in results)


def test_store_dim_must_match_the_transform(tmp_path: Path) -> None:
    with EpisodicStore(tmp_path / "wrong.db", DIM) as store:
        with pytest.raises(SeparationConfigError):
            SeparationPolicy(store, VectorEmbedder({}), mode="expand_sparsify")


# -- the tradeoff, measured on real embeddings -------------------------------
SIBLINGS = (
    "auth.py request timeout: raised the connection pool deadline to 30s and the incident closed",
    "auth.py request timeout: the pool was leaking sockets, so the release path got the fix",
    "auth.py request timeout: a stale DNS entry still pointed at the retired replica",
    "auth.py request timeout: the retry budget was exhausted by a hot loop in the client",
)
PARAPHRASED_CUES = (
    "which auth timeout was fixed by giving the pool longer before it gives up",
    "which auth timeout came from sockets never being handed back",
    "which auth timeout pointed at a host that had been decommissioned",
    "which auth timeout was caused by the caller retrying too aggressively",
)


def _sibling_similarity(vectors: np.ndarray) -> float:
    block = np.asarray(vectors, dtype=np.float64)
    return float((block @ block.T)[np.triu_indices(len(block), k=1)].mean())

def _cue_margin(policy: SeparationPolicy, ids: list[str]) -> tuple[float, float]:
    """Cue accuracy against the siblings, and the mean winning margin."""
    stored = policy.store.get_vectors(ids).astype(np.float64)
    cues = np.asarray(policy.encode_queries(list(PARAPHRASED_CUES)), dtype=np.float64)
    scores, index = cues @ stored.T, np.arange(len(ids))
    correct, rivals = scores[index, index], scores.copy()
    rivals[index, index] = -np.inf
    best = rivals.max(axis=1)
    return float((correct > best).mean()), float((correct - best).mean())


@pytest.mark.parametrize("mode", ["decorrelate", "expand_sparsify"])
def test_separation_raises_distance_without_destroying_retrieval(mode: str) -> None:
    """The load-bearing measurement: apart, but still findable from a paraphrase."""
    embedder = get_embedder()
    traces = [Trace(text=t, id=f"s{i}", ts=float(i)) for i, t in enumerate(SIBLINGS)]
    ids = [t.id for t in traces]
    similarities, accuracies, margins = [], [], []
    for s in (0.0, 0.1, 0.3):
        policy = build_policy(embedder, mode=mode, s=s, calibrate=True)
        with policy.store as store:
            policy.write_many(traces)
            stored = store.get_vectors(ids)
            assert np.isfinite(stored).all()
            assert np.allclose(np.linalg.norm(stored.astype(np.float64), axis=1), 1.0, atol=1e-6)
            similarities.append(_sibling_similarity(stored))
            accuracy, margin = _cue_margin(policy, ids)
            accuracies.append(accuracy)
            margins.append(margin)
    assert similarities[1] < similarities[0], "s > 0 must push near-duplicates apart"
    assert similarities[2] < similarities[1], "separation must be monotone in s"
    assert min(accuracies) >= accuracies[0], "separation must not lose a cue it had"
    assert min(margins) > 0.0, "every correct episode must still outrank its siblings"


def test_same_seed_same_stored_vectors_across_processes(tmp_path: Path) -> None:
    script = (
        "import numpy as np\n"
        "from brain.separation import build_policy\n"
        "from brain.embed import get_embedder\n"
        "from brain.store import Trace\n"
        f"texts = {list(SIBLINGS)!r}\n"
        "traces = [Trace(text=t, id=f's{i}', ts=float(i)) for i, t in enumerate(texts)]\n"
        "p = build_policy(get_embedder(), mode='expand_sparsify', s=0.3, calibrate=True)\n"
        "p.write_many(traces)\n"
        "v = p.store.get_vectors([t.id for t in traces])\n"
        "print(np.asarray(v).tobytes().hex()[:120])\n"
    )
    outputs = []
    for hash_seed in ("0", "1", "2"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONHASHSEED=hash_seed),
            capture_output=True, text=True, check=True,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1] == outputs[2] and len(outputs[0]) == 120
