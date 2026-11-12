"""Embedder contract: shape, dtype, unit norm, determinism, batching, cache.

The model is loaded once per session (module-scoped fixtures) because a load
costs about 0.2s and every test here needs the same frozen weights.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from brain.embed import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    Embedder,
    VoyageEmbedder,
    get_embedder,
    l2_normalize,
)

TEXTS = [
    "the migration lock was held by a stale worker",
    "deploy to staging failed at three in the morning",
    "short",
    "the migration lock was held by a stale worker",
]

# Batch composition shifts the last ulps of float32 transformer inference, so
# cross-batch comparisons use a tolerance instead of byte equality. Measured
# worst case on this machine is about 1e-7.
ULP = 1e-6


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(cache_size=0)


@pytest.fixture(scope="module")
def cached_embedder() -> Embedder:
    return Embedder(cache_size=64)


def test_import_does_not_load_the_model() -> None:
    fresh = Embedder()
    assert fresh.is_loaded is False
    out = subprocess.run(
        [sys.executable, "-c", "import brain.embed, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def test_shape_dtype_and_dim(embedder: Embedder) -> None:
    vectors = embedder.encode(TEXTS)
    assert vectors.shape == (len(TEXTS), embedder.dim)
    assert embedder.dim == 384
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert embedder.model_name == DEFAULT_MODEL
    assert embedder.max_seq_length == 256


def test_smoke_flag_does_not_change_the_batch_size() -> None:
    """A smaller batch would shift the float32 rounding, so SMOKE_TEST must not."""
    code = "from brain.embed import DEFAULT_BATCH_SIZE; print(DEFAULT_BATCH_SIZE)"
    env = dict(os.environ)
    env["BRAIN_SMOKE_TEST"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == str(DEFAULT_BATCH_SIZE) == "32"


def test_vectors_are_unit_norm(embedder: Embedder) -> None:
    vectors = embedder.encode(TEXTS)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # Cosine must equal dot product for the store to use dot products.
    assert np.isclose(float(vectors[0] @ vectors[3]), 1.0, atol=1e-6)


def test_repeat_encode_is_bit_identical(embedder: Embedder) -> None:
    first = embedder.encode(TEXTS)
    second = embedder.encode(TEXTS)
    assert first.tobytes() == second.tobytes()


def test_duplicate_rows_are_bit_identical(embedder: Embedder) -> None:
    vectors = embedder.encode(TEXTS)
    assert vectors[0].tobytes() == vectors[3].tobytes()


def test_single_and_batch_paths_agree(embedder: Embedder) -> None:
    batched = embedder.encode(TEXTS)
    for index, text in enumerate(TEXTS):
        solo = embedder.encode(text)
        assert solo.shape == (1, embedder.dim)
        assert np.allclose(solo[0], batched[index], atol=ULP)
        assert float(solo[0] @ batched[index]) > 1.0 - ULP
    one = embedder.encode_one(TEXTS[0])
    assert one.shape == (embedder.dim,)
    assert np.allclose(one, batched[0], atol=ULP)


def test_batch_size_does_not_change_results(embedder: Embedder) -> None:
    small = embedder.encode(TEXTS, batch_size=1)
    large = embedder.encode(TEXTS, batch_size=64)
    assert np.allclose(small, large, atol=ULP)


def test_encode_is_one_model_call_not_one_per_text(
    embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    inference: list[bool] = []
    import torch

    original = embedder.model.encode

    def counting_encode(texts, **kwargs):
        calls.append(len(texts))
        inference.append(torch.is_inference_mode_enabled())
        return original(texts, **kwargs)

    monkeypatch.setattr(embedder.model, "encode", counting_encode)
    embedder.encode(TEXTS, batch_size=32)
    assert calls == [len(TEXTS)]
    # No autograd graph is built during encoding.
    assert inference == [True]


def test_cache_does_not_change_results(cached_embedder: Embedder) -> None:
    cached_embedder.clear_cache()
    warm = cached_embedder.encode(TEXTS)
    again = cached_embedder.encode(TEXTS)
    assert warm.tobytes() == again.tobytes()

    uncached = Embedder(cache_size=0).encode(TEXTS)
    assert np.allclose(warm, uncached, atol=ULP)
    assert np.allclose(np.einsum("ij,ij->i", warm, uncached), 1.0, atol=ULP)


def test_cache_avoids_recomputation(
    cached_embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached_embedder.clear_cache()
    cached_embedder.encode(TEXTS)
    original = cached_embedder.model.encode

    def fail_encode(texts, **kwargs):  # pragma: no cover - fires only on a miss
        raise AssertionError(f"recomputed cached texts: {texts}")

    monkeypatch.setattr(cached_embedder.model, "encode", fail_encode)
    cached_embedder.encode(TEXTS)
    monkeypatch.setattr(cached_embedder.model, "encode", original)


def test_cache_eviction_is_safe() -> None:
    tiny = Embedder(cache_size=1)
    vectors = tiny.encode(TEXTS)
    assert vectors.shape == (len(TEXTS), tiny.dim)
    assert len(tiny._cache) == 1
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_returned_array_is_independent_of_cache(cached_embedder: Embedder) -> None:
    cached_embedder.clear_cache()
    first = cached_embedder.encode(TEXTS)
    first[:] = 0.0
    second = cached_embedder.encode(TEXTS)
    assert np.allclose(np.linalg.norm(second, axis=1), 1.0, atol=1e-6)


def test_empty_and_type_errors(embedder: Embedder) -> None:
    empty = embedder.encode([])
    assert empty.shape == (0, embedder.dim)
    assert empty.dtype == np.float32
    with pytest.raises(TypeError):
        embedder.encode(["ok", 3])
    with pytest.raises(TypeError):
        embedder.encode_one(["not a string"])


def test_empty_string_is_encoded_not_dropped(embedder: Embedder) -> None:
    vectors = embedder.encode(["", "   "])
    assert vectors.shape == (2, embedder.dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_normalize_false_returns_finite_raw_vectors() -> None:
    raw = Embedder(normalize=False, cache_size=0).encode(TEXTS[:2])
    assert raw.dtype == np.float32
    assert np.isfinite(raw).all()


def test_encoding_is_reproducible_across_processes(embedder: Embedder) -> None:
    """The hard determinism requirement: same bytes in a fresh interpreter."""
    expected = embedder.encode(TEXTS).tobytes().hex()
    code = (
        "from brain.embed import Embedder;"
        f" print(Embedder(cache_size=0).encode({TEXTS!r}).tobytes().hex())"
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "31337"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == expected


def test_get_embedder_shares_one_instance() -> None:
    first = get_embedder()
    assert first is get_embedder()
    assert isinstance(first, Embedder)
    with pytest.raises(ValueError):
        get_embedder(backend="nonsense")


def test_package_reexports_resolve() -> None:
    import brain.embed as package

    for name in package.__all__:
        assert getattr(package, name) is not None
    with pytest.raises(AttributeError):
        package.not_a_real_name


def test_voyage_batches_and_normalizes_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    voyage = VoyageEmbedder()
    sent: list[int] = []

    class FakeResponse:
        def __init__(self, rows: list[list[float]]) -> None:
            self.embeddings = rows

    class FakeClient:
        def embed(self, texts, model, input_type=None):  # noqa: ANN001
            sent.append(len(texts))
            return FakeResponse([[float(len(t)), 2.0, 0.0, 0.0] for t in texts])

    voyage._client = FakeClient()
    vectors = voyage.encode(["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2)
    assert sent == [2, 2, 1]
    assert vectors.shape == (5, 4)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert voyage.dim == 4


def test_voyage_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageEmbedder()
    monkeypatch.setenv("BRAIN_EMBEDDER", "voyage")
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        get_embedder()


def test_l2_normalize_rejects_degenerate_input() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="zero-norm"):
        l2_normalize(matrix)
    with pytest.raises(ValueError, match="non-finite"):
        l2_normalize(np.array([[np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        l2_normalize(np.ones(4, dtype=np.float32))

    unit = l2_normalize(matrix[:1])
    assert unit.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(unit[0])), 1.0, atol=1e-7)
