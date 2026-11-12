"""Scaffold tests: the package imports, and the seed surface is deterministic."""

from __future__ import annotations

import importlib
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import brain
from brain import config

SUBPACKAGES = (
    "store",
    "embed",
    "separation",
    "baselines",
    "corpus",
    "eval",
    "experiments",
)


def _run_child(code: str, **env_overrides: str) -> str:
    """Run `code` in a clean interpreter and return its stdout, stripped."""
    env = dict(os.environ)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"brain.{name}")
    assert module.__name__ == f"brain.{name}"


def test_root_package_reexports_config() -> None:
    assert brain.seed_everything is config.seed_everything
    assert brain.get_rng is config.get_rng
    assert brain.SEED == config.SEED


def test_seed_everything_is_deterministic() -> None:
    def sample() -> tuple[float, list[float]]:
        return random.random(), np.random.rand(4).tolist()

    assert config.seed_everything(123) == 123
    first = sample()
    assert config.seed_everything(123) == 123
    assert sample() == first

    config.seed_everything(124)
    assert sample() != first


def test_seed_everything_defaults_to_config_seed() -> None:
    assert config.seed_everything() == config.SEED


def test_get_rng_is_deterministic_and_independent() -> None:
    a = config.get_rng(7).random(5)
    b = config.get_rng(7).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, config.get_rng(8).random(5))

    # A local generator must not be disturbed by global reseeding.
    rng = config.get_rng(7)
    config.seed_everything(999)
    assert np.array_equal(rng.random(5), config.get_rng(7).random(5))


def test_get_rng_streams_are_independent_and_stable() -> None:
    corpus = config.get_rng(5, stream="corpus").random(5)
    assert np.array_equal(corpus, config.get_rng(5, stream="corpus").random(5))
    assert not np.array_equal(corpus, config.get_rng(5, stream="separation").random(5))
    assert not np.array_equal(corpus, config.get_rng(5).random(5))
    assert not np.array_equal(corpus, config.get_rng(6, stream="corpus").random(5))

    # Stable across processes: a substream must not depend on PYTHONHASHSEED.
    out = _run_child(
        "import brain.config as c;"
        " print(c.get_rng(5, stream='corpus').random(5).tolist())",
        PYTHONHASHSEED="12345",
    )
    assert out == str(corpus.tolist())


def test_get_rng_defaults_to_config_seed() -> None:
    assert np.array_equal(
        config.get_rng().random(3), config.get_rng(config.SEED).random(3)
    )


def test_paths_are_absolute_and_under_repo_root() -> None:
    for path in (config.DATA_DIR, config.CORPUS_DIR, config.RESULTS_DIR, config.DB_DEFAULT):
        assert path.is_absolute()
        assert config.REPO_ROOT in path.parents
    assert config.CORPUS_DIR == config.DATA_DIR / "corpus"
    assert config.RESULTS_DIR.name == "results"
    assert config.DB_DEFAULT.parent == config.DATA_DIR
    assert (config.REPO_ROOT / "pyproject.toml").is_file()


def test_import_creates_no_directories(tmp_path: Path) -> None:
    out = _run_child(
        "import brain.config as c; print(c.DATA_DIR)", BRAIN_ROOT=str(tmp_path)
    )
    assert out == str(tmp_path / "data")
    assert list(tmp_path.iterdir()) == []


def test_ensure_dir_creates_lazily_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "results" / "nested"
    assert config.ensure_dir(target) == target
    assert target.is_dir()
    assert config.ensure_dir(target) == target


def test_seed_and_smoke_flag_come_from_env() -> None:
    out = _run_child(
        "import brain.config as c; print(c.SEED, c.SMOKE_TEST, c.seed_everything())",
        BRAIN_SEED="7",
        BRAIN_SMOKE_TEST="1",
    )
    assert out == "7 True 7"

    out = _run_child(
        "import brain.config as c; print(c.SEED, c.SMOKE_TEST)",
        BRAIN_SEED="0",
        BRAIN_SMOKE_TEST="0",
    )
    assert out == "0 False"
