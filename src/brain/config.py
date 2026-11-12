"""Global configuration: repo paths, the seed, and the randomness entry points.

Determinism is a hard project requirement, so every source of randomness in the
package must route through this module: either `seed_everything` (global state,
for third-party libraries that only expose global seeding) or `get_rng` (a local
generator, preferred for our own code because it cannot be disturbed by an
unrelated call elsewhere in the process).

Nothing here touches the filesystem at import time. Directories are created only
when a caller asks for one via `ensure_dir`.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "CORPUS_DIR",
    "RESULTS_DIR",
    "results_dir",
    "DB_DEFAULT",
    "SEED",
    "SMOKE_TEST",
    "seed_everything",
    "get_rng",
    "ensure_dir",
]

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, tolerant of the usual spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _find_repo_root() -> Path:
    """Locate the repository root.

    `BRAIN_ROOT` wins when set (used by tests and by any out-of-tree run).
    Otherwise walk up from this file to the first directory holding
    `pyproject.toml`, which is correct for both an editable install and a plain
    source checkout. The final fallback keeps the src-layout assumption
    (`<root>/src/brain/config.py`).
    """
    override = os.environ.get("BRAIN_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


REPO_ROOT: Path = _find_repo_root()
DATA_DIR: Path = REPO_ROOT / "data"
CORPUS_DIR: Path = DATA_DIR / "corpus"
RESULTS_DIR: Path = REPO_ROOT / "results"
DB_DEFAULT: Path = DATA_DIR / "brain.db"


def results_dir(experiment: str) -> Path:
    """Where one experiment family writes its artifacts.

    Results are grouped by the experiment that produced them rather than pooled
    into one directory. Pooled, a timestamped filename is the only thing saying
    which experiment a run belongs to, and that stops being true the moment two
    experiments share a stem. The path is returned, not created; callers that
    write go through `ensure_dir`.
    """
    return RESULTS_DIR / experiment

#: Global seed. Every task derives its randomness from this value.
SEED: int = int(os.environ.get("BRAIN_SEED", "0"))

#: Tiny-N fast path so the whole pipeline can be validated on CPU in seconds.
SMOKE_TEST: bool = _env_flag("BRAIN_SMOKE_TEST")


def seed_everything(seed: int | None = None) -> int:
    """Seed the global RNGs and return the seed actually used.

    Seeds `random`, numpy's legacy global RNG, and torch if it is importable.
    Torch is imported lazily and optionally so that importing `brain` stays fast
    and works before the embedder dependencies are installed.
    """
    used = SEED if seed is None else int(seed)
    random.seed(used)
    np.random.seed(used)
    _seed_torch(used)
    return used


def _seed_torch(seed: int) -> None:
    try:
        import torch
    except ImportError:
        return
    # CPU-only build for this launch, so manual_seed covers every generator.
    torch.manual_seed(seed)


def get_rng(seed: int | None = None, *, stream: str | None = None) -> np.random.Generator:
    """Return a fresh numpy Generator for the given seed.

    Preferred over `seed_everything` for our own sampling: a local generator is
    unaffected by any other library consuming global entropy, so a run stays
    reproducible even when call order changes elsewhere.

    Pass `stream` to get an independent substream of the same seed. Without it,
    two components sharing a seed draw the identical sequence, which is a silent
    confound when both feed the same experiment. The label is folded in with a
    stable hash (not `hash()`, which varies per process), so the substream is
    reproducible across runs and machines.
    """
    base = SEED if seed is None else int(seed)
    if stream is None:
        return np.random.default_rng(base)
    tag = int.from_bytes(
        hashlib.blake2b(stream.encode("utf-8"), digest_size=8).digest(), "big"
    )
    return np.random.default_rng(np.random.SeedSequence([base, tag]))


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) on demand and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
