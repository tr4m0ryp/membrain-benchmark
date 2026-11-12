"""brain: research testbed for write-time representational separation.

Launch 1 builds only what the minimal falsifying experiment needs: an episodic
store, a deterministic embedder, the separation transform, baselines, a frozen
synthetic interference corpus, and the evaluation harness that runs the gate.
"""

from brain.config import (
    CORPUS_DIR,
    DATA_DIR,
    DB_DEFAULT,
    REPO_ROOT,
    RESULTS_DIR,
    SEED,
    SMOKE_TEST,
    ensure_dir,
    get_rng,
    results_dir,
    seed_everything,
)

__version__ = "0.1.0"

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
    "__version__",
]
