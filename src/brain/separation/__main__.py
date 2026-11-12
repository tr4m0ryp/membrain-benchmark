"""Smoke and measurement run: `python -m brain.separation [--full]`.

Prints the two numbers the mechanism lives or dies by, per arm:

* **sibling similarity** -- near-duplicate pairs must move APART (separation).
* **cue rank / margin** -- the correct episode must still win against its
  siblings when the cue is a paraphrase, not a copy (completion). Separation
  that destroys this has bought nothing, and the frontier between the two is the
  tunable `s`, not a fixed property of the transform.

Nothing here writes to disk and nothing touches the frozen corpus, which is read
only when it is already present.
"""

from __future__ import annotations

import sys

import numpy as np

from brain.config import SMOKE_TEST
from brain.separation.policy import SeparationPolicy, build_policy
from brain.store import Trace

_TEXTS: tuple[str, ...] = (
    "auth.py request timeout: raised the connection pool deadline to 30s and the incident closed",
    "auth.py request timeout: the pool was leaking sockets, so the release path got the fix",
    "auth.py request timeout: a stale DNS entry still pointed at the retired replica",
    "auth.py request timeout: the retry budget was exhausted by a hot loop in the client",
    "analytics rollup: renamed a column and backfilled the daily aggregate table",
)
#: Paraphrased cues, one per episode above. Deliberately NOT verbatim: a cue that
#: copies the episode measures nothing, since lexical overlap alone would answer.
_CUES: tuple[str, ...] = (
    "which auth timeout was fixed by giving the pool longer before it gives up",
    "which auth timeout came from sockets never being handed back",
    "which auth timeout pointed at a host that had been decommissioned",
    "which auth timeout was caused by the caller retrying too aggressively",
    "which change touched the daily aggregate for reporting",
)


def _pairwise_sibling_mean(vectors: np.ndarray, n_siblings: int) -> float:
    """Mean cosine over the near-duplicate block (the first `n_siblings` rows)."""
    block = vectors[:n_siblings].astype(np.float64)
    gram = block @ block.T
    upper = gram[np.triu_indices(n_siblings, k=1)]
    return float(upper.mean())


def _cue_accuracy(policy: SeparationPolicy, ids: list[str]) -> tuple[float, float]:
    """Fraction of cues whose own episode ranks first, and the mean margin.

    One cue per stored episode, so the tiny-N path must slice both together.
    """
    stored = policy.store.get_vectors(ids).astype(np.float64)
    cues = np.asarray(policy.encode_queries(list(_CUES[: len(ids)])), dtype=np.float64)
    scores = cues @ stored.T
    correct = scores[np.arange(len(ids)), np.arange(len(ids))]
    rivals = scores.copy()
    rivals[np.arange(len(ids)), np.arange(len(ids))] = -np.inf
    best_rival = rivals.max(axis=1)
    return float((correct > best_rival).mean()), float((correct - best_rival).mean())


def _run(mode: str, s: float, traces: list[Trace], n_siblings: int) -> None:
    from brain.embed import get_embedder

    embedder = get_embedder()
    # tau is quoted in raw-embedding units and calibrated into whatever space
    # the mode stores in, so the two arms are compared at the same collision
    # criterion rather than at the same number.
    policy = build_policy(embedder, mode=mode, s=s, calibrate=True)
    with policy.store as store:
        results = policy.write_many(traces)
        ids = [trace.id for trace in traces]
        stored = store.get_vectors(ids)
        norms = np.linalg.norm(stored.astype(np.float64), axis=1)
        accuracy, margin = _cue_accuracy(policy, ids)
        print(
            f"  mode={mode:<15} s={s:<5} tau={policy.tau:.3f} "
            f"sibling_sim={_pairwise_sibling_mean(stored, n_siblings):.4f} "
            f"cue_acc={accuracy:.3f} margin={margin:+.4f} "
            f"pushes={sum(r.n_collisions for r in results)} "
            f"norm=[{norms.min():.6f}, {norms.max():.6f}] "
            f"finite={bool(np.isfinite(stored).all())}"
        )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    full = "--full" in args and not SMOKE_TEST
    texts = list(_TEXTS[:3]) if SMOKE_TEST else list(_TEXTS)
    n_siblings = min(4, len(texts))
    traces = [Trace(text=text, id=f"e{i}", ts=float(i)) for i, text in enumerate(texts)]
    strengths = (0.0, 0.1, 0.25, 0.5) if full else (0.0, 0.25)
    print(f"texts={len(texts)} siblings={n_siblings} smoke={SMOKE_TEST}")
    for mode in ("decorrelate", "expand_sparsify"):
        for strength in strengths:
            _run(mode, strength, traces, n_siblings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
