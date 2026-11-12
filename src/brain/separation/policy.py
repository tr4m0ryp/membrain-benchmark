"""SeparationPolicy: the write path of Invention C-A / model M-A.

One write is: embed, probe the store for neighbours above tau, drop the ones
that are byte-identical re-captures, apply T_s to the newcomer and to every
colliding distinct trace, then write all of them back.

Three invariants this file exists to hold:

* **Separation is write-time, on the stored representation.** No read-time
  reranking, no weight-space anything. What a later retrieval sees is what the
  write left behind.
* **Dedup is byte-identical only.** A repeat is detected by content hash, never
  by similarity, and a repeat is never separated (there is nothing to tell apart)
  and never re-pushed (a re-run of the same corpus must not drift the store).
  Similar-but-distinct traces are the thing the mechanism protects.
* **s = 0 is an exact no-op.** No probe, no fetch, no transform: the stored bytes
  equal the embedder's output, so the plain-vector control is genuinely the same
  code path minus the mechanism.

Cost per separating write: 1 id lookup, 1 k-NN probe (brute force, O(N*d), see
M-A's corrected cost note), 1 batched vector fetch, 1 insert, m vector updates.
The probe dominates.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from brain.separation.calibrate import calibrate_tau
from brain.separation.result import (
    DEFAULT_K_PROBE,
    DEFAULT_S,
    DEFAULT_TAU,
    DistinctHint,
    SeparationConfigError,
    WriteResult,
)
from brain.separation.transform import (
    DEFAULT_EXPANSION,
    SeparationTransform,
    make_transform,
    stored_dim,
)
from brain.store import EpisodicStore, Trace

__all__ = ["SeparationPolicy", "build_policy"]


class SeparationPolicy:
    """Write-time separation over an `EpisodicStore` (M-A, decision T6)."""

    def __init__(
        self,
        store: EpisodicStore,
        embedder: Any,
        tau: float = DEFAULT_TAU,
        s: float = DEFAULT_S,
        mode: str | None = None,
        *,
        k_probe: int = DEFAULT_K_PROBE,
        scope_to_project: bool = True,
        transform: SeparationTransform | None = None,
        **transform_kwargs: object,
    ) -> None:
        self.tau = float(tau)
        if not -1.0 <= self.tau <= 1.0:
            raise ValueError(f"tau is a cosine threshold in [-1, 1], got {tau!r}")
        self.s = float(s)
        if not np.isfinite(self.s) or self.s < 0.0:
            raise ValueError(f"s must be finite and >= 0, got {s!r}")
        self.k_probe = int(k_probe)
        if self.k_probe < 1:
            raise ValueError(f"k_probe must be >= 1, got {k_probe!r}")
        self.scope_to_project = bool(scope_to_project)
        self._store = store
        self._embedder = embedder
        if transform is not None:
            if transform_kwargs or mode is not None:
                raise TypeError(
                    "pass either a built transform or a mode plus its parameters, not both"
                )
            self.transform = transform
        else:
            self.transform = make_transform(
                mode or "decorrelate", dim=int(embedder.dim), **transform_kwargs
            )
        self.mode = self.transform.name
        if store.dim != self.transform.out_dim:
            raise SeparationConfigError(
                f"store dim is {store.dim} but mode {self.mode!r} stores "
                f"{self.transform.out_dim}-dim vectors; open the store at "
                f"stored_dim({embedder.dim}, mode={self.mode!r})"
            )

    def __repr__(self) -> str:
        return (
            f"SeparationPolicy(mode={self.mode!r}, tau={self.tau}, s={self.s}, "
            f"k_probe={self.k_probe}, dim={self.transform.out_dim})"
        )

    @property
    def store(self) -> EpisodicStore:
        return self._store

    def params(self) -> dict[str, object]:
        """Run-manifest view of the arm this policy represents."""
        return {
            "tau": self.tau,
            "s": self.s,
            "k_probe": self.k_probe,
            "scope_to_project": self.scope_to_project,
            **self.transform.params(),
        }

    # -- read side ---------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """A cue mapped into the stored space. Retrieval must go through here.

        In `expand_sparsify` the store holds expanded codes, so a raw embedding
        would be the wrong width and, worse, silently the wrong geometry.
        """
        return self.transform.encode(self._embedder.encode_one(text))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.transform.encode(self._embedder.encode(list(texts)))

    def probe(self, vector: np.ndarray, *, exclude: Sequence[str] = (), project: str | None = None,
              ) -> tuple[list[tuple[str, float]], bool]:
        """Neighbours with sim > tau, plus whether the top-k was saturated.

        Returned nearest-first in the store's deterministic (distance, id) order.
        """
        hits = self._store.knn(vector, self.k_probe, project=project, valid_only=True)
        blocked = set(exclude)
        above = [
            (tid, 1.0 - dist)
            for tid, dist in hits
            if tid not in blocked and (1.0 - dist) > self.tau
        ]
        saturated = len(hits) >= self.k_probe and bool(hits) and (1.0 - hits[-1][1]) > self.tau
        return above, saturated

    # -- write side --------------------------------------------------------

    def write(self, trace: Trace, *, distinct_hint: DistinctHint = None) -> WriteResult:
        """Embed, separate against colliding distinct neighbours, store."""
        return self._write(trace, self.transform.encode(self._embedder.encode_one(trace.text)),
                           distinct_hint)

    def write_many(
        self, traces: Sequence[Trace], *, distinct_hint: DistinctHint = None
    ) -> list[WriteResult]:
        """Sequential writes over one batched encode.

        Sequential is the semantics, not an implementation detail: each trace
        must see the store as separation left it, so a later member of a cluster
        collides with its already-separated siblings. Only the encode is batched,
        which is also why it happens up front: batch composition shifts a float32
        embedding by ~1e-7, so every arm must encode the corpus the same way.
        """
        items = list(traces)
        if not items:
            return []
        block = self.transform.encode(self._embedder.encode([t.text for t in items]))
        return [
            self._write(trace, block[index], distinct_hint)
            for index, trace in enumerate(items)
        ]

    def _write(
        self, trace: Trace, vector: np.ndarray, distinct_hint: DistinctHint
    ) -> WriteResult:
        seen = self._store.get(trace.id)
        if seen is not None and seen.content_sha == trace.content_sha:
            # A byte-identical re-capture. It deduped when it was first written;
            # pushing again here would make the store depend on how many times
            # the corpus was replayed.
            return WriteResult(trace_id=trace.id, inserted=False, deduped=True)

        if self.s == 0.0:
            stored = self._store.add(trace, vector)
            return WriteResult(
                trace_id=stored, inserted=stored == trace.id, deduped=stored != trace.id
            )

        scope = trace.project if self.scope_to_project else None
        collisions, saturated = self.probe(vector, exclude=(trace.id,), project=scope)
        collisions, skipped = self._keep_distinct(trace, collisions, distinct_hint)
        if not collisions:
            stored = self._store.add(trace, vector)
            return WriteResult(
                trace_id=stored,
                inserted=stored == trace.id,
                deduped=stored != trace.id,
                probe_saturated=saturated,
                skipped_ids=tuple(skipped),
            )

        ids = [tid for tid, _ in collisions]
        moved_new, updates = self.transform.separate(vector, self._store.get_vectors(ids), self.s)
        # Insert before the write-back: if the store resolves this to an existing
        # row (a byte-identical capture arriving under another id), nothing was
        # inserted and nothing collided, so no neighbour may be disturbed.
        stored = self._store.add(trace, moved_new)
        if stored != trace.id:
            return WriteResult(trace_id=stored, inserted=False, deduped=True)
        self._store.update_vectors(zip(ids, updates))
        return WriteResult(
            trace_id=stored,
            inserted=True,
            deduped=False,
            collisions=tuple(collisions),
            updated_ids=tuple(ids),
            probe_saturated=saturated,
            skipped_ids=tuple(skipped),
        )

    def _keep_distinct(
        self,
        trace: Trace,
        collisions: list[tuple[str, float]],
        distinct_hint: DistinctHint,
    ) -> tuple[list[tuple[str, float]], list[str]]:
        """Apply the caller's distinctness hint. Default: everything is distinct.

        The default is deliberate. Treating a similar neighbour as the same
        episode is the industry failure mode this project exists to refute, so
        the burden is on the caller to assert sameness, never on the policy to
        infer it.
        """
        if distinct_hint is None or distinct_hint is True:
            return collisions, []
        if distinct_hint is False:
            return [], [tid for tid, _ in collisions]
        kept: list[tuple[str, float]] = []
        skipped: list[str] = []
        for tid, sim in collisions:
            other = self._store.get(tid)
            if other is not None and other.content_sha == trace.content_sha:
                skipped.append(tid)  # byte-identical: dedup, never separate
                continue
            if other is None or bool(distinct_hint(trace, other)):
                kept.append((tid, sim))
            else:
                skipped.append(tid)
        return kept, skipped


def build_policy(
    embedder: Any,
    *,
    db_path: str = ":memory:",
    mode: str = "decorrelate",
    tau: float = DEFAULT_TAU,
    s: float = DEFAULT_S,
    expansion: int = DEFAULT_EXPANSION,
    calibrate: bool = False,
    k_probe: int = DEFAULT_K_PROBE,
    scope_to_project: bool = True,
    **kwargs: object,
) -> SeparationPolicy:
    """Open a store at the width `mode` needs and wrap it in a policy.

    The store fixes its dim at construction and `expand_sparsify` stores D-dim
    codes, so getting that pairing wrong is the one configuration error that
    would otherwise surface as a dimension mismatch deep inside a write.

    `calibrate=True` reads `tau` in raw-embedding units and maps it into the
    stored space (see `calibrate_tau`). Without it, a raw threshold handed to
    `expand_sparsify` sits far above every expanded similarity and the collision
    detector never fires, which looks like a working arm that separates nothing.
    """
    dim = stored_dim(int(embedder.dim), mode=mode, expansion=expansion)
    extra = dict(kwargs)
    if mode == "expand_sparsify":
        extra.setdefault("expansion", expansion)
    transform = make_transform(mode, dim=int(embedder.dim), **extra)
    threshold = calibrate_tau(transform, tau) if calibrate else float(tau)
    store = EpisodicStore(db_path, dim)
    return SeparationPolicy(
        store,
        embedder,
        tau=threshold,
        s=s,
        transform=transform,
        k_probe=k_probe,
        scope_to_project=scope_to_project,
    )
