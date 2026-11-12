"""The separation arm: Task 005's write path behind Task 006's `Retriever` interface.

Task 005 shipped a WRITE path (`SeparationPolicy.write_many`), not a retriever,
so the gate could not call it interchangeably with the baselines. This wraps it
without changing it: index goes through `write_many`, retrieve goes through
`encode_query` and the same `store.knn` the plain-vector arm uses.

Three things here exist to keep the comparison fair rather than to make it work:

* **One embedding pass.** The policy embeds through `embedder.encode`, and
  re-encoding the corpus under a different batch composition moves a float32
  embedding by about 1e-7 (`brain.baselines.vectors`). `CacheEmbedder` routes the
  policy through the SAME process-wide `VectorCache` every other vector arm uses,
  so the separation arm and the plain-vector control see bit-identical inputs.
* **The query passes through the transform.** In `expand_sparsify` the store
  holds D-dim expanded codes, so a raw embedding would be the wrong width and,
  at equal width, the wrong geometry. `Retriever.retrieve` therefore encodes the
  cue with `policy.encode_query`, never with the raw embedder.
* **tau is calibrated, not copied.** A raw 0.75 lands near 0.41 once expanded and
  sparsified, so handing the literal number to the sparse arm would leave its
  collision detector permanently silent. `calibrate=True` compares the two arms
  at the same CRITERION instead of the same number.

The store is opened at `stored_dim(mode)`, which is why `VectorStoreArm` could not
be reused: it asserts store dim == embedder dim, true only for `decorrelate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brain.baselines.base import BaseRetriever, IndexingError
from brain.baselines.vectors import VectorCache, shared_vector_cache
from brain.separation import (
    DEFAULT_AGGREGATION,
    DEFAULT_EXPANSION,
    DEFAULT_K_PROBE,
    DEFAULT_S,
    DEFAULT_TAU,
    build_policy,
)
from brain.store import Trace

__all__ = ["CacheEmbedder", "SeparationArm", "arm_label"]


class CacheEmbedder:
    """The shared `VectorCache` wearing the embedder interface the policy needs.

    Exposes exactly the three members `SeparationPolicy` touches (`dim`,
    `encode`, `encode_one`) and nothing else, so it cannot accidentally become a
    second embedding path.
    """

    def __init__(self, cache: VectorCache | None = None) -> None:
        self._cache = cache if cache is not None else shared_vector_cache()

    @property
    def cache(self) -> VectorCache:
        return self._cache

    @property
    def dim(self) -> int:
        return self._cache.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._cache.matrix(list(texts))

    def encode_one(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"text is {type(text).__name__}, expected str")
        return self._cache.vector(text)


def arm_label(mode: str, s: float, aggregation: str = DEFAULT_AGGREGATION) -> str:
    """Stable, file-safe arm name carrying the configuration it was run at."""
    return f"separation-{mode}-s{float(s):g}-{aggregation}"


class SeparationArm(BaseRetriever):
    """Write-time separation (M-A) as a retrieval arm.

    `projection_seed` seeds the expansion matrix of `expand_sparsify`. It is the
    only randomness any arm in this launch has, so the harness varies it per run
    seed; that is what turns the across-seed interval for the sparse arm into a
    real measurement rather than a zero-width artefact. `decorrelate` ignores it,
    correctly: it has no random component.
    """

    name = "separation"

    def __init__(
        self,
        *,
        mode: str = "decorrelate",
        s: float = DEFAULT_S,
        tau: float = DEFAULT_TAU,
        aggregation: str = DEFAULT_AGGREGATION,
        expansion: int = DEFAULT_EXPANSION,
        projection_seed: int | None = None,
        calibrate: bool = True,
        k_probe: int = DEFAULT_K_PROBE,
        db_path: str | Path = ":memory:",
        cache: VectorCache | None = None,
        name: str | None = None,
    ) -> None:
        # `expansion` is passed to build_policy as its own argument (it also
        # decides the store width), so it must NOT be repeated here.
        extra: dict[str, Any] = {"aggregation": str(aggregation)}
        if mode == "expand_sparsify" and projection_seed is not None:
            extra["seed"] = int(projection_seed)
        self._policy = build_policy(
            CacheEmbedder(cache),
            db_path=str(db_path),
            mode=str(mode),
            tau=float(tau),
            s=float(s),
            expansion=int(expansion),
            calibrate=bool(calibrate),
            k_probe=int(k_probe),
            # The frozen corpus carries no project labels, so a project-scoped
            # probe would filter on the empty string: a no-op dressed as a
            # filter. Disabled so the probe scope is explicit.
            scope_to_project=False,
            **extra,
        )
        super().__init__(
            name=name or arm_label(mode, s, aggregation),
            params={
                "mechanism": "M-A write-time separation",
                "raw_tau": float(tau),
                "tau_calibrated": bool(calibrate),
                "projection_seed": projection_seed,
                "stored_dim": self._policy.store.dim,
                "read_time_rerank": None,
                "metric": "cosine",
                **self._policy.params(),
            },
        )
        self.stats.update(
            {
                "separating_writes": 0,
                "collisions": 0,
                "vector_updates": 0,
                "probe_saturated_writes": 0,
                "deduped_writes": 0,
            }
        )

    @property
    def policy(self) -> Any:
        return self._policy

    def close(self) -> None:
        self._policy.store.close()

    # -- write path --------------------------------------------------------

    def _index(self, traces: Sequence[Trace]) -> None:
        items = list(traces)
        results = self._policy.write_many(items)
        written = self._policy.store.count()
        if written != len(items):
            # content_sha is UNIQUE, so byte-identical episodes collapse into one
            # row and every metric shifts without an error being raised.
            raise IndexingError(
                f"{self.name}: wrote {len(items)} traces but the store holds "
                f"{written} rows; byte-identical texts were deduped"
            )
        self.stats["separating_writes"] = sum(1 for r in results if r.separated)
        self.stats["collisions"] = sum(r.n_collisions for r in results)
        self.stats["vector_updates"] = sum(len(r.updated_ids) for r in results)
        self.stats["probe_saturated_writes"] = sum(1 for r in results if r.probe_saturated)
        self.stats["deduped_writes"] = sum(1 for r in results if r.deduped)
        # rho as M-A states it: colliding pairs found per write that had any
        # store to collide with. The first write has none, hence len - 1.
        denominator = max(1, len(items) - 1)
        self.stats["collisions_per_write"] = float(self.stats["collisions"]) / denominator

    # -- read path ---------------------------------------------------------

    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Cue through the same transform, then the same cosine k-NN as plain vector."""
        vector = self._policy.encode_query(query)
        return [
            (trace_id, 1.0 - distance)
            for trace_id, distance in self._policy.store.knn(vector, k)
        ]
