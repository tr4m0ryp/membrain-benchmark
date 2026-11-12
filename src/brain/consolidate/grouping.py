"""Which episodes the baseline decides to merge, and why.

The universal shipped policy is "merge what is similar above tau". The naive
implementation of that is the transitive closure of `sim >= tau`, i.e. single
linkage, which chains A-B-C together when A and C are far apart and so merges
pairs the policy itself never judged similar. That would inflate the measured
harm in our favour, so the default here is **complete linkage**: every pair
inside a merged group is itself above tau, and every merge is one the shipped
policy would defend. `single` reproduces the naive closure and `average` sits
between them; both stay selectable so the choice can be swept rather than
assumed.

Nothing in this module calls a model, which is what lets `plan_groups` price a
whole sweep before a cent is spent.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from brain.consolidate.types import (
    DEFAULT_LINKAGE,
    DEFAULT_THRESHOLDS,
    LINKAGES,
    MergeGroup,
)
from brain.store import EpisodicStore

__all__ = ["group_traces", "plan_groups", "groups_for"]

_EPSILON = 1e-12


def group_traces(
    ids: Sequence[str],
    vectors: Any,
    *,
    threshold: float,
    linkage: str = DEFAULT_LINKAGE,
) -> list[tuple[tuple[str, ...], float]]:
    """Agglomerate on cosine similarity and cut at `threshold`.

    Returns `(member ids, score)` per group of size >= 2, where `score` is the
    weakest link admitted; under complete linkage that is exactly the group's
    minimum pairwise similarity.

    Determinism: ids are sorted before the matrix is built and the greedy step
    takes `argmax` over that id-sorted matrix, so an exact tie resolves to the
    lexicographically smallest pair instead of to whatever order the caller
    happened to pass. Similarities are computed in float64 and clipped to
    [-1, 1], which absorbs the float32 rounding that can otherwise put an exact
    self-match fractionally above 1.

    Complexity is O(n^3) worst case with an O(n^2) vectorised inner step. At
    corpus scale that is milliseconds, and the clarity is worth more than an
    asymptotically better nearest-neighbour-chain implementation would be.
    """
    if linkage not in LINKAGES:
        raise ValueError(f"linkage must be one of {LINKAGES}, got {linkage!r}")
    order = sorted(range(len(ids)), key=lambda i: str(ids[i]))
    labels = [str(ids[i]) for i in order]
    n = len(labels)
    if n < 2:
        return []
    matrix = np.asarray(vectors, dtype=np.float64)[order]
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), _EPSILON)
    sim = np.clip(matrix @ matrix.T, -1.0, 1.0)
    np.fill_diagonal(sim, -np.inf)

    members: list[list[int]] = [[i] for i in range(n)]
    sizes = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    scores = [1.0] * n
    tau = float(threshold)
    while True:
        # A merged-away cluster is masked by writing -inf into its row and
        # column rather than by rebuilding a masked copy each iteration: same
        # argmax, O(n) writes instead of an O(n^2) allocation per merge.
        flat = int(np.argmax(sim))
        best = float(sim.flat[flat])
        if not np.isfinite(best) or best < tau:
            break
        row, col = divmod(flat, n)
        i, j = (row, col) if row < col else (col, row)
        # Lance-Williams update, expressed on similarity rather than distance.
        if linkage == "single":
            new = np.maximum(sim[i], sim[j])
        elif linkage == "complete":
            new = np.minimum(sim[i], sim[j])
        else:
            new = (sizes[i] * sim[i] + sizes[j] * sim[j]) / (sizes[i] + sizes[j])
        sim[i, :], sim[:, i] = new, new
        sim[j, :], sim[:, j] = -np.inf, -np.inf
        sim[i, i] = -np.inf
        members[i] = members[i] + members[j]
        sizes[i] += sizes[j]
        scores[i] = best
        active[j] = False

    groups = [
        (tuple(sorted(labels[m] for m in members[i])), scores[i])
        for i in range(n)
        if active[i] and len(members[i]) > 1
    ]
    groups.sort(key=lambda group: group[0])
    return groups


def groups_for(
    traces: Sequence[Any],
    vectors: Any,
    *,
    threshold: float,
    linkage: str = DEFAULT_LINKAGE,
) -> list[MergeGroup]:
    """`group_traces` plus the corpus cluster labels the members carried."""
    by_id = {trace.id: trace for trace in traces}
    ids = [trace.id for trace in traces]
    return [
        MergeGroup(members, score, tuple(by_id[i].cluster_id or "" for i in members))
        for members, score in group_traces(
            ids, vectors, threshold=threshold, linkage=linkage
        )
    ]


def plan_groups(
    store: EpisodicStore,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    linkage: str = DEFAULT_LINKAGE,
) -> dict[float, list[MergeGroup]]:
    """Groups per threshold with no model call, so a pass can be priced first.

    A merge depends only on its group's texts, so the LLM cost of a whole sweep
    is exactly `sum(len(groups))` calls: the destructive and non-destructive
    modes share every answer, and so does every seed at temperature 0.
    """
    traces = store.all_traces(valid_only=True)
    vectors = store.get_vectors([trace.id for trace in traces])
    return {
        float(tau): groups_for(traces, vectors, threshold=float(tau), linkage=linkage)
        for tau in sorted({float(t) for t in thresholds})
    }
