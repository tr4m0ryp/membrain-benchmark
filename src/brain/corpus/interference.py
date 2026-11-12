"""Formal Model M-A diagnostics measured on the corpus itself.

M-A defines tau as the cosine collision threshold (two distinct traces collide
when sim > tau) and rho = P(sim > tau) as the pairwise collision probability,
then predicts that a plain vector store degrades as

    Acc(m) = 1 / (1 + rho * (m - 1))

while write-time separation stays flat. Both quantities are properties of the
CORPUS plus the embedding geometry, not of any mechanism, so they belong in the
manifest: they say how much interference the instrument actually carries before
anything is built on top of it. If rho were near zero the corpus would not test
the hypothesis at all, and the gate's second kill clause (the baseline curve is
flat too) would fire for want of an instrument rather than for want of an effect.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["DEFAULT_TAUS", "collision_rates", "fit_collision_rho"]

#: Thresholds swept for rho. Chosen to bracket the sibling similarity this
#: corpus shows under all-MiniLM-L6-v2, which sits near 0.82.
DEFAULT_TAUS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)


def collision_rates(
    vectors: np.ndarray, cluster_ids: Sequence[str], taus: Sequence[float] = DEFAULT_TAUS
) -> list[dict[str, float]]:
    """Empirical rho at each tau, split by whether the pair are siblings.

    Every pair counted here is a pair of DISTINCT episodes, which is exactly the
    population M-A's rho ranges over: byte-identical captures are deduplicated
    upstream and never reach the store as two traces.
    """
    gram = vectors @ vectors.T
    labels = np.asarray(cluster_ids)
    upper = np.triu(np.ones_like(gram, dtype=bool), k=1)
    same = labels[:, None] == labels[None, :]
    sibling = gram[same & upper]
    non_sibling = gram[(~same) & upper]
    all_pairs = gram[upper]
    rows: list[dict[str, float]] = []
    for tau in taus:
        rows.append(
            {
                "tau": float(tau),
                "rho_sibling": _rate(sibling, tau),
                "rho_non_sibling": _rate(non_sibling, tau),
                "rho_all_pairs": _rate(all_pairs, tau),
            }
        )
    return rows


def _rate(values: np.ndarray, tau: float) -> float:
    if values.size == 0:
        return math.nan
    return round(float((values > tau).mean()), 6)


def fit_collision_rho(accuracy_by_cluster_size: Mapping[str, float]) -> dict[str, Any]:
    """Least-squares fit of Acc(m) = 1 / (1 + rho * (m - 1)).

    Linearised as 1/Acc - 1 = rho * (m - 1), a regression through the origin, so
    the estimate is closed form and cannot fail to converge. Sizes with zero
    accuracy are dropped (the transform is undefined there) and m = 1 carries no
    information about rho, so it is dropped too.
    """
    xs: list[float] = []
    ys: list[float] = []
    for raw_size, accuracy in sorted(accuracy_by_cluster_size.items(), key=lambda kv: int(kv[0])):
        size = int(raw_size)
        if size < 2 or accuracy <= 0.0:
            continue
        xs.append(float(size - 1))
        ys.append(1.0 / float(accuracy) - 1.0)
    if not xs:
        return {"rho": math.nan, "n_points": 0, "max_abs_error": math.nan, "predicted": {}}
    x = np.asarray(xs)
    y = np.asarray(ys)
    denominator = float((x * x).sum())
    rho = float((x * y).sum() / denominator) if denominator > 0 else math.nan
    predicted = {
        str(int(xi) + 1): round(1.0 / (1.0 + rho * xi), 6) for xi in xs
    }
    errors = [
        abs(predicted[str(int(xi) + 1)] - float(accuracy_by_cluster_size[str(int(xi) + 1)]))
        for xi in xs
    ]
    return {
        "model": "Acc(m) = 1 / (1 + rho * (m - 1))",
        "rho": round(rho, 6),
        "n_points": len(xs),
        "predicted": predicted,
        "max_abs_error": round(max(errors), 6),
    }
