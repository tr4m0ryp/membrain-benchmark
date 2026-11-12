"""Propose pairs for a human to label. Propose only.

The four classes are a function of `(rho, nu)`: whether two episodes share a
referent, and whether they assert the same value. On real transcripts neither bit
is observable. The extractor recovers an approximate referent from the diff, and
nothing at all recovers the value. So this module does the one thing that is
sound: it decides which pairs are WORTH a human's time, and refuses to guess what
they are.

Selection is stratified rather than top-k by similarity, because top-k alone would
return an all-positive review set and give a labeller no way to estimate the base
rate:

    same_entity    same file. Named for what it buckets on rather than for rho,
                   which is unknown. Where rho = 1 lives, so DUPLICATE and
                   SUPERSESSION concentrate here and nowhere else.
    near_miss      different file, cosine above the 99th percentile of the
                   cross-referent distribution. Where INTERFERENCE concentrates.
    mid_band       different file, above the median. The genuinely hard cases.
    distant        different file, below the median. The COINCIDENCE control that
                   keeps the review set from being all-positive.

A stratum is a sampling bucket. It is recorded on the pair so the selection
function is auditable, and a labeller who lets it steer the label has broken the
protocol.

The default budget of 68 pairs per stratum per regime yields 544 pairs, matched to
the 540 of the frozen v2 relational split, so a labelled real corpus would be
comparable to the synthetic one at equal N rather than at whatever N fell out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from brain.config import SEED, get_rng
from brain.corpus.transcripts.types import STRATA, CandidateEpisode, CandidatePair

__all__ = ["MiningConfig", "DEFAULT_PER_STRATUM", "mine_pairs", "mining_summary"]

#: Parity with `data/corpus/relations-v2` (540 pairs). Fixed before any pair was
#: mined, so the budget is not a function of what the mining happened to find.
DEFAULT_PER_STRATUM = 68

_NEAR_MISS_Q = 0.99
_MID_Q = 0.50


@dataclass(frozen=True, slots=True)
class MiningConfig:
    per_stratum: int = DEFAULT_PER_STRATUM
    near_miss_quantile: float = _NEAR_MISS_Q
    mid_quantile: float = _MID_Q
    seed: int = SEED

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_stratum": self.per_stratum,
            "near_miss_quantile": self.near_miss_quantile,
            "mid_quantile": self.mid_quantile,
            "seed": self.seed,
        }


def _days_between(first: str, second: str) -> float:
    try:
        a = datetime.fromisoformat(first.replace("Z", "+00:00"))
        b = datetime.fromisoformat(second.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return round(abs((b - a).total_seconds()) / 86400.0, 4)


def _order(episodes: Sequence[CandidateEpisode]) -> list[int]:
    """Stable chronological order. Ties broken on id so `a` before `b` is total."""
    return sorted(range(len(episodes)), key=lambda i: (episodes[i].timestamp, episodes[i].episode_id))


def _sample(rng: np.random.Generator, items: list[tuple[int, int, float]], budget: int):
    if len(items) <= budget:
        return list(items)
    picks = rng.choice(len(items), size=budget, replace=False)
    return [items[int(i)] for i in sorted(picks)]


def _bucket(
    episodes: Sequence[CandidateEpisode], gram: np.ndarray, config: MiningConfig
) -> tuple[dict[str, list[tuple[int, int, float]]], dict[str, float]]:
    """Every admissible pair, assigned to exactly one stratum.

    Thresholds are quantiles of the observed CROSS-referent cosine distribution
    rather than fixed constants, because the real distribution is the unknown
    being measured and a constant tuned on synthetic data would import the very
    assumption this corpus exists to test. The realized values are returned and
    recorded in the manifest.
    """
    entities = [e.referent.get("entity", "") for e in episodes]
    n = len(episodes)
    rows, cols = np.triu_indices(n, k=1)
    cosines = gram[rows, cols]
    same = np.asarray([entities[i] == entities[j] for i, j in zip(rows, cols)])
    cross = cosines[~same]
    if cross.size:
        tau_hi = float(np.quantile(cross, config.near_miss_quantile))
        tau_mid = float(np.quantile(cross, config.mid_quantile))
    else:
        tau_hi = tau_mid = 1.0
    buckets: dict[str, list[tuple[int, int, float]]] = {s: [] for s in STRATA}
    for i, j, cos, is_same in zip(rows, cols, cosines, same):
        value = float(cos)
        if is_same:
            stratum = "same_entity"
        elif value >= tau_hi:
            stratum = "near_miss"
        elif value >= tau_mid:
            stratum = "mid_band"
        else:
            stratum = "distant"
        buckets[stratum].append((int(i), int(j), round(value, 6)))
    return buckets, {"near_miss": round(tau_hi, 6), "mid_band": round(tau_mid, 6)}


def mine_pairs(
    episodes: Sequence[CandidateEpisode],
    vectors: np.ndarray,
    *,
    config: MiningConfig | None = None,
) -> tuple[list[CandidatePair], dict[str, Any]]:
    """Mine one regime. Returns unlabelled pairs and an audit of the selection."""
    config = config or MiningConfig()
    if len(episodes) < 2:
        return [], {"n_episodes": len(episodes), "available": {}, "thresholds": {}}
    order = _order(episodes)
    ordered = [episodes[i] for i in order]
    matrix = np.asarray(vectors, dtype=np.float64)[order]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    gram = matrix @ matrix.T
    buckets, thresholds = _bucket(ordered, gram, config)
    regime = ordered[0].regime
    rng = get_rng(config.seed, stream=f"transcripts.mine.{regime}")
    pairs: list[CandidatePair] = []
    available = {stratum: len(buckets[stratum]) for stratum in STRATA}
    for stratum in STRATA:
        for i, j, cos in _sample(rng, buckets[stratum], config.per_stratum):
            a, b = ordered[i], ordered[j]
            pairs.append(
                CandidatePair(
                    pair_id=f"{regime}:{stratum}:{len(pairs):04d}",
                    a=a.episode_id,
                    b=b.episode_id,
                    regime=regime,
                    stratum=stratum,
                    cos=cos,
                    same_referent=int(a.referent_id == b.referent_id),
                    same_entity=int(
                        a.referent.get("entity", "") == b.referent.get("entity", "")
                    ),
                    days_apart=_days_between(a.timestamp, b.timestamp),
                    label=None,
                )
            )
    audit = {
        "n_episodes": len(episodes),
        "available": available,
        "selected": {s: sum(1 for p in pairs if p.stratum == s) for s in STRATA},
        "thresholds": thresholds,
        "config": config.to_dict(),
    }
    return pairs, audit


def mining_summary(pairs: Sequence[CandidatePair]) -> dict[str, Any]:
    """Shape of the review set. Contains no episode text."""
    by_stratum: dict[str, int] = {s: 0 for s in STRATA}
    for pair in pairs:
        by_stratum[pair.stratum] = by_stratum.get(pair.stratum, 0) + 1
    cosines = np.asarray([p.cos for p in pairs], dtype=np.float64)
    return {
        "n_pairs": len(pairs),
        "n_labelled": sum(1 for p in pairs if p.label is not None),
        "by_stratum": by_stratum,
        "by_regime": {
            regime: sum(1 for p in pairs if p.regime == regime)
            for regime in sorted({p.regime for p in pairs})
        },
        "same_entity_fraction": (
            round(float(np.mean([p.same_entity for p in pairs])), 4) if pairs else 0.0
        ),
        "cos": {
            "mean": round(float(cosines.mean()), 4) if cosines.size else 0.0,
            "min": round(float(cosines.min()), 4) if cosines.size else 0.0,
            "max": round(float(cosines.max()), 4) if cosines.size else 0.0,
        },
    }
