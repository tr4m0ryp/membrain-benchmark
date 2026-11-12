"""Per-class similarity distributions, and the overlap Theorem 1 actually claims.

The corpus is encoded ONCE, in sorted `trace_id` order, into one array every
downstream number reads from. Re-encoding under a different batch composition
perturbs float32 inference by about 1e-7 (embedder docstring): far below anything
measured here, but needless variance between two tables that are the same
measurement.

**Which pairs exist.** `pairs.jsonl` is a released SAMPLE at the Evaluation Plan
composition, not the exhaustive relation. Every within-regime pair is labellable
from `episodes.jsonl` through `brain.corpus.relations.pair_of`, so absence from
the sample is not a negative label. Both populations are built and both reported.

**Which interference population.** The action map makes every `(rho=0, nu=0)`
pair INTERFERENCE, including two episodes from unrelated workstreams that no
collision probe would hand to a classifier. `manifest.conventions` resolves the
ambiguity and is applied class by class:

    INTERFERENCE   same-workstream pairs only   (conventions.interference_selection)
    COINCIDENCE    any same-value, different-referent pair, no context constraint
    SUPERSESSION   every (rho=1, nu=0) pair
    DUPLICATE      every (rho=1, nu=1) pair

`all_interference` keeps the literal action map and is carried only as a
sensitivity: cross-workstream background pairs sit low, which inflates the
separability of the very comparison under test.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from brain.corpus.relations import (
    CLASSES,
    INTERFERENCE,
    REGIMES,
    SUPERSESSION,
    RelationCorpus,
    pair_of,
)

__all__ = [
    "POPULATIONS", "PairRow", "encode_corpus", "gram_matrix", "all_pair_rows", "sample_rows",
    "population", "stratify", "by_class", "summarize", "distribution_table", "support_overlap",
    "direction_report", "subsample",
]

POPULATIONS: tuple[str, ...] = ("derivable", "all_interference", "sample")


@dataclass(frozen=True, slots=True)
class PairRow:
    """One labelled pair and its surface similarity `s`.

    `cluster_id` is the resampling unit named by `manifest.conventions`: the
    workstream of the EARLIER episode. It is the bootstrap block, never the pair.
    """

    pair_id: str
    cls: str
    regime: str
    same_topic: int
    cluster_id: str
    score: float


def encode_corpus(
    corpus: RelationCorpus, embedder: Any | None = None
) -> tuple[list[str], np.ndarray]:
    """Encode every episode ONCE, sorted by trace_id. Returns (ids, unit-norm rows).

    Normalisation is redone in float64 rather than trusted, so an un-normalising
    embedder still yields cosine from a dot product instead of a wrong scale.
    """
    model = embedder
    if model is None:
        from brain.embed.embedder import get_embedder

        model = get_embedder()
    episodes = sorted(corpus.episodes, key=lambda e: e.trace_id)
    ids = [episode.trace_id for episode in episodes]
    vectors = np.asarray(model.encode([e.text for e in episodes]), dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError(f"embedder returned {vectors.shape}, expected ({len(ids)}, dim)")
    if not np.isfinite(vectors).all():
        raise ValueError("embedder produced non-finite vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if float(norms.min()) <= 0.0:
        raise ValueError("embedder produced a zero vector; cosine is undefined")
    return ids, vectors / norms[:, None]


def gram_matrix(vectors: np.ndarray) -> np.ndarray:
    """All-pairs cosine, clipped to [-1, 1] so float error cannot leave the range."""
    return np.clip(vectors @ vectors.T, -1.0, 1.0)


def _row(pair: Any, index: dict[str, int], gram: np.ndarray) -> PairRow:
    return PairRow(
        pair_id=pair.pair_id,
        cls=pair.cls,
        regime=pair.regime,
        same_topic=int(pair.same_topic),
        cluster_id=pair.cluster_id,
        score=float(gram[index[pair.a], index[pair.b]]),
    )


def all_pair_rows(
    corpus: RelationCorpus, ids: Sequence[str], gram: np.ndarray
) -> list[PairRow]:
    """Label EVERY within-regime pair from the episodes, not from `pairs.jsonl`.

    Cross-regime pairs are excluded: both regimes render the SAME plan, so a long
    note and its own short restatement are one assertion, and pairing them would
    measure the rendering rather than the relation.
    """
    index = {trace_id: position for position, trace_id in enumerate(ids)}
    by_regime = corpus.episodes_by_regime()
    rows = [
        _row(pair_of(f"{a.trace_id}|{b.trace_id}", a, b), index, gram)
        for regime in REGIMES
        for a, b in itertools.combinations(by_regime.get(regime, []), 2)
    ]
    rows.sort(key=lambda row: row.pair_id)
    return rows


def sample_rows(
    corpus: RelationCorpus, ids: Sequence[str], gram: np.ndarray
) -> list[PairRow]:
    """The released `pairs.jsonl` sample, scored from the same encoded array."""
    index = {trace_id: position for position, trace_id in enumerate(ids)}
    return [_row(pair, index, gram) for pair in sorted(corpus.pairs, key=lambda p: p.pair_id)]


def population(rows: Sequence[PairRow], name: str) -> list[PairRow]:
    """Apply one of `POPULATIONS` to the fully labelled pair set."""
    if name == "all_interference":
        return list(rows)
    if name == "derivable":
        # conventions.interference_selection: only colliding pairs reach a
        # classifier, so cross-workstream (rho=0, nu=0) pairs are out of scope.
        return [row for row in rows if row.cls != INTERFERENCE or row.same_topic]
    raise ValueError(f"unknown population {name!r}; expected one of {POPULATIONS}")


def stratify(rows: Iterable[PairRow], *, same_topic_only: bool) -> list[PairRow]:
    """Optionally keep only same-workstream pairs: the task-3a confound control."""
    return [row for row in rows if row.same_topic] if same_topic_only else list(rows)


def by_class(rows: Iterable[PairRow], cls: str, regime: str | None = None) -> np.ndarray:
    """Scores for one class, optionally one regime, in stable pair-id order."""
    chosen = sorted(
        (r for r in rows if r.cls == cls and (regime is None or r.regime == regime)),
        key=lambda r: r.pair_id,
    )
    return np.array([row.score for row in chosen], dtype=np.float64)


def summarize(scores: np.ndarray) -> dict[str, Any]:
    """n, mean, sd, p05, p50, p95, min, max. Empty input returns NaNs, not an error."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        keys = ("mean", "sd", "p05", "p50", "p95", "min", "max")
        return {"n": 0, **{key: math.nan for key in keys}}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def distribution_table(rows: Sequence[PairRow]) -> list[dict[str, Any]]:
    """`class, regime, n, mean, sd, p05, p50, p95, max` for every non-empty cell."""
    table: list[dict[str, Any]] = []
    for cls in CLASSES:
        for regime in REGIMES:
            stats = summarize(by_class(rows, cls, regime))
            if stats["n"] == 0:
                continue
            same = sum(1 for r in rows if r.cls == cls and r.regime == regime and r.same_topic)
            table.append({"cls": cls, "regime": regime, "same_topic_n": same, **stats})
    return table


def support_overlap(sup: np.ndarray, inter: np.ndarray) -> dict[str, Any]:
    """Is `supp(s|I) intersect supp(s|S)` non-empty, and how much mass sits there.

    The non-empty intersection is exactly the falsifiable prediction Theorem 1 was
    reduced to. The mass fractions say whether it is a measure-zero touch or a
    real overlap, which the prediction alone cannot distinguish.
    """
    sup = np.asarray(sup, dtype=np.float64)
    inter = np.asarray(inter, dtype=np.float64)
    if sup.size == 0 or inter.size == 0:
        return {"overlaps": False, "interval": None, "reason": "a class is empty in this cell"}
    lower = max(float(sup.min()), float(inter.min()))
    upper = min(float(sup.max()), float(inter.max()))
    overlaps = bool(lower <= upper)

    def mass(values: np.ndarray) -> float:
        return float(np.mean((values >= lower) & (values <= upper))) if overlaps else 0.0

    return {
        "overlaps": overlaps,
        "interval": [lower, upper] if overlaps else None,
        "width": float(upper - lower) if overlaps else 0.0,
        "supersession_support": [float(sup.min()), float(sup.max())],
        "interference_support": [float(inter.min()), float(inter.max())],
        "supersession_mass_in_overlap": mass(sup),
        "interference_mass_in_overlap": mass(inter),
        "supersession_inside_interference_range": float(
            np.mean((sup >= inter.min()) & (sup <= inter.max()))
        ),
        "interference_inside_supersession_range": float(
            np.mean((inter >= sup.min()) & (inter <= sup.max()))
        ),
    }


def direction_report(rows: Sequence[PairRow], *, same_topic_only: bool) -> dict[str, Any]:
    """`E[s|S]` against `E[s|I]` per regime, and whether the sign flips.

    The spec WITHDREW the directional prediction `E[s|I] > E[s|S]`: it depends on
    the ratio of slot-carrying text to shared context. This scores the withdrawn
    prediction rather than rationalising it after the fact.
    """
    chosen = stratify(rows, same_topic_only=same_topic_only)
    per_regime: dict[str, Any] = {}
    for regime in REGIMES:
        sup, inter = by_class(chosen, SUPERSESSION, regime), by_class(chosen, INTERFERENCE, regime)
        if sup.size == 0 or inter.size == 0:
            continue
        delta = float(sup.mean() - inter.mean())
        per_regime[regime] = {
            "n_supersession": int(sup.size),
            "n_interference": int(inter.size),
            "mean_supersession": float(sup.mean()),
            "mean_interference": float(inter.mean()),
            "delta_S_minus_I": delta,
            "direction": "S > I" if delta > 0 else ("S < I" if delta < 0 else "S = I"),
            "withdrawn_prediction_holds": bool(delta < 0),
            **support_overlap(sup, inter),
        }
    signs = {r: math.copysign(1.0, b["delta_S_minus_I"]) for r, b in per_regime.items()}
    return {
        "same_topic_only": bool(same_topic_only),
        "per_regime": per_regime,
        "flips_between_regimes": bool(len(set(signs.values())) > 1),
        "withdrawn_prediction_holds_in": sorted(
            r for r, b in per_regime.items() if b["withdrawn_prediction_holds"]
        ),
    }


def subsample(corpus: RelationCorpus, *, per_class: int = 8) -> RelationCorpus:
    """Tiny-N corpus that still carries all four classes, for the smoke path.

    Selected through the released pairs rather than as a prefix of the sorted
    trace ids: that prefix is entirely background episodes, which share no slot
    and therefore yield no supersession pair at all.
    """
    taken: dict[tuple[str, str], int] = {}
    keep: set[str] = set()
    for pair in sorted(corpus.pairs, key=lambda p: p.pair_id):
        key = (pair.cls, pair.regime)
        if taken.get(key, 0) >= int(per_class):
            continue
        taken[key] = taken.get(key, 0) + 1
        keep |= {pair.a, pair.b}
    return RelationCorpus(
        episodes=[e for e in corpus.episodes if e.trace_id in keep],
        pairs=[p for p in corpus.pairs if p.a in keep and p.b in keep],
        manifest=corpus.manifest,
    )
