"""Records and the action map for the four-class relational corpus.

The four classes are not a taxonomy invented here. They are the spec's action
map from the project's formal model, which is a
function of the pair `(rho, nu)` and of nothing else:

    rho  nu   class          action
     1    1   DUPLICATE      increment the observation count, keep one
     1    0   SUPERSESSION   set invalid_at on the old, link forward
     0    1   COINCIDENCE    independent, keep both, no link
     0    0   INTERFERENCE   keep both, mark siblings, never merge

`rho_ij` is exact equality of the referent `r = (entity, attribute)`, `nu_ij` is
exact equality of the value `v`. Both are read off the episode records and never
off a similarity score: the labels have to be independent of the quantity under
test. Serialization reuses `brain.corpus.types.serialize_records`, so this split
hashes under the same canonical-JSONL rule as the existing frozen corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from brain.corpus.types import canonical_line, serialize_records
from brain.corpus.validate import MAX_EPISODE_WORDS, CorpusValidationError

__all__ = [
    "SCHEMA_VERSION", "SUPERSESSION", "INTERFERENCE", "DUPLICATE", "COINCIDENCE",
    "CLASSES", "REGIMES", "ACTION_MAP", "ACTIONS", "RelationEpisode", "LabelledPair",
    "RelationCorpus", "classify", "pair_of", "validate_relations", "canonical_line",
    "serialize_records",
]

#: Bumped whenever the on-disk record shape changes. Recorded in the manifest.
SCHEMA_VERSION = "corpus-relations-v1"

SUPERSESSION = "SUPERSESSION"
INTERFERENCE = "INTERFERENCE"
DUPLICATE = "DUPLICATE"
COINCIDENCE = "COINCIDENCE"

#: Sorted so any iteration over the classes is order-stable (rules.md).
CLASSES: tuple[str, ...] = (COINCIDENCE, DUPLICATE, INTERFERENCE, SUPERSESSION)

#: `long` = multi-sentence session note, boilerplate dominates (matches the
#: existing frozen corpus). `short` = terse assertion, the slot phrase dominates.
#: Theorem 1's correction says the similarity ORDERING of supersession versus
#: interference depends on this ratio, so both are mandatory.
REGIMES: tuple[str, ...] = ("long", "short")

ACTION_MAP: dict[tuple[int, int], str] = {
    (1, 1): DUPLICATE,
    (1, 0): SUPERSESSION,
    (0, 1): COINCIDENCE,
    (0, 0): INTERFERENCE,
}

ACTIONS: dict[str, str] = {
    DUPLICATE: "increment observation count, keep one",
    SUPERSESSION: "set invalid_at = t_new on the old, link forward, keep queryable",
    COINCIDENCE: "independent, keep both, no link",
    INTERFERENCE: "keep both, mark siblings, never merge",
}


def classify(rho: int, nu: int) -> str:
    """The action map, exactly as the Formal Model states it."""
    key = (int(rho), int(nu))
    if key not in ACTION_MAP:
        raise ValueError(f"rho and nu are indicator bits; got {key}")
    return ACTION_MAP[key]


@dataclass(frozen=True, slots=True)
class RelationEpisode:
    """One episode carrying an explicit referent `r` and an asserted value `v`.

    `slot` is the referent `(entity, attribute)`; `value_id` is the canonical
    identity of `v` and `value` its rendered phrase, one for one.
    """

    trace_id: str
    regime: str
    date: str
    text: str
    slot: dict[str, str] = field(default_factory=dict)
    value: str = ""
    value_id: str = ""
    cluster_id: str = ""
    topic: dict[str, str] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    template_variant: int = 0

    @property
    def referent(self) -> tuple[str, str]:
        return (self.slot.get("entity", ""), self.slot.get("attribute", ""))

    @property
    def slot_id(self) -> str:
        entity, attribute = self.referent
        return f"{entity}.{attribute}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RelationEpisode":
        return cls(
            trace_id=str(raw["trace_id"]),
            regime=str(raw["regime"]),
            date=str(raw["date"]),
            text=str(raw["text"]),
            slot={str(k): str(v) for k, v in dict(raw.get("slot", {})).items()},
            value=str(raw.get("value", "")),
            value_id=str(raw.get("value_id", "")),
            cluster_id=str(raw.get("cluster_id", "")),
            topic={str(k): str(v) for k, v in dict(raw.get("topic", {})).items()},
            facts={str(k): str(v) for k, v in dict(raw.get("facts", {})).items()},
            template_variant=int(raw.get("template_variant", 0)),
        )


@dataclass(frozen=True, slots=True)
class LabelledPair:
    """Ground truth for one pair. `a` is the earlier episode by construction.

    `cluster_id` is the resampling unit for the cluster-level BCa bootstrap
    (rules.md: never resample pairs); `slot_id` is the finer alternative unit,
    set only when the pair shares a referent. `same_topic` lets the reader
    stratify on shared context, the `g * c` term in Theorem 1 and the one
    structural confound this corpus cannot remove.
    """

    pair_id: str
    a: str
    b: str
    rho: int
    nu: int
    cls: str
    regime: str
    cluster_id: str = ""
    slot_id: str = ""
    same_topic: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LabelledPair":
        return cls(
            pair_id=str(raw["pair_id"]),
            a=str(raw["a"]),
            b=str(raw["b"]),
            rho=int(raw["rho"]),
            nu=int(raw["nu"]),
            cls=str(raw["cls"]),
            regime=str(raw["regime"]),
            cluster_id=str(raw.get("cluster_id", "")),
            slot_id=str(raw.get("slot_id", "")),
            same_topic=int(raw.get("same_topic", 0)),
        )


def pair_of(pair_id: str, first: RelationEpisode, second: RelationEpisode) -> LabelledPair:
    """Label a pair from the episodes alone, ordering `a` before `b` in time.

    Ties on date fall back to the trace id so the ordering is total and seed
    independent. Supersession then always satisfies `date(a) < date(b)`, which
    is what makes "b replaces a" well defined.
    """
    if first.trace_id == second.trace_id:
        raise CorpusValidationError(f"{pair_id} pairs an episode with itself")
    if first.regime != second.regime:
        raise CorpusValidationError(f"{pair_id} spans two regimes")
    a, b = sorted((first, second), key=lambda e: (e.date, e.trace_id))
    rho = int(a.referent == b.referent)
    nu = int(a.value_id == b.value_id)
    return LabelledPair(
        pair_id=pair_id,
        a=a.trace_id,
        b=b.trace_id,
        rho=rho,
        nu=nu,
        cls=classify(rho, nu),
        regime=a.regime,
        cluster_id=a.cluster_id,
        slot_id=a.slot_id if rho else "",
        same_topic=int(a.cluster_id == b.cluster_id),
    )


@dataclass(frozen=True, slots=True)
class RelationCorpus:
    """Episodes plus labelled pairs plus the manifest they were frozen with."""

    episodes: list[RelationEpisode]
    pairs: list[LabelledPair]
    manifest: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def content_hash(self) -> str:
        return str(self.manifest.get("content_sha256", ""))

    def episode_by_id(self) -> dict[str, RelationEpisode]:
        return {e.trace_id: e for e in self.episodes}

    def clusters(self) -> dict[str, list[RelationEpisode]]:
        """Topic cluster -> its episodes, both levels in sorted id order."""
        out: dict[str, list[RelationEpisode]] = {}
        for episode in sorted(self.episodes, key=lambda e: e.trace_id):
            out.setdefault(episode.cluster_id, []).append(episode)
        return {cid: out[cid] for cid in sorted(out)}

    def episodes_by_regime(self) -> dict[str, list[RelationEpisode]]:
        return {
            regime: [e for e in sorted(self.episodes, key=lambda x: x.trace_id) if e.regime == regime]
            for regime in REGIMES
        }

    def pairs_by_class(self) -> dict[str, list[LabelledPair]]:
        ordered = sorted(self.pairs, key=lambda p: p.pair_id)
        return {cls: [p for p in ordered if p.cls == cls] for cls in CLASSES}

    def class_regime_counts(self) -> dict[str, dict[str, int]]:
        counts = {cls: {regime: 0 for regime in REGIMES} for cls in CLASSES}
        for pair in self.pairs:
            counts[pair.cls][pair.regime] += 1
        return counts


def validate_relations(corpus: RelationCorpus) -> None:
    """Assert the ground truth is unambiguous. A failure here is a generator bug.

    Every label is RE-DERIVED from the episodes rather than trusted, so a pair
    row that disagrees with the action map cannot survive generation.
    """
    by_id = corpus.episode_by_id()
    if len(by_id) != len(corpus.episodes):
        raise CorpusValidationError("duplicate trace_id")
    if len({p.pair_id for p in corpus.pairs}) != len(corpus.pairs):
        raise CorpusValidationError("duplicate pair_id")
    if len({e.text for e in corpus.episodes}) != len(corpus.episodes):
        raise CorpusValidationError("two episodes share identical text")
    for episode in sorted(corpus.episodes, key=lambda e: e.trace_id):
        if episode.regime not in REGIMES:
            raise CorpusValidationError(f"{episode.trace_id} has regime {episode.regime!r}")
        if set(episode.slot) != {"entity", "attribute"}:
            raise CorpusValidationError(f"{episode.trace_id} slot is not (entity, attribute)")
        if not episode.value_id or not episode.value:
            raise CorpusValidationError(f"{episode.trace_id} asserts no value")
        words = len(episode.text.split())
        if words > MAX_EPISODE_WORDS:
            raise CorpusValidationError(
                f"{episode.trace_id} is {words} words, over the {MAX_EPISODE_WORDS} word cap; "
                "the embedder would truncate it and drop the asserted value"
            )
    _validate_pairs(corpus, by_id)


def _validate_pairs(corpus: RelationCorpus, by_id: dict[str, RelationEpisode]) -> None:
    seen: set[tuple[str, str]] = set()
    for pair in sorted(corpus.pairs, key=lambda p: p.pair_id):
        a, b = by_id.get(pair.a), by_id.get(pair.b)
        if a is None or b is None:
            raise CorpusValidationError(f"{pair.pair_id} points at a missing trace")
        key = tuple(sorted((pair.a, pair.b)))
        if key in seen:
            raise CorpusValidationError(f"{pair.pair_id} repeats the pair {key}")
        seen.add(key)
        rebuilt = pair_of(pair.pair_id, a, b)
        if rebuilt != pair:
            raise CorpusValidationError(
                f"{pair.pair_id} disagrees with the action map: stored {pair}, derived {rebuilt}"
            )
        if pair.cls == SUPERSESSION and not a.date < b.date:
            raise CorpusValidationError(
                f"{pair.pair_id} is a supersession without date(a) < date(b)"
            )
    counts = corpus.class_regime_counts()
    empty = sorted(
        f"{cls}/{regime}"
        for cls, per_regime in counts.items()
        for regime, n in per_regime.items()
        if n == 0
    )
    if empty:
        raise CorpusValidationError(f"empty class/regime cells: {', '.join(empty)}")
