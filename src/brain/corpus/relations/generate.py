"""Deterministic generator for the four-class relational corpus.

Design contract (spec: `## Formal Model`, the action map, F17, Evaluation Plan):

- The store is planned ONCE as referent timelines. A referent `r = (entity,
  attribute)` accumulates states over time; a state holds one value `v` and one
  or two episodes asserting it. Within-timeline pairs are therefore DUPLICATE
  (same state) or SUPERSESSION (different states, ordered in time) by
  construction, with no reference to any similarity score.
- The plan is rendered TWICE, once per regime, so `long` and `short` carry the
  identical relational content and differ only in the ratio of slot-carrying text
  to shared context text. That ratio is what Theorem 1's correction says the
  supersession/interference ordering depends on, so the per-regime comparison is
  paired rather than confounded by content.
- **The renderer never sees a pair class** (`content.render_text`). Classes are
  read off the plan afterwards. This is the instrument-bias rule: generate
  naturalistically, then stop.
  No similarity is computed anywhere in this package.
- `home_topic_prob` was fixed a priori at 0.6, because a fact is usually, but not
  always, revised in the workstream that first recorded it. It is the knob that
  moves the shared-context term, so it is manifested and never tuned.
- **Task 007 adds `scaled_relations_config` and nothing else.** The K1 interval
  came back straddling the bar in the long regime, which is a POWER problem, so v2
  multiplies every COUNT in the v1 config by one integer and moves no per-slot
  parameter, no probability and no template. A slot is an independent draw from
  the same process, so v2 is v1 with more draws, comparable by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Sequence

import numpy as np

from brain.config import SEED, SMOKE_TEST, get_rng
from brain.corpus.relations.content import (
    N_INCIDENTAL,
    N_VARIANTS,
    TOPICS_BY_AREA,
    VALID_SLOTS,
    VALUES,
    Attribute,
    Entity,
    Workstream,
    render_text,
)
from brain.corpus.relations.types import (
    COINCIDENCE, DUPLICATE, INTERFERENCE, REGIMES, SUPERSESSION,
    LabelledPair, RelationCorpus, RelationEpisode, pair_of, validate_relations,
)

__all__ = ["RelationsConfig", "SCALED_COUNTS", "V2_SCALE", "default_relations_config",
           "scaled_relations_config", "generate_relations"]

_EPOCH = date(2026, 1, 5)

#: The only fields task 007 is allowed to touch: pure counts. Everything else
#: (probabilities, states per slot, date span, seed, template family) is a shape
#: parameter, and moving one would make v2 a different distribution from v1.
SCALED_COUNTS: tuple[str, ...] = ("coincidence_pairs", "duplicate_pairs", "interference_pairs",
                                  "n_background", "n_slots", "spotcheck_n", "supersession_pairs")

#: Fixed from COUNTS ALONE, before any similarity was computed (rules.md, the
#: instrument-bias rule): the smallest integer scale that both saturates the
#: 30-workstream bank, so every cluster carries same-topic supersession pairs,
#: and clears the 180-pair target for the long-regime same-topic stratum.
#: k=1 gives 19 clusters / 69 pairs, k=2 gives 24 / 137, k=3 gives 30 / 234.
V2_SCALE = 3


@dataclass(frozen=True, slots=True)
class RelationsConfig:
    """Every knob that affects corpus content. Serialized into the manifest.

    Pair targets are PER REGIME. The plan is rendered in both regimes, so the
    released totals are twice these numbers: at v1 that is 60 supersession, 60
    interference, 30 duplicate, 30 coincidence, the Evaluation Plan composition.
    """

    seed: int = SEED
    n_slots: int = 24
    states_per_slot: int = 3
    restatement_prob: float = 0.4
    n_background: int = 40
    home_topic_prob: float = 0.6
    supersession_pairs: int = 30
    interference_pairs: int = 30
    duplicate_pairs: int = 15
    coincidence_pairs: int = 15
    date_span: int = 240
    spotcheck_n: int = 40
    spotcheck_seed: int = SEED
    generator: str = "relations-template-v1"
    label: str = "relations"

    def __post_init__(self) -> None:
        if self.states_per_slot < 2:
            raise ValueError("a referent needs at least two states to supersede anything")
        if self.n_slots + self.n_background > len(VALID_SLOTS):
            raise ValueError(f"need {self.n_slots + self.n_background} slots, bank holds {len(VALID_SLOTS)}")
        if not 0.0 <= self.home_topic_prob <= 1.0 or not 0.0 <= self.restatement_prob <= 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
        if min(self.targets.values()) < 1:
            raise ValueError("every class needs at least one pair per regime")
        if self.states_per_slot > min(len(bank) for bank in VALUES.values()):
            raise ValueError("states_per_slot exceeds the smallest value bank")

    @property
    def targets(self) -> dict[str, int]:
        """Pairs to draw per class per regime, in a fixed sorted order."""
        return {COINCIDENCE: self.coincidence_pairs, DUPLICATE: self.duplicate_pairs,
                INTERFERENCE: self.interference_pairs, SUPERSESSION: self.supersession_pairs}

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "pairs_per_regime": self.targets, "regimes": list(REGIMES)}


def default_relations_config(*, smoke: bool | None = None, **overrides: Any) -> RelationsConfig:
    """Full config by default; a tiny one when SMOKE_TEST is on."""
    base: dict[str, Any] = {}
    if SMOKE_TEST if smoke is None else smoke:
        base = dict(n_slots=8, n_background=8, supersession_pairs=4, interference_pairs=4,
                    duplicate_pairs=2, coincidence_pairs=2, spotcheck_n=6, label="smoke")
    return RelationsConfig(**{**base, **overrides})


def scaled_relations_config(scale: int = V2_SCALE, **overrides: Any) -> RelationsConfig:
    """The v1 config with every COUNT multiplied by `scale`, and nothing else.

    This is the whole of task 007's generator change. `SCALED_COUNTS` lists the
    fields that move; every shape parameter is copied from the v1 default, so
    each of the `scale * n_slots` referent timelines is drawn from the same
    process that produced v1's 24. The pair-sample targets scale by the same
    factor, which holds `pairs.jsonl` at the v1 class composition and keeps the
    spot-check at the same fraction of the released sample.
    """
    if int(scale) < 1:
        raise ValueError(f"scale must be a positive integer, got {scale!r}")
    base = RelationsConfig()
    counts = {name: getattr(base, name) * int(scale) for name in SCALED_COUNTS}
    label = base.label if int(scale) == 1 else f"{base.label}-v2-x{int(scale)}"
    return RelationsConfig(**{**counts, "label": label, **overrides})


@dataclass(frozen=True, slots=True)
class _PlanEpisode:
    """One assertion, rendered later into both regimes."""

    key: str
    entity: Entity
    attribute: Attribute
    value_id: str
    value: str
    topic: Workstream
    day: str
    variant: int
    incidental: tuple[int, int]
    state: int
    role: str


def _draw_topic(entity: Entity, home: Workstream, config: RelationsConfig,
                rng: np.random.Generator) -> Workstream:
    """Home workstream with probability `home_topic_prob`, else a neighbour.

    That variability is the `sim_ctx` term in Theorem 1, and it has to come from
    the world model rather than a per-class assignment.
    """
    neighbours = [w for w in TOPICS_BY_AREA[entity.area] if w.topic_id != home.topic_id]
    if not neighbours or rng.random() < config.home_topic_prob:
        return home
    return neighbours[int(rng.integers(0, len(neighbours)))]


def _plan_store(config: RelationsConfig, rng: np.random.Generator) -> list[_PlanEpisode]:
    """Draw slots, their value timelines, their dates and their workstreams."""
    chosen = rng.choice(len(VALID_SLOTS), size=config.n_slots + config.n_background, replace=False)
    plan: list[_PlanEpisode] = []
    for index, slot_index in enumerate(int(x) for x in chosen):
        entity, attribute = VALID_SLOTS[slot_index]
        bank = VALUES[attribute.value_type]
        area_topics = TOPICS_BY_AREA[entity.area]
        home = area_topics[int(rng.integers(0, len(area_topics)))]
        timeline = index < config.n_slots
        n_states = config.states_per_slot if timeline else 1
        picks = [int(x) for x in rng.choice(len(bank), size=n_states, replace=False)]
        repeats = [1 + int(rng.random() < config.restatement_prob) if timeline else 1 for _ in picks]
        # One sorted date draw per slot, consumed in state order, so every state
        # strictly precedes the next and date(a) < date(b) holds for supersession.
        offsets = sorted(int(x) for x in rng.choice(config.date_span, size=sum(repeats), replace=False))
        cursor = 0
        for state, (pick, count) in enumerate(zip(picks, repeats)):
            base_variant = int(rng.integers(0, N_VARIANTS))
            for repeat in range(count):
                # A restatement is a REPHRASING: a different variant keeps duplicate
                # pairs non-identical, which the validator requires. Keyword order is
                # load bearing: `topic` then `incidental` is the RNG draw order.
                plan.append(_PlanEpisode(
                    key=f"s{index:02d}-e{cursor:02d}" if timeline else f"bg{index - config.n_slots:02d}",
                    entity=entity, attribute=attribute, value=bank[pick],
                    value_id=f"{attribute.value_type}.{pick:02d}",
                    topic=_draw_topic(entity, home, config, rng),
                    day=(_EPOCH + timedelta(days=offsets[cursor])).isoformat(),
                    variant=(base_variant + repeat) % N_VARIANTS,
                    incidental=(int(rng.integers(0, N_INCIDENTAL)), int(rng.integers(0, N_INCIDENTAL))),
                    state=state, role="timeline" if timeline else "background",
                ))
                cursor += 1
    return plan


def _episode(item: _PlanEpisode, regime: str) -> RelationEpisode:
    return RelationEpisode(
        trace_id=f"{regime}-{item.key}",
        regime=regime,
        date=item.day,
        text=render_text(entity=item.entity, attribute=item.attribute, value=item.value,
                         topic=item.topic, day=item.day, variant=item.variant,
                         incidental=item.incidental, regime=regime),
        slot={"entity": item.entity.entity_id, "attribute": item.attribute.attribute_id},
        value=item.value,
        value_id=item.value_id,
        cluster_id=item.topic.topic_id,
        topic={"topic_id": item.topic.topic_id, "area": item.topic.area,
               "path": item.topic.path, "focus": item.topic.focus},
        facts={"entity_noun": item.entity.noun, "entity_kind": item.entity.kind,
               "attribute_phrase": item.attribute.phrase,
               "value_type": item.attribute.value_type,
               "state": str(item.state), "role": item.role},
        template_variant=item.variant,
    )


def _candidates(plan: Sequence[_PlanEpisode]) -> dict[str, list[tuple[str, str]]]:
    """Enumerate every candidate pair by class, from the plan alone.

    Interference is restricted to SAME-WORKSTREAM pairs. Under the action map every
    (rho = 0, nu = 0) pair is interference, but only colliding pairs ever reach a
    classifier: a merge policy fires above tau, and cross-workstream pairs never get
    there (F17 measured the unrelated bucket at 0.346). Widening the class to the
    whole store would answer an easier question than K1 asks. Both flavours stay
    recoverable: every episode carries its slot and its value, so any pair at all is
    relabellable from `episodes.jsonl`.
    """
    out: dict[str, list[tuple[str, str]]] = {c: [] for c in (COINCIDENCE, DUPLICATE, INTERFERENCE, SUPERSESSION)}
    for i in range(len(plan)):
        first = plan[i]
        for j in range(i + 1, len(plan)):
            second = plan[j]
            same_slot = (first.entity, first.attribute) == (second.entity, second.attribute)
            same_value = first.value_id == second.value_id
            if same_slot:
                out[DUPLICATE if same_value else SUPERSESSION].append((first.key, second.key))
            elif same_value:
                out[COINCIDENCE].append((first.key, second.key))
            elif first.topic.topic_id == second.topic.topic_id:
                out[INTERFERENCE].append((first.key, second.key))
    return {cls: sorted(items) for cls, items in out.items()}


def _sample(candidates: dict[str, list[tuple[str, str]]], config: RelationsConfig,
            rng: np.random.Generator) -> list[tuple[str, str]]:
    """Draw the target number of pairs per class, without replacement."""
    picked: list[tuple[str, str]] = []
    # Sorted, not dict order: iteration order feeds the RNG, so it must not depend
    # on how the targets dict happened to be built (rules.md, determinism).
    for cls in sorted(config.targets):
        pool, want = candidates[cls], config.targets[cls]
        if len(pool) < want:
            raise ValueError(f"only {len(pool)} {cls} candidates for a target of {want}; enlarge "
                             "the store (n_slots, n_background), never the text templates")
        for index in sorted(int(x) for x in rng.choice(len(pool), size=want, replace=False)):
            picked.append(pool[index])
    return picked


def generate_relations(config: RelationsConfig | None = None) -> RelationCorpus:
    """Generate the corpus. Same config plus same seed gives identical records."""
    config = config or default_relations_config()
    plan = _plan_store(config, get_rng(config.seed, stream="relations-store"))
    episodes: list[RelationEpisode] = []
    by_key: dict[tuple[str, str], RelationEpisode] = {}
    for regime in REGIMES:
        for item in plan:
            episode = _episode(item, regime)
            episodes.append(episode)
            by_key[(regime, item.key)] = episode
    sampled = _sample(_candidates(plan), config, get_rng(config.seed, stream="relations-pairs"))
    pairs: list[LabelledPair] = []
    for index, (left, right) in enumerate(sampled):
        for regime in REGIMES:
            pairs.append(pair_of(f"p{index:04d}-{regime}", by_key[(regime, left)],
                                 by_key[(regime, right)]))
    episodes.sort(key=lambda e: e.trace_id)
    pairs.sort(key=lambda p: p.pair_id)
    corpus = RelationCorpus(episodes=episodes, pairs=pairs, manifest={})
    validate_relations(corpus)
    return corpus
