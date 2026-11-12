"""Deterministic generator for the synthetic interference corpus.

Design contract (spec: Evaluation Plan, F2, Formal Model M-A):

- A cluster shares ONE topic exactly. Siblings are near-duplicates on the
  surface and differ only on the discriminative axes (trigger, diagnosis,
  outcome, date, artifact), each sampled WITHOUT replacement inside a cluster.
  That gives genuinely different events, never a paraphrase of one event.
- Ground truth is therefore free: the query built from episode e's cue can only
  be answered by e, and its siblings are the distractors by construction.
- Cluster sizes VARY, with the cluster count per size chosen so every size
  contributes the same number of episodes. The accuracy-vs-cluster-size curve
  (M-A's headline prediction) then has comparable precision at every point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from brain.config import SEED, SMOKE_TEST, get_rng
from brain.corpus.content import (
    CUE_AXES,
    DIAGNOSES,
    N_SHARED_MAX,
    OUTCOMES,
    TOPICS,
    TRIGGERS,
    VARIANTS,
    render_answer,
    render_episode,
    render_query,
)
from brain.corpus.types import Corpus, Episode, Query
from brain.corpus.validate import validate_corpus

__all__ = [
    "GeneratorConfig",
    "generate_corpus",
    "default_config",
]

_EPOCH = date(2026, 1, 5)
_DATE_SPAN = 240
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Every knob that affects corpus content. Serialized into the manifest."""

    seed: int = SEED
    similarity: float = 0.7
    cluster_sizes: tuple[int, ...] = (2, 3, 4, 6, 8)
    clusters_per_size: tuple[int, ...] = (12, 8, 6, 4, 3)
    queries_per_episode: int = 3
    generator: str = "template-v1"
    model: str | None = None
    label: str = "gate"

    def __post_init__(self) -> None:
        if len(self.cluster_sizes) != len(self.clusters_per_size):
            raise ValueError("cluster_sizes and clusters_per_size must have equal length")
        if min(self.cluster_sizes) < 2:
            raise ValueError("a cluster of size 1 has no distractors; minimum size is 2")
        if max(self.cluster_sizes) > min(len(TRIGGERS), len(DIAGNOSES), len(OUTCOMES)):
            raise ValueError("cluster size exceeds an axis bank; siblings could not stay distinct")
        if self.n_clusters > len(TOPICS):
            raise ValueError(f"need {self.n_clusters} topics, bank holds {len(TOPICS)}")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("similarity must lie in [0, 1]")
        if not 1 <= self.queries_per_episode <= len(CUE_AXES):
            raise ValueError(f"queries_per_episode must lie in [1, {len(CUE_AXES)}]")

    @property
    def n_clusters(self) -> int:
        return sum(self.clusters_per_size)

    @property
    def n_episodes(self) -> int:
        return sum(s * c for s, c in zip(self.cluster_sizes, self.clusters_per_size))

    @property
    def n_shared_sentences(self) -> int:
        """Shared-sentence budget. Monotone in `similarity`, floor 1, cap N_SHARED_MAX."""
        return 1 + int((N_SHARED_MAX - 1) * self.similarity + 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "similarity": self.similarity,
            "cluster_sizes": list(self.cluster_sizes),
            "clusters_per_size": list(self.clusters_per_size),
            "queries_per_episode": self.queries_per_episode,
            "generator": self.generator,
            "model": self.model,
            "label": self.label,
            "n_clusters": self.n_clusters,
            "n_episodes": self.n_episodes,
            "n_shared_sentences": self.n_shared_sentences,
        }


def default_config(*, smoke: bool | None = None, **overrides: Any) -> GeneratorConfig:
    """Gate config by default; the tiny config when SMOKE_TEST is on."""
    use_smoke = SMOKE_TEST if smoke is None else smoke
    base: dict[str, Any] = (
        {"cluster_sizes": (3,), "clusters_per_size": (2,), "label": "smoke"}
        if use_smoke
        else {}
    )
    base.update(overrides)
    return GeneratorConfig(**base)


@dataclass(slots=True)
class _ClusterPlan:
    cluster_id: str
    topic_index: int
    size: int
    fields: dict[str, Any] = field(default_factory=dict)


def _plan_clusters(config: GeneratorConfig, rng: np.random.Generator) -> list[_ClusterPlan]:
    """Assign a distinct topic and a size to every cluster."""
    sizes: list[int] = []
    for size, count in zip(config.cluster_sizes, config.clusters_per_size):
        sizes.extend([int(size)] * int(count))
    order = rng.permutation(len(sizes))
    topic_choice = rng.choice(len(TOPICS), size=len(sizes), replace=False)
    return [
        _ClusterPlan(f"c{index:03d}", int(topic_choice[index]), sizes[int(position)])
        for index, position in enumerate(order)
    ]


def _slug(text: str) -> str:
    return _NON_ALNUM.sub("-", text.lower()).strip("-")


def _outcome_text(template: str, rng: np.random.Generator) -> str:
    """Fill the numeric slots. Draws happen unconditionally so the RNG stream
    advances identically whichever template was picked."""
    slots = {
        "a": int(rng.integers(2, 100)),
        "b": int(rng.integers(0, 10)),
        "c": int(rng.integers(10, 991)),
        "d": int(rng.integers(1, 10)),
        "e": int(rng.integers(1, 10)),
        "m": int(rng.integers(3, 91)),
    }
    return template.format(**slots)


def _build_cluster(
    plan: _ClusterPlan, config: GeneratorConfig, rng: np.random.Generator
) -> tuple[list[Episode], list[Query]]:
    topic = TOPICS[plan.topic_index]
    size = plan.size
    triggers = [TRIGGERS[i] for i in rng.choice(len(TRIGGERS), size=size, replace=False)]
    diagnoses = [DIAGNOSES[i] for i in rng.choice(len(DIAGNOSES), size=size, replace=False)]
    outcomes = [OUTCOMES[i] for i in rng.choice(len(OUTCOMES), size=size, replace=False)]
    day_offsets = sorted(int(x) for x in rng.choice(_DATE_SPAN, size=size, replace=False))
    pull_ids = sorted(int(x) for x in rng.choice(9000, size=size, replace=False))
    base_variant = int(rng.integers(0, len(VARIANTS)))

    episodes: list[Episode] = []
    queries: list[Query] = []
    trace_ids = [f"{plan.cluster_id}-e{index}" for index in range(size)]
    for index in range(size):
        trigger, diagnosis, outcome = triggers[index], diagnoses[index], outcomes[index]
        outcome_text = _outcome_text(outcome.episode, rng)
        # Siblings keep the cluster scaffold with probability `similarity`, so the
        # knob controls shared PHRASING as well as shared topic sentences.
        variant = base_variant if rng.random() < config.similarity else int(rng.integers(0, len(VARIANTS)))
        day = _EPOCH + timedelta(days=day_offsets[index])
        artifact = f"PR #{1000 + pull_ids[index]} on branch fix/{_slug(topic.symptom)}-{index + 1}"
        episodes.append(
            Episode(
                trace_id=trace_ids[index],
                cluster_id=plan.cluster_id,
                cluster_size=size,
                date=day.isoformat(),
                text=render_episode(
                    topic=topic,
                    date=day.isoformat(),
                    trigger=trigger,
                    diagnosis=diagnosis,
                    outcome=outcome_text,
                    artifact=artifact,
                    variant=variant,
                    n_shared=config.n_shared_sentences,
                ),
                topic={
                    "topic_id": topic.topic_id,
                    "path": topic.path,
                    "subsystem": topic.subsystem,
                    "component": topic.component,
                    "symptom": topic.symptom,
                    "language": topic.language,
                },
                facts={
                    "trigger": trigger.value_id,
                    "diagnosis": diagnosis.value_id,
                    "outcome": outcome.value_id,
                    "outcome_text": outcome_text,
                    "artifact": artifact,
                },
                template_variant=variant,
            )
        )
        distractors = sorted(tid for tid in trace_ids if tid != trace_ids[index])
        cues = {
            "trigger": (trigger.cue, trigger.value_id),
            "outcome": (outcome.cue, outcome.value_id),
            "cause": (diagnosis.cause_cue, diagnosis.value_id),
        }
        for kind, (cue_axis, answer_axes) in enumerate(CUE_AXES[: config.queries_per_episode]):
            cue_text, cue_value_id = cues[cue_axis]
            queries.append(
                Query(
                    query_id=f"{trace_ids[index]}-q{kind}",
                    cluster_id=plan.cluster_id,
                    cluster_size=size,
                    text=render_query(topic=topic, cue_axis=cue_axis, cue=cue_text),
                    correct_trace_id=trace_ids[index],
                    distractor_ids=distractors,
                    cue_axis=cue_axis,
                    cue_value_id=cue_value_id,
                    answer_axes=list(answer_axes),
                    answer=render_answer(
                        cue_axis=cue_axis,
                        trigger=trigger,
                        diagnosis=diagnosis,
                        outcome=outcome_text,
                    ),
                )
            )
    return episodes, queries


def generate_corpus(config: GeneratorConfig | None = None) -> Corpus:
    """Generate the corpus. Same config plus same seed gives identical records."""
    config = config or default_config()
    rng = get_rng(config.seed, stream="corpus")
    episodes: list[Episode] = []
    queries: list[Query] = []
    for plan in _plan_clusters(config, rng):
        cluster_episodes, cluster_queries = _build_cluster(plan, config, rng)
        episodes.extend(cluster_episodes)
        queries.extend(cluster_queries)
    episodes.sort(key=lambda e: e.trace_id)
    queries.sort(key=lambda q: q.query_id)
    corpus = Corpus(episodes=episodes, queries=queries, manifest={})
    validate_corpus(corpus)
    return corpus
