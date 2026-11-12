"""Record types for the frozen synthetic interference corpus.

The corpus is the instrument the gate experiment rests on, so the records are
immutable and their serialization is canonical: byte-identical input must give a
byte-identical file, otherwise the content hash in the manifest is meaningless.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "SCHEMA_VERSION",
    "Episode",
    "Query",
    "Corpus",
    "canonical_line",
    "serialize_records",
]

#: Bumped whenever the on-disk record shape changes. Recorded in the manifest.
SCHEMA_VERSION = "corpus-v1"


@dataclass(frozen=True, slots=True)
class Episode:
    """One coding-session episode: a member of exactly one interference cluster."""

    trace_id: str
    cluster_id: str
    cluster_size: int
    date: str
    text: str
    topic: dict[str, str] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    template_variant: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Episode":
        return cls(
            trace_id=str(raw["trace_id"]),
            cluster_id=str(raw["cluster_id"]),
            cluster_size=int(raw["cluster_size"]),
            date=str(raw["date"]),
            text=str(raw["text"]),
            topic=dict(raw.get("topic", {})),
            facts=dict(raw.get("facts", {})),
            template_variant=int(raw.get("template_variant", 0)),
        )


@dataclass(frozen=True, slots=True)
class Query:
    """A question answerable by exactly one episode of a cluster.

    `distractor_ids` are that episode's siblings: same file, same subsystem, same
    error family, different event. Disambiguation accuracy is the fraction of
    queries where the correct trace outranks all of them.
    """

    query_id: str
    cluster_id: str
    cluster_size: int
    text: str
    correct_trace_id: str
    distractor_ids: list[str]
    cue_axis: str
    cue_value_id: str
    answer_axes: list[str]
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Query":
        return cls(
            query_id=str(raw["query_id"]),
            cluster_id=str(raw["cluster_id"]),
            cluster_size=int(raw["cluster_size"]),
            text=str(raw["text"]),
            correct_trace_id=str(raw["correct_trace_id"]),
            distractor_ids=[str(x) for x in raw["distractor_ids"]],
            cue_axis=str(raw["cue_axis"]),
            cue_value_id=str(raw["cue_value_id"]),
            answer_axes=[str(x) for x in raw["answer_axes"]],
            answer=str(raw["answer"]),
        )


@dataclass(frozen=True, slots=True)
class Corpus:
    """Episodes plus queries plus the manifest they were frozen with."""

    episodes: list[Episode]
    queries: list[Query]
    manifest: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def content_hash(self) -> str:
        return str(self.manifest.get("content_sha256", ""))

    def episode_by_id(self) -> dict[str, Episode]:
        return {e.trace_id: e for e in self.episodes}

    def clusters(self) -> dict[str, list[Episode]]:
        """Cluster id -> its episodes, both levels in sorted id order.

        Sorted on purpose: dict iteration order over ids is the one thing the
        seed cannot pin (rules.md, determinism addendum).
        """
        out: dict[str, list[Episode]] = {}
        for episode in sorted(self.episodes, key=lambda e: e.trace_id):
            out.setdefault(episode.cluster_id, []).append(episode)
        return {cid: out[cid] for cid in sorted(out)}

    def cluster_sizes(self) -> dict[str, int]:
        return {cid: len(eps) for cid, eps in self.clusters().items()}

    def queries_by_cluster_size(self) -> dict[int, list[Query]]:
        """Grouping for the accuracy-vs-cluster-size curve (Formal Model M-A)."""
        out: dict[int, list[Query]] = {}
        for query in sorted(self.queries, key=lambda q: q.query_id):
            out.setdefault(query.cluster_size, []).append(query)
        return {size: out[size] for size in sorted(out)}


def canonical_line(payload: Mapping[str, Any]) -> str:
    """One JSONL line in canonical form: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def serialize_records(records: Iterable[Any], *, key: str) -> bytes:
    """Serialize records to canonical JSONL bytes, ordered by `key`.

    Ordering by an explicit id (not by insertion) is what makes the content hash
    a property of the corpus rather than of the generation loop.
    """
    payloads = [record.to_dict() for record in records]
    payloads.sort(key=lambda p: str(p[key]))
    return "".join(canonical_line(p) for p in payloads).encode("utf-8")
