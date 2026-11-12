"""Score a write-path dump: destruction, relation recording, indistinguishability.

Scoring is deliberately separated from execution. The worker produces a dump of
whatever the incumbent persisted and makes no judgement; everything judged lives
here, in this project's own environment, against the definitions frozen in the
pre-registration.

Three quantities, all per case:

* **destroyed** -- at least one of the two written episodes is no longer
  separately addressable. A record is attributable to an episode when its
  payload still carries that episode's `trace_id`; a merge that rewrites the
  record's payload therefore reads as destruction of the older episode, which is
  the intended reading, because the store can no longer say which event the text
  belongs to.
* **relation_recorded** -- some persisted field links the two records AND does
  not link the control's two records. A link that also appears between two
  unrelated writes records co-occurrence, not confusability, and does not count.
* **indistinguishable** -- the pair case and its control leave the same
  structural signature. The signature is content-blind by construction: it is
  exactly the five components named in the pre-registration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = ["CaseOutcome", "signature", "score_case", "link_edges"]

#: Payload keys that identify a record's source episode.
_TRACE_KEYS = ("trace_id",)
#: Payload keys carrying temporal validity, across the systems probed.
_TEMPORAL_KEYS = ("created_at", "updated_at", "valid_at", "invalid_at", "expired_at", "is_deleted")


def _payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return dict(record.get("payload") or {})


def _trace_of(record: Mapping[str, Any]) -> str | None:
    payload = _payload(record)
    for key in _TRACE_KEYS:
        value = payload.get(key)
        if value:
            return str(value)
    return None


def attributable(state: Mapping[str, Any], trace_id: str) -> list[str]:
    """Ids of the records still attributable to `trace_id`."""
    return [
        str(record.get("id"))
        for record in state.get("memories", [])
        if _trace_of(record) == str(trace_id)
    ]


def link_edges(state: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Every persisted memory-to-memory link, as an unordered id pair.

    Two sources are read. A record whose own payload names other record ids
    links to them directly. An auxiliary record (Mem0's entity store is the case
    in point) whose payload holds a list of memory ids links every pair in that
    list, because that is what the store can later traverse.
    """
    ids = {str(record.get("id")) for record in state.get("memories", [])}
    edges: set[tuple[str, str]] = set()

    def _named(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value in ids else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item) in ids]
        return []

    for record in state.get("memories", []):
        source = str(record.get("id"))
        for value in _payload(record).values():
            for target in _named(value):
                if target != source:
                    edges.add(tuple(sorted((source, target))))  # type: ignore[arg-type]

    for record in state.get("entities", []) or []:
        for value in _payload(record).values():
            members = sorted(set(_named(value)))
            for i, left in enumerate(members):
                for right in members[i + 1 :]:
                    edges.add((left, right))
    return edges


def signature(state: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-registered content-blind structural signature.

    Five components and no more: record count, the field names present per
    record, which of them are non-null, the link types persisted, and the
    temporal validity flags. Content is excluded because the pair case and the
    control differ in content by construction; only structure is under test.
    """
    memories = list(state.get("memories", []))
    fields: list[list[str]] = []
    nonnull: list[list[str]] = []
    temporal: list[list[str]] = []
    for record in sorted(memories, key=lambda item: str(item.get("id"))):
        payload = _payload(record)
        fields.append(sorted(payload))
        nonnull.append(sorted(key for key, value in payload.items() if value not in (None, "", [], {})))
        temporal.append(sorted(key for key in _TEMPORAL_KEYS if payload.get(key) not in (None, "")))
    link_types = sorted(
        {
            key
            for record in memories + list(state.get("entities", []) or [])
            for key, value in _payload(record).items()
            if isinstance(value, (list, tuple)) and value
        }
    )
    return {
        "n_memories": len(memories),
        "record_fields": sorted(fields),
        "record_nonnull_fields": sorted(nonnull),
        "link_types": link_types,
        "temporal_flags": sorted(temporal),
    }


def signature_key(state: Mapping[str, Any]) -> str:
    return json.dumps(signature(state), sort_keys=True)


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """One pair case scored against its control."""

    case_id: str
    pair_id: str
    regime: str
    cluster_id: str
    destroyed: bool
    both_attributable: bool
    n_memories: int
    relation_recorded: bool
    indistinguishable: bool
    neighbour_offered: bool
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pair_id": self.pair_id,
            "regime": self.regime,
            "cluster_id": self.cluster_id,
            "destroyed": bool(self.destroyed),
            "both_attributable": bool(self.both_attributable),
            "n_memories": int(self.n_memories),
            "relation_recorded": bool(self.relation_recorded),
            "indistinguishable": bool(self.indistinguishable),
            "neighbour_offered": bool(self.neighbour_offered),
            "detail": dict(self.detail),
        }


def _neighbour_offered(result: Mapping[str, Any]) -> bool:
    """Did the write path actually show the first episode to the judge?

    A keep-both decision that was never offered the neighbour would say nothing
    about the write path, so this is a validity condition rather than a metric.
    """
    judge = result.get("judge") or {}
    calls = (judge.get("write") or {}).get("calls") or []
    return any(call.get("existing_ids_offered") for call in calls[1:])


def score_case(
    pair_result: Mapping[str, Any],
    control_result: Mapping[str, Any],
    case: Mapping[str, Any],
) -> CaseOutcome:
    """Score one pair case against its matched control."""
    state = pair_result.get("state") or {}
    control_state = control_result.get("state") or {}
    traces = [str(write["trace_id"]) for write in case["writes"]]
    resident = {trace: attributable(state, trace) for trace in traces}
    both = all(resident[trace] for trace in traces)
    same_record = both and len({tuple(resident[trace]) for trace in traces}) == 1

    pair_edges = link_edges(state)
    control_edges = link_edges(control_state)
    # A link only records confusability if the unrelated write does not get one.
    distinguishing = bool(pair_edges) and not bool(control_edges)

    return CaseOutcome(
        case_id=str(case["case_id"]),
        pair_id=str(case["pair_id"]),
        regime=str(case["regime"]),
        cluster_id=str(case["cluster_id"]),
        destroyed=(not both) or same_record,
        both_attributable=both,
        n_memories=len(state.get("memories", [])),
        relation_recorded=distinguishing,
        indistinguishable=signature_key(state) == signature_key(control_state),
        neighbour_offered=_neighbour_offered(pair_result),
        detail={
            "resident_records": {trace: resident[trace] for trace in traces},
            "pair_link_edges": sorted(pair_edges),
            "control_link_edges": sorted(control_edges),
            "n_control_memories": len(control_state.get("memories", [])),
            "signature": signature(state),
            "control_signature": signature(control_state),
        },
    )


def group_by_cluster(outcomes: Iterable[CaseOutcome], attribute: str) -> dict[str, list[float]]:
    """Cluster id -> indicators, the shape `cluster_bootstrap_ci` consumes."""
    groups: dict[str, list[float]] = {}
    for outcome in outcomes:
        groups.setdefault(outcome.cluster_id, []).append(float(bool(getattr(outcome, attribute))))
    return {cluster: groups[cluster] for cluster in sorted(groups)}
