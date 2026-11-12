"""How a consolidated store is written: the merged episode, and what it replaces.

Split out of the policy so the two questions a reviewer will ask about the
destructive path can be answered on one page: what does the replacement episode
carry, and what exactly happens to the originals.

`retire_members` closes the members' validity interval rather than deleting
rows. Every `valid_only` read (`knn`, `all_traces`, `count`) then excludes them
exactly as a delete would, so the retrieval behaviour is the destructive policy's
behaviour, while the store's append-only contract and its evidence trail survive.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from brain.consolidate.prompt import PROMPT_VERSION
from brain.consolidate.types import MergeGroup
from brain.store import EpisodicStore, Trace

__all__ = [
    "merged_trace",
    "write_merges",
    "retire_members",
    "count_collapsed",
    "replacement_map",
]


def merged_trace(group: MergeGroup, members: Sequence[Trace], text: str) -> Trace:
    """The consolidated episode. Its id is a function of the group, not the text.

    Timestamps span the group (`valid_at` from the earliest member, `ts` from the
    latest) because the consolidated memory stands for all of them. Session and
    project are kept only when the group agrees on them, since a merged episode
    cannot honestly claim one scope while spanning two. A group crossing corpus
    clusters takes the smallest cluster id and is flagged on the `MergeGroup`, so
    a cluster-level bootstrap downstream still has a label and can see where that
    label is approximate.
    """
    key = hashlib.sha256(
        "|".join([PROMPT_VERSION, *group.member_ids]).encode("utf-8")
    ).hexdigest()
    sessions = {member.session for member in members}
    projects = {member.project for member in members}
    clusters = sorted({m.cluster_id for m in members if m.cluster_id})
    return Trace(
        text=text,
        id=f"m_{key[:16]}",
        session=sessions.pop() if len(sessions) == 1 else "",
        ts=max(float(member.ts) for member in members),
        project=projects.pop() if len(projects) == 1 else "",
        salience=max(float(member.salience) for member in members),
        valid_at=min(
            float(m.valid_at if m.valid_at is not None else m.ts) for m in members
        ),
        cluster_id=clusters[0] if clusters else None,
    )


def write_merges(
    store: EpisodicStore, merged: Sequence[tuple[MergeGroup, Trace]], vectors: Any
) -> list[str]:
    """Add the consolidated episodes; returns the ids the store actually kept.

    The returned id differs from the trace's own id exactly when the merge text
    was byte-identical to an episode already stored, because `content_sha` is
    UNIQUE. The caller counts those rather than letting the store shrink quietly.
    """
    if not merged:
        return []
    traces = [trace for _, trace in merged]
    return store.add_many(traces, vectors.matrix([trace.text for trace in traces]))


def retire_members(
    store: EpisodicStore,
    merged: Sequence[tuple[MergeGroup, Trace]],
    written: Sequence[str],
) -> tuple[list[str], int]:
    """Close each group's members at the merged episode's timestamp.

    Returns `(retired ids, collapsed merges)`. A member whose id came back as the
    stored merge id is never retired: it IS the consolidated episode.
    """
    retired: list[str] = []
    collapsed = 0
    for (group, trace), actual in zip(merged, written):
        if actual != trace.id:
            collapsed += 1
        for member in group.member_ids:
            if member == actual:
                continue
            store.invalidate(member, float(trace.ts))
            retired.append(member)
    return retired, collapsed


def count_collapsed(
    merged: Sequence[tuple[MergeGroup, Trace]], written: Sequence[str]
) -> int:
    """Collapsed merges only, for the non-destructive path which retires nothing."""
    return sum(1 for (_, trace), actual in zip(merged, written) if actual != trace.id)


def replacement_map(
    merged: Sequence[tuple[MergeGroup, Trace]], written: Sequence[str]
) -> dict[str, str]:
    """Original episode id -> the consolidated episode that now stands for it.

    Downstream accuracy metrics need this. Without it a query whose answer lived
    in a merged-away episode scores as a miss by definition, which would measure
    the id change rather than the information loss and would hand our mechanism
    a free win. Populated in both modes: the summary speaks for its members
    whether or not the members were retired.
    """
    return {
        member: actual
        for (group, _), actual in zip(merged, written)
        for member in group.member_ids
    }
