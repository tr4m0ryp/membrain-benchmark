"""Inventory of the session store: how much is there, and what KIND of thing is it.

The second question is the one that mattered. The paper's external-validity
paragraph promises 114 sessions of real work; the store holds nearly a thousand
files, and the inventory is what establishes that almost all of them are the same
headless single-turn genre rather than interactive development. A count alone
would have hidden that, so `genres` is reported beside every size statistic and
the substantive-session count is defined per genre rather than per file.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from brain.corpus.transcripts.extract import iter_session_paths, read_session
from brain.corpus.transcripts.types import SessionRecord

__all__ = ["Inventory", "take_inventory", "summarize"]

#: Genres that carry a work event an episode can be cut from.
EPISODE_BEARING: frozenset[str] = frozenset({"commit_subject"})


@dataclass
class Inventory:
    root: str
    sessions: list[SessionRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sessions)

    @property
    def genres(self) -> dict[str, int]:
        return dict(sorted(Counter(s.genre for s in self.sessions).items()))

    @property
    def episode_bearing(self) -> list[SessionRecord]:
        return [s for s in self.sessions if s.genre in EPISODE_BEARING]

    def date_range(self) -> tuple[str, str]:
        stamps = sorted(s.started_at for s in self.sessions if s.started_at)
        return (stamps[0], stamps[-1]) if stamps else ("", "")

    def to_dict(self) -> dict[str, Any]:
        return summarize(self)


def take_inventory(root: Path, *, limit: int | None = None) -> Inventory:
    """Read every session under `root`. Deterministic order, sorted by id."""
    inventory = Inventory(root=str(root))
    for index, path in enumerate(iter_session_paths(root)):
        if limit is not None and index >= limit:
            break
        inventory.sessions.append(read_session(path))
    return inventory


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(float(statistics.median(values)), 3),
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
    }


def summarize(inventory: Inventory) -> dict[str, Any]:
    """Counts, sizes, dates and genres. No transcript content of any kind."""
    sessions = inventory.sessions
    first, last = inventory.date_range()
    cwds = Counter(s.cwd for s in sessions if s.cwd)
    branches = Counter(s.git_branch for s in sessions if s.git_branch)
    return {
        "root": inventory.root,
        "n_sessions": len(sessions),
        "n_bytes": sum(s.n_bytes for s in sessions),
        "date_first": first,
        "date_last": last,
        "genres": inventory.genres,
        "n_episode_bearing": len(inventory.episode_bearing),
        "n_multi_turn": sum(1 for s in sessions if s.n_user_turns > 1),
        "n_with_tool_use": sum(1 for s in sessions if s.n_tool_uses > 0),
        "total_tool_uses": sum(s.n_tool_uses for s in sessions),
        "bytes_per_session": _stats([s.n_bytes for s in sessions]),
        "lines_per_session": _stats([s.n_lines for s in sessions]),
        "user_turns_per_session": _stats([s.n_user_turns for s in sessions]),
        "prompt_chars": _stats([len(s.prompt) for s in sessions]),
        "distinct_cwd": len(cwds),
        "distinct_branches": len(branches),
        "top_branches": dict(branches.most_common(5)),
    }
