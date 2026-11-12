"""Probe cases built from the frozen relational corpus.

The corpus already carries the labels this probe needs, and its action map already
states what a correct system should do, so no new judgement is introduced here:

    INTERFERENCE -> keep both, mark siblings, never merge

A probe case is one fresh store and one ordered sequence of writes. Each
INTERFERENCE pair yields two cases:

* the **pair** case, writing the two confusable episodes; and
* the **control** case, writing the first episode and then an episode from a
  different workstream entirely.

The control is what makes the third-branch question answerable. "Both records
are still there" is true of an unrelated write as well, so a system only has an
affirmative keep-both operation if the two cases leave different traces. Without
the control there is nothing to compare a residual against.

The control episode is drawn deterministically from the same regime, so the two
cases are matched on text length and template, and differ only in whether the
second write collides with the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from brain.config import SEED, get_rng

__all__ = ["ProbeCase", "CasePlan", "build_case_plan", "INTERFERENCE"]

INTERFERENCE = "INTERFERENCE"

#: One stream, so the control choice is reproducible and independent of any
#: other randomness in the project.
_RNG_STREAM = "incumbents.control"


@dataclass(frozen=True, slots=True)
class ProbeCase:
    """One store, one ordered write sequence."""

    case_id: str
    role: str  # "pair" or "control"
    pair_id: str
    cls: str
    regime: str
    cluster_id: str
    writes: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "pair_id": self.pair_id,
            "cls": self.cls,
            "regime": self.regime,
            "cluster_id": self.cluster_id,
            "writes": [dict(write) for write in self.writes],
        }


@dataclass(frozen=True, slots=True)
class CasePlan:
    """Every case, plus the pair-to-control map the scoring needs."""

    cases: tuple[ProbeCase, ...]
    control_of: dict[str, str] = field(default_factory=dict)
    corpus_sha256: str = ""

    def by_id(self) -> dict[str, ProbeCase]:
        return {case.case_id: case for case in self.cases}

    def pairs(self) -> list[ProbeCase]:
        return [case for case in self.cases if case.role == "pair"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "control_of": dict(self.control_of),
            "corpus_sha256": self.corpus_sha256,
            "control_selection": (
                "uniform over same-regime episodes whose cluster differs from both "
                f"members of the pair; get_rng({SEED}, stream={_RNG_STREAM!r})"
            ),
        }


def _texts(episodes: Iterable[Any]) -> dict[str, Any]:
    return {str(episode.trace_id): episode for episode in episodes}


def build_case_plan(
    corpus: Any,
    *,
    cls: str = INTERFERENCE,
    regimes: Sequence[str] = ("short", "long"),
    limit: int | None = None,
) -> CasePlan:
    """Pair and control cases for every `cls` pair in `regimes`, id-ordered."""
    episodes = _texts(corpus.episodes)
    by_regime: dict[str, list[Any]] = {}
    for episode in sorted(corpus.episodes, key=lambda e: str(e.trace_id)):
        by_regime.setdefault(str(episode.regime), []).append(episode)

    selected = [
        pair
        for pair in sorted(corpus.pairs, key=lambda p: str(p.pair_id))
        if str(pair.cls) == cls and str(pair.regime) in set(regimes)
    ]
    if limit is not None:
        selected = selected[: int(limit)]

    rng = get_rng(SEED, stream=_RNG_STREAM)
    cases: list[ProbeCase] = []
    control_of: dict[str, str] = {}
    for pair in selected:
        first, second = episodes[str(pair.a)], episodes[str(pair.b)]
        candidates = [
            episode
            for episode in by_regime.get(str(pair.regime), [])
            if str(episode.cluster_id) not in {str(first.cluster_id), str(second.cluster_id)}
        ]
        if not candidates:
            raise ValueError(f"no cross-cluster control available for pair {pair.pair_id}")
        control = candidates[int(rng.integers(0, len(candidates)))]

        pair_case = ProbeCase(
            case_id=f"pair-{pair.pair_id}",
            role="pair",
            pair_id=str(pair.pair_id),
            cls=str(pair.cls),
            regime=str(pair.regime),
            cluster_id=str(pair.cluster_id),
            writes=(_write(first), _write(second)),
        )
        control_case = ProbeCase(
            case_id=f"ctrl-{pair.pair_id}",
            role="control",
            pair_id=str(pair.pair_id),
            cls="CONTROL",
            regime=str(pair.regime),
            cluster_id=str(pair.cluster_id),
            writes=(_write(first), _write(control)),
        )
        cases.extend((pair_case, control_case))
        control_of[pair_case.case_id] = control_case.case_id

    return CasePlan(
        cases=tuple(cases),
        control_of=control_of,
        corpus_sha256=str(corpus.manifest.get("content_sha256", "")),
    )


def _write(episode: Any) -> dict[str, str]:
    return {"trace_id": str(episode.trace_id), "text": str(episode.text)}
