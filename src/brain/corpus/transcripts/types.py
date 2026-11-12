"""Records for the transcript-derived CANDIDATE corpus.

Nothing in this module carries a ground-truth label. That is the point. The
frozen synthetic split (`brain.corpus.relations.types`) derives its class from
the action map applied to `(rho, nu)` read off a generator plan; here there is no
plan, so `rho` and `nu` are unknown and a class cannot be computed. A
`CandidatePair` therefore has `label = None` and a `stratum`, which is a sampling
bucket used to spread a human's review budget across the four classes, never a
prediction of the class.

Two safety invariants hold over every record here:

1. A `ScanFinding` carries a category and a count and never the matched text.
   The scanner deliberately has no code path that can place a detected value on
   any record, in any log line, or in any file.
2. Every string field of a `CandidateEpisode` is post-pseudonymisation. The
   inverse map lives in one gitignored file that these records never reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

__all__ = [
    "SCHEMA_VERSION", "REGIMES", "STRATA", "SEVERITIES",
    "SessionRecord", "CandidateEpisode", "CandidatePair", "ScanFinding", "ScanReport",
]

#: Bumped whenever the on-disk record shape changes. Recorded in the manifest.
SCHEMA_VERSION = "corpus-transcripts-candidate-v1"

#: Matches `brain.corpus.relations.types.REGIMES`. `short` is the commit subject
#: alone (a terse assertion), `long` is the subject plus the changed prose (a
#: multi-sentence session note). Both are rendered from the SAME underlying work
#: event, so the regime comparison is paired exactly as it is in the synthetic
#: split.
REGIMES: tuple[str, ...] = ("long", "short")

#: Review strata. Sampling buckets for the labelling budget, NOT class guesses.
#: `same_entity` is named for what it buckets on, a shared file path, rather than
#: for rho, which is unknown; it is where DUPLICATE and SUPERSESSION concentrate
#: because both require rho = 1. `near_miss` is where INTERFERENCE concentrates.
#: `distant` is the COINCIDENCE control that keeps the set from being all-positive.
STRATA: tuple[str, ...] = ("same_entity", "near_miss", "mid_band", "distant")

#: `block` quarantines the episode outright, `flag` quarantines it pending human
#: review, `redact` is handled by pseudonymisation and never quarantines.
SEVERITIES: tuple[str, ...] = ("block", "flag", "redact")


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One transcript file, reduced to what the extractor needs.

    `prompt` and `answer` are raw at construction time and are scrubbed before
    anything derived from them is written. `genre` records what kind of session
    this is, because the store turned out to be far less varied than the paper
    assumed and that fact has to survive into the report.
    """

    session_id: str
    genre: str
    started_at: str
    ended_at: str
    n_lines: int
    n_bytes: int
    n_user_turns: int
    n_tool_uses: int
    cwd: str = ""
    git_branch: str = ""
    prompt: str = ""
    answer: str = ""

    @property
    def substantive(self) -> bool:
        """Does this session carry a work event an episode could be cut from?"""
        return bool(self.prompt) and self.genre != "unknown"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("prompt", None)
        row.pop("answer", None)
        return row


@dataclass(frozen=True, slots=True)
class CandidateEpisode:
    """One unit of work: one changed file inside one recorded commit event.

    `referent` is the analogue of the synthetic split's `slot`: `entity` is the
    repository path the change lands on, `attribute` is the section or symbol
    inside it. It is an EXTRACTED guess, not ground truth, which is why the pair
    records refuse to compute a class from it.
    """

    episode_id: str
    session_id: str
    regime: str
    timestamp: str
    text: str
    subject: str = ""
    referent: dict[str, str] = field(default_factory=dict)
    n_words: int = 0
    n_added_lines: int = 0
    n_removed_lines: int = 0
    scan_categories: tuple[str, ...] = ()

    @property
    def referent_id(self) -> str:
        return f"{self.referent.get('entity', '')}::{self.referent.get('attribute', '')}"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["scan_categories"] = list(self.scan_categories)
        return row

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateEpisode":
        return cls(
            episode_id=str(raw["episode_id"]),
            session_id=str(raw["session_id"]),
            regime=str(raw["regime"]),
            timestamp=str(raw["timestamp"]),
            text=str(raw["text"]),
            subject=str(raw.get("subject", "")),
            referent={str(k): str(v) for k, v in dict(raw.get("referent", {})).items()},
            n_words=int(raw.get("n_words", 0)),
            n_added_lines=int(raw.get("n_added_lines", 0)),
            n_removed_lines=int(raw.get("n_removed_lines", 0)),
            scan_categories=tuple(raw.get("scan_categories", ())),
        )


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """A pair proposed for human labelling. `label` is None and stays None.

    `a` precedes `b` in time, so "b supersedes a" is well defined if a labeller
    decides that is what it is. `cos` is the encoder similarity that put the pair
    in front of the labeller; it is recorded so the selection function is
    auditable, and it is exactly the quantity a label must not be derived from.
    """

    pair_id: str
    a: str
    b: str
    regime: str
    stratum: str
    cos: float
    same_referent: int
    same_entity: int
    days_apart: float
    label: str | None = None
    labelled_by: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidatePair":
        label = raw.get("label")
        return cls(
            pair_id=str(raw["pair_id"]),
            a=str(raw["a"]),
            b=str(raw["b"]),
            regime=str(raw["regime"]),
            stratum=str(raw["stratum"]),
            cos=float(raw["cos"]),
            same_referent=int(raw.get("same_referent", 0)),
            same_entity=int(raw.get("same_entity", 0)),
            days_apart=float(raw.get("days_apart", 0.0)),
            label=None if label is None else str(label),
            labelled_by=str(raw.get("labelled_by", "")),
            note=str(raw.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class ScanFinding:
    """A detector hit. Category and count only.

    There is intentionally no field that could hold the matched text, so no
    caller can serialize a secret by accident and no future edit can add one
    without changing this class.
    """

    category: str
    severity: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScanReport:
    """Aggregate scan outcome over a set of texts. Counts and categories only."""

    n_scanned: int
    n_clean: int
    n_blocked: int
    n_flagged: int
    n_redacted: int
    findings: tuple[ScanFinding, ...] = ()

    def by_category(self) -> dict[str, dict[str, int]]:
        return {
            f.category: {"severity_rank": SEVERITIES.index(f.severity), "count": f.count}
            for f in sorted(self.findings, key=lambda x: x.category)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_scanned": self.n_scanned,
            "n_clean": self.n_clean,
            "n_blocked": self.n_blocked,
            "n_flagged": self.n_flagged,
            "n_redacted": self.n_redacted,
            "findings": [f.to_dict() for f in sorted(self.findings, key=lambda x: x.category)],
        }
