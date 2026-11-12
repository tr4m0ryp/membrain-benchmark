"""Cut candidate episodes out of a session.

An episode is one changed FILE inside one recorded work event, not one session.
A commit that edits three files asserts three things about three referents, and
collapsing them into one record would destroy exactly the referent structure the
four write classes are defined over.

Each episode is rendered in both regimes from the same underlying event, mirroring
`brain.corpus.relations.generate`:

    short   "<path>: <subject>"                  terse assertion, slot dominates
    long    subject, section, then the added prose   session note, context dominates

Both are capped at `MAX_EPISODE_WORDS`, the cap the frozen synthetic corpus uses,
because the shared encoder truncates at 256 wordpieces and a truncated tail would
silently delete the distinguishing content. Using a different cap here would make
every real-versus-synthetic length comparison meaningless.

Nothing in this module labels anything. `referent` is an extraction, and the pair
miner is forbidden from turning it into a class.
"""

from __future__ import annotations

import re
from typing import Iterator

from brain.corpus.transcripts.types import REGIMES, CandidateEpisode, SessionRecord
from brain.corpus.validate import MAX_EPISODE_WORDS

__all__ = ["MAX_EPISODE_WORDS", "MIN_EPISODE_WORDS", "FileChange", "parse_diff", "episodes_of"]

#: Below this an episode is a whitespace or version-bump change carrying no
#: assertion a labeller could reason about. Fixed a priori from the diff shape,
#: never tuned against a downstream number.
MIN_EPISODE_WORDS = 12

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)\s*$", re.M)
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@\s*(?P<ctx>.*)$")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+(?P<md>.+?)\s*#*|(?:def|class)\s+(?P<py>\w+))\s*$")
_SUBJECT = re.compile(r"^[a-z]+(?:\([^)]+\))?:\s+.+$")


class FileChange:
    """One `diff --git` block, reduced to prose."""

    __slots__ = ("path", "section", "added", "removed")

    def __init__(self, path: str, section: str, added: list[str], removed: list[str]) -> None:
        self.path = path
        self.section = section
        self.added = added
        self.removed = removed

    @property
    def prose(self) -> str:
        return " ".join(self.added)


def _clean(line: str) -> str:
    return line.rstrip().replace("\t", " ").strip()


def parse_diff(diff_text: str) -> list[FileChange]:
    """Split a unified diff into per-file changes, added lines as plain prose."""
    headers = list(_DIFF_HEADER.finditer(diff_text))
    changes: list[FileChange] = []
    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(diff_text)
        body = diff_text[start:end]
        added: list[str] = []
        removed: list[str] = []
        section = ""
        for raw in body.splitlines():
            hunk = _HUNK.match(raw)
            if hunk:
                context = _clean(hunk.group("ctx"))
                if context and not section:
                    section = context
                continue
            if raw.startswith("+++") or raw.startswith("---"):
                continue
            if raw.startswith("+"):
                text = _clean(raw[1:])
                if text:
                    added.append(text)
                    if not section:
                        match = _HEADING.match(text)
                        if match:
                            section = match.group("md") or match.group("py") or ""
            elif raw.startswith("-"):
                text = _clean(raw[1:])
                if text:
                    removed.append(text)
        changes.append(FileChange(header.group("b"), section[:120], added, removed))
    return changes


def _cap(words: list[str]) -> str:
    return " ".join(words[:MAX_EPISODE_WORDS])


def _subject_of(answer: str) -> str:
    """The generated commit subject, or the first plausible line of the answer."""
    for line in answer.splitlines():
        candidate = line.strip().strip("`").strip()
        if candidate and _SUBJECT.match(candidate):
            return candidate[:120]
    for line in answer.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:120]
    return ""


def episodes_of(session: SessionRecord) -> Iterator[CandidateEpisode]:
    """Yield one episode per changed file per regime. Order is deterministic."""
    if session.genre != "commit_subject" or not session.prompt:
        return
    subject = _subject_of(session.answer)
    for index, change in enumerate(parse_diff(session.prompt)):
        if not change.added and not change.removed:
            continue
        referent = {"entity": change.path, "attribute": change.section}
        head = f"{change.path}: {subject}" if subject else change.path
        long_words = f"{head}. {change.section + '. ' if change.section else ''}{change.prose}".split()
        if len(long_words) < MIN_EPISODE_WORDS:
            continue
        bodies = {"short": head, "long": _cap(long_words)}
        for regime in REGIMES:
            text = bodies[regime]
            yield CandidateEpisode(
                episode_id=f"{session.session_id}:{index:02d}:{regime}",
                session_id=session.session_id,
                regime=regime,
                timestamp=session.started_at,
                text=text,
                subject=subject,
                referent=referent,
                n_words=len(text.split()),
                n_added_lines=len(change.added),
                n_removed_lines=len(change.removed),
            )
