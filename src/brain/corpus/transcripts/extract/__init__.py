"""Transcript reading and episode extraction."""

from __future__ import annotations

from brain.corpus.transcripts.extract.episodes import (
    MAX_EPISODE_WORDS,
    MIN_EPISODE_WORDS,
    FileChange,
    episodes_of,
    parse_diff,
)
from brain.corpus.transcripts.extract.reader import (
    GENRE_PREFIXES,
    iter_session_paths,
    read_session,
    session_id_of,
)

__all__ = [
    "MAX_EPISODE_WORDS", "MIN_EPISODE_WORDS", "FileChange", "episodes_of", "parse_diff",
    "GENRE_PREFIXES", "iter_session_paths", "read_session", "session_id_of",
]
