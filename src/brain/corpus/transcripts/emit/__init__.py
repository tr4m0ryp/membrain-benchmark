"""Artifact writing and report rendering."""

from __future__ import annotations

from brain.corpus.transcripts.emit.artifacts import (
    EPISODES_FILE,
    MANIFEST_FILE,
    PAIRS_FILE,
    QUARANTINE_DIR,
    TRANSCRIPTS_DIR,
    NotIgnored,
    assert_ignored,
    git_identity_literals,
    write_artifacts,
    write_json,
    write_jsonl,
)
from brain.corpus.transcripts.emit.report import render_markdown

__all__ = [
    "EPISODES_FILE", "MANIFEST_FILE", "PAIRS_FILE", "QUARANTINE_DIR", "TRANSCRIPTS_DIR",
    "NotIgnored", "assert_ignored", "git_identity_literals",
    "write_artifacts", "write_json", "write_jsonl", "render_markdown",
]
