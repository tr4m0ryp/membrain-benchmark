"""Load the frozen corpus and refuse it if the bytes no longer match the manifest.

The gate experiment must never be able to attribute a number to a corpus that
has drifted since it was frozen, so verification is on by default and failure is
an exception, not a warning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from brain.config import CORPUS_DIR
from brain.corpus.types import Corpus, Episode, Query, serialize_records

__all__ = [
    "SYNTHETIC_DIR",
    "EPISODES_FILE",
    "QUERIES_FILE",
    "MANIFEST_FILE",
    "CorpusHashMismatch",
    "CorpusNotFrozen",
    "content_hashes",
    "load_corpus",
]

SYNTHETIC_DIR: Path = CORPUS_DIR / "synthetic"
EPISODES_FILE = "episodes.jsonl"
QUERIES_FILE = "queries.jsonl"
MANIFEST_FILE = "manifest.json"

_HASH_DOMAIN = b"brain.corpus.v1"


class CorpusNotFrozen(FileNotFoundError):
    """The corpus directory is missing or incomplete."""


class CorpusHashMismatch(RuntimeError):
    """On-disk content does not match the hash recorded in the manifest."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_hashes(episodes: Iterable[Episode], queries: Iterable[Query]) -> dict[str, str]:
    """Hashes over the canonical serialization of the records themselves.

    Only episodes and queries are covered. Manifest metadata (timestamps,
    measured similarity) is deliberately outside the hash: recomputing a
    diagnostic on another machine must not invalidate a frozen corpus.
    """
    episode_bytes = serialize_records(episodes, key="trace_id")
    query_bytes = serialize_records(queries, key="query_id")
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(b"\x00episodes\x00")
    digest.update(episode_bytes)
    digest.update(b"\x00queries\x00")
    digest.update(query_bytes)
    return {
        "episodes_sha256": _sha256(episode_bytes),
        "queries_sha256": _sha256(query_bytes),
        "content_sha256": digest.hexdigest(),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CorpusNotFrozen(f"missing corpus file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_corpus(directory: Path | str | None = None, *, verify: bool = True) -> Corpus:
    """Load episodes, queries and manifest; verify the content hash by default."""
    root = Path(directory) if directory is not None else SYNTHETIC_DIR
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise CorpusNotFrozen(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = [Episode.from_dict(raw) for raw in _read_jsonl(root / EPISODES_FILE)]
    queries = [Query.from_dict(raw) for raw in _read_jsonl(root / QUERIES_FILE)]
    episodes.sort(key=lambda e: e.trace_id)
    queries.sort(key=lambda q: q.query_id)
    if verify:
        _verify(episodes, queries, manifest, root)
    return Corpus(episodes=episodes, queries=queries, manifest=manifest)


def _verify(
    episodes: list[Episode], queries: list[Query], manifest: dict[str, Any], root: Path
) -> None:
    recorded = {key: manifest.get(key) for key in ("episodes_sha256", "queries_sha256", "content_sha256")}
    if not recorded.get("content_sha256"):
        raise CorpusHashMismatch(f"manifest at {root} carries no content hash")
    actual = content_hashes(episodes, queries)
    differing = sorted(key for key, value in actual.items() if recorded.get(key) != value)
    if differing:
        raise CorpusHashMismatch(
            "corpus content changed after freezing; differing hashes: "
            + ", ".join(f"{key} expected {recorded.get(key)} got {actual[key]}" for key in differing)
        )
    counts = manifest.get("counts", {})
    expected_episodes = counts.get("episodes")
    expected_queries = counts.get("queries")
    if expected_episodes is not None and expected_episodes != len(episodes):
        raise CorpusHashMismatch(f"manifest says {expected_episodes} episodes, found {len(episodes)}")
    if expected_queries is not None and expected_queries != len(queries):
        raise CorpusHashMismatch(f"manifest says {expected_queries} queries, found {len(queries)}")
