"""Load the relational split, and refuse it if the bytes have drifted.

Mirrors `brain.corpus.loader`: the content hash covers `episodes.jsonl` and
`pairs.jsonl` only, manifest metadata is excluded so recomputing a diagnostic on
another machine cannot invalidate a frozen split, and verification is on by
default with failure as an exception rather than a warning.

The hash domain tag differs from the existing corpus's, so a relational file can
never be mistaken for a synthetic one even at identical bytes.

This module also pins the EXISTING frozen corpus. The relational split is
additive; if `data/corpus/synthetic/` ever moves, every result published against
it, including the kill and its theorem, loses its reproducibility claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from brain.config import CORPUS_DIR
from brain.corpus.loader import CorpusHashMismatch, CorpusNotFrozen
from brain.corpus.loader import content_hashes as synthetic_content_hashes
from brain.corpus.loader import load_corpus
from brain.corpus.relations.types import (
    LabelledPair,
    RelationCorpus,
    RelationEpisode,
    serialize_records,
)

__all__ = [
    "RELATIONS_DIR", "EPISODES_FILE", "PAIRS_FILE", "MANIFEST_FILE", "SPOTCHECK_FILE",
    "SYNTHETIC_CONTENT_SHA256", "content_hashes", "load_relations",
    "assert_synthetic_corpus_unchanged",
]

RELATIONS_DIR: Path = CORPUS_DIR / "relations"
EPISODES_FILE = "episodes.jsonl"
PAIRS_FILE = "pairs.jsonl"
MANIFEST_FILE = "manifest.json"
SPOTCHECK_FILE = "spotcheck.jsonl"

#: The existing frozen corpus, pinned. Every result to date is stated against it.
SYNTHETIC_CONTENT_SHA256 = "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb"

_HASH_DOMAIN = b"brain.corpus.relations.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_hashes(
    episodes: Iterable[RelationEpisode], pairs: Iterable[LabelledPair]
) -> dict[str, str]:
    """Hashes over the canonical serialization of the records themselves."""
    episode_bytes = serialize_records(episodes, key="trace_id")
    pair_bytes = serialize_records(pairs, key="pair_id")
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(b"\x00episodes\x00")
    digest.update(episode_bytes)
    digest.update(b"\x00pairs\x00")
    digest.update(pair_bytes)
    return {
        "episodes_sha256": _sha256(episode_bytes),
        "pairs_sha256": _sha256(pair_bytes),
        "content_sha256": digest.hexdigest(),
    }


def assert_synthetic_corpus_unchanged(directory: Path | str | None = None) -> str:
    """Recompute the existing corpus hash from disk and pin it. Returns the hash.

    Both the bytes and the value the manifest records are checked, so neither a
    silent edit nor a rewritten manifest can pass.
    """
    corpus = load_corpus(directory, verify=False)
    actual = synthetic_content_hashes(corpus.episodes, corpus.queries)["content_sha256"]
    recorded = str(corpus.manifest.get("content_sha256", ""))
    if actual != SYNTHETIC_CONTENT_SHA256 or recorded != SYNTHETIC_CONTENT_SHA256:
        raise CorpusHashMismatch(
            "the existing frozen corpus is NOT byte-identical: expected "
            f"{SYNTHETIC_CONTENT_SHA256}, manifest says {recorded}, bytes give {actual}"
        )
    return actual


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CorpusNotFrozen(f"missing corpus file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_relations(directory: Path | str | None = None, *, verify: bool = True) -> RelationCorpus:
    """Load the split and refuse it if the bytes no longer match the manifest."""
    root = Path(directory) if directory is not None else RELATIONS_DIR
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise CorpusNotFrozen(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = [RelationEpisode.from_dict(raw) for raw in _read_jsonl(root / EPISODES_FILE)]
    pairs = [LabelledPair.from_dict(raw) for raw in _read_jsonl(root / PAIRS_FILE)]
    episodes.sort(key=lambda e: e.trace_id)
    pairs.sort(key=lambda p: p.pair_id)
    if verify:
        _verify(episodes, pairs, manifest, root)
    return RelationCorpus(episodes=episodes, pairs=pairs, manifest=manifest)


def _verify(
    episodes: list[RelationEpisode],
    pairs: list[LabelledPair],
    manifest: dict[str, Any],
    root: Path,
) -> None:
    keys = ("episodes_sha256", "pairs_sha256", "content_sha256")
    recorded = {key: manifest.get(key) for key in keys}
    if not recorded.get("content_sha256"):
        raise CorpusHashMismatch(f"manifest at {root} carries no content hash")
    actual = content_hashes(episodes, pairs)
    differing = sorted(key for key, value in actual.items() if recorded.get(key) != value)
    if differing:
        raise CorpusHashMismatch(
            "relational corpus changed after freezing; differing hashes: "
            + ", ".join(f"{k} expected {recorded.get(k)} got {actual[k]}" for k in differing)
        )
    counts = manifest.get("counts", {})
    for name, found in (("episodes", len(episodes)), ("pairs", len(pairs))):
        if counts.get(name) is not None and counts[name] != found:
            raise CorpusHashMismatch(f"manifest says {counts[name]} {name}, found {found}")
