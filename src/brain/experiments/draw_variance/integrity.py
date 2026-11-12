"""The frozen-corpus tripwire: every draw is generated in memory, and this is
what proves the frozen bytes on disk never moved while that happened.

Reuses `directory_sha` (whole-directory content digest) from
`brain.experiments.gate_veto.report`, the same tripwire `run_gate_v1_v2.py`
used for task 007, plus the manifest-hash asserts each corpus package already
ships. Belt and suspenders: `generate_corpus` / `generate_relations` never
write to `CORPUS_DIR` on their own, so this should always pass; it is here so
a future change to those functions cannot silently start doing so unnoticed.
"""

from __future__ import annotations

from typing import Any

from brain.config import CORPUS_DIR
from brain.corpus.relations.freeze import RELATIONS_V1_CONTENT_SHA256, RELATIONS_V2_DIR
from brain.corpus.relations.loader import (
    RELATIONS_DIR,
    SYNTHETIC_CONTENT_SHA256,
    assert_synthetic_corpus_unchanged,
    load_relations,
)
from brain.experiments.gate_veto.report import directory_sha

__all__ = ["FROZEN_DIRS", "RELATIONS_V2_CONTENT_SHA256", "snapshot", "assert_unchanged"]

FROZEN_DIRS: dict[str, Any] = {
    "synthetic": CORPUS_DIR / "synthetic",
    "relations_v1": RELATIONS_DIR,
    "relations_v2": RELATIONS_V2_DIR,
}
#: Finding F25 / task 007. Checked the same way the
#: other two frozen corpora are: load with verification on, which raises on any
#: manifest/byte mismatch, then pin the digest to the value this project cites.
RELATIONS_V2_CONTENT_SHA256 = "24ff2689a02ef9ffa3d9bf4573431cef7be53d8e3413faac6dfdf63d2dbf7158"


def snapshot() -> dict[str, str]:
    """Whole-directory content hash for each frozen corpus, right now."""
    return {name: directory_sha(path) for name, path in FROZEN_DIRS.items()}


def assert_unchanged(before: dict[str, str]) -> dict[str, Any]:
    """Raise if any frozen directory's bytes moved since `before` was taken.

    Also re-verifies the two manifest-level content hashes this project
    already ships checks for, so a drift is caught two independent ways.
    """
    after = snapshot()
    if after != before:
        changed = {k: (before.get(k), after.get(k)) for k in after if after[k] != before.get(k)}
        raise RuntimeError(f"a frozen corpus directory changed during the run: {changed}")
    synthetic_sha = assert_synthetic_corpus_unchanged()
    if synthetic_sha != SYNTHETIC_CONTENT_SHA256:
        raise RuntimeError(f"synthetic corpus content hash drifted: {synthetic_sha}")
    from brain.corpus.relations.freeze import assert_relations_v1_unchanged

    v1_sha = assert_relations_v1_unchanged()
    if v1_sha != RELATIONS_V1_CONTENT_SHA256:
        raise RuntimeError(f"relations v1 content hash drifted: {v1_sha}")
    v2_sha = load_relations(RELATIONS_V2_DIR).content_hash  # verifies, raises on drift
    if v2_sha != RELATIONS_V2_CONTENT_SHA256:
        raise RuntimeError(f"relations v2 content hash drifted: {v2_sha}")
    return {"directory_sha_before": before, "directory_sha_after": after,
            "directories_unchanged": True, "synthetic_content_sha256": synthetic_sha,
            "relations_v1_content_sha256": v1_sha, "relations_v2_content_sha256": v2_sha}
