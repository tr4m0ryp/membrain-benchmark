"""The individual pipeline stages: encode, scan, pseudonymise, profile.

Split out of `pipeline` so the driver reads as the ordered safety argument it is
(scan, quarantine, pseudonymise, re-scan, write) without the mechanics of each
step in the way.

`_log` writes stage names and counts to stderr. It is never handed episode text,
so a run captured in a terminal log or a CI transcript cannot leak what it is in
the middle of scrubbing.
"""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brain.config import REPO_ROOT
from brain.corpus.transcripts.emit import git_identity_literals
from brain.corpus.transcripts.scrub import (
    Pseudonymiser,
    aggregate,
    load_or_create_salt,
    scan_text,
    verdict,
)
from brain.corpus.transcripts.stats import corpus_profile
from brain.corpus.transcripts.types import CandidateEpisode

__all__ = ["SYNTHETIC_DIR", "log", "embed", "synthetic_profiles", "scan_all", "pseudonymise"]

SYNTHETIC_DIR: Path = REPO_ROOT / "data" / "corpus" / "relations-v2"


def log(message: str) -> None:
    """Progress to stderr. Counts and stage names only, never episode text."""
    print(f"[transcripts] {message}", file=sys.stderr, flush=True)


def embed(texts: Sequence[str], *, cache_dir: Path | None = None) -> np.ndarray | None:
    """Encode, caching on the sha256 of the exact text list.

    The cache is keyed on the whole list rather than per text, because batch
    composition is fixed for a measurement (cross-batch float32 rounding of ~1e-7
    is intrinsic) and a per-text cache would silently change it. It lives under
    `data/transcripts/`, so it is gitignored like everything else derived here.
    """
    ordered = list(texts)
    digest = sha256("\x00".join(ordered).encode("utf-8")).hexdigest()
    path = (cache_dir / f"{digest}.npy") if cache_dir else None
    if path is not None and path.is_file():
        log(f"embed cache hit, {len(ordered)} texts")
        return np.load(path)
    try:
        from brain.embed import get_embedder

        log(f"encoding {len(ordered)} texts")
        vectors = np.asarray(get_embedder().encode(ordered), dtype=np.float64)
    except Exception:  # pragma: no cover - offline, missing weights, no torch
        return None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    return vectors


def synthetic_profiles(cache_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Profile the frozen relational split, per regime, for the comparison."""
    try:
        from brain.corpus.relations.loader import load_relations

        corpus = load_relations(SYNTHETIC_DIR)
    except Exception:  # pragma: no cover - corpus absent on a fresh checkout
        log("synthetic split unavailable, comparison will be empty")
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for regime, episodes in corpus.episodes_by_regime().items():
        texts = [e.text for e in episodes]
        clusters = [e.cluster_id for e in episodes]
        log(f"synthetic {regime}: {len(texts)} episodes")
        profiles[regime] = corpus_profile(texts, clusters, embed(texts, cache_dir=cache_dir))
    return profiles


def scan_all(episodes: Sequence[CandidateEpisode]):
    """Scan raw text. Returns survivors (masked text), quarantine rows, report.

    Quarantine rows keep the REAL session id and drop the text: a triaging human
    needs to find the transcript, and does not need a second copy of what was
    already flagged. They land under `data/transcripts/quarantine/`, gitignored.
    """
    survivors: list[CandidateEpisode] = []
    quarantined: list[dict[str, Any]] = []
    counts_seen: list[dict[str, int]] = []
    for episode in episodes:
        masked, counts = scan_text(episode.text)
        counts_seen.append(counts)
        outcome = verdict(counts)
        record = CandidateEpisode(
            episode_id=episode.episode_id,
            session_id=episode.session_id,
            regime=episode.regime,
            timestamp=episode.timestamp,
            text=masked,
            subject=episode.subject,
            referent=episode.referent,
            n_words=len(masked.split()),
            n_added_lines=episode.n_added_lines,
            n_removed_lines=episode.n_removed_lines,
            scan_categories=tuple(sorted(c for c, n in counts.items() if n)),
        )
        if outcome in ("block", "flag"):
            row = record.to_dict()
            row["quarantine_reason"] = outcome
            row.pop("text", None)
            quarantined.append(row)
        else:
            survivors.append(record)
    return survivors, quarantined, aggregate(counts_seen)


def pseudonymise(episodes: Sequence[CandidateEpisode], directory: Path):
    """Replace every identifier, accumulating the inverse map as it goes."""
    salt = load_or_create_salt(directory)
    scrubber = Pseudonymiser(salt=salt, literals=git_identity_literals())
    out: list[CandidateEpisode] = []
    for episode in episodes:
        text = scrubber.apply(episode.text)
        out.append(
            CandidateEpisode(
                episode_id=episode.episode_id,
                # Through `render`, not `token`, so the substitution lands in the
                # inverse map. Without it a reviewer holding a candidate episode
                # has no way back to the transcript it came from, and provenance
                # is exactly what a disputed label needs.
                session_id=scrubber.render("session_id", episode.session_id),
                regime=episode.regime,
                timestamp=episode.timestamp,
                text=text,
                subject=scrubber.apply(episode.subject),
                referent={
                    "entity": episode.referent.get("entity", ""),
                    "attribute": scrubber.apply(episode.referent.get("attribute", "")),
                },
                n_words=len(text.split()),
                n_added_lines=episode.n_added_lines,
                n_removed_lines=episode.n_removed_lines,
                scan_categories=episode.scan_categories,
            )
        )
    return out, scrubber
