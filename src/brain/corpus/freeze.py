"""Freeze the corpus: write the records, the manifest, and the content hash.

Freezing is what makes the corpus an instrument rather than a moving target. The
spec's Threats to Validity names the failure this guards against: we build both
the mechanism and its benchmark, so the benchmark is fixed and hashed BEFORE any
mechanism is tuned. Re-freezing over different content is refused unless the
caller passes `force`, which is a deliberate, visible act.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.config import ensure_dir
from brain.corpus.generate import GeneratorConfig, default_config, generate_corpus
from brain.corpus.loader import (
    EPISODES_FILE,
    MANIFEST_FILE,
    QUERIES_FILE,
    SYNTHETIC_DIR,
    content_hashes,
)
from brain.corpus.similarity import corpus_similarity_stats
from brain.corpus.types import SCHEMA_VERSION, Corpus, serialize_records
from brain.corpus.validate import CorpusValidationError

__all__ = ["CorpusAlreadyFrozen", "build_manifest", "freeze_corpus", "main"]


class CorpusAlreadyFrozen(RuntimeError):
    """A frozen corpus exists at this path with different content."""


def _cluster_labels(corpus: Corpus) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for cluster_id, members in corpus.clusters().items():
        head = members[0]
        labels.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "path": head.topic["path"],
                "subsystem": head.topic["subsystem"],
                "symptom": head.topic["symptom"],
                "trace_ids": [member.trace_id for member in members],
            }
        )
    return labels


def _counts(corpus: Corpus) -> dict[str, Any]:
    sizes = corpus.cluster_sizes()
    histogram: dict[str, int] = {}
    for size in sorted(sizes.values()):
        histogram[str(size)] = histogram.get(str(size), 0) + 1
    by_size = corpus.queries_by_cluster_size()
    return {
        "clusters": len(sizes),
        "episodes": len(corpus.episodes),
        "queries": len(corpus.queries),
        "cluster_size_histogram": histogram,
        "episodes_per_cluster_size": {
            str(size): size * histogram[str(size)] for size in sorted(int(k) for k in histogram)
        },
        "queries_per_cluster_size": {str(size): len(items) for size, items in by_size.items()},
    }


def build_manifest(
    corpus: Corpus,
    config: GeneratorConfig,
    *,
    measure_embeddings: bool = True,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the manifest: seed, params, labels, difficulty stats, hashes."""
    stamp = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": stamp,
        "seed": config.seed,
        "generator": config.to_dict(),
        "files": {"episodes": EPISODES_FILE, "queries": QUERIES_FILE},
        "counts": _counts(corpus),
        "clusters": _cluster_labels(corpus),
        "similarity": corpus_similarity_stats(corpus, measure_embeddings=measure_embeddings),
        "hash_scope": "episodes.jsonl and queries.jsonl only; manifest metadata excluded",
    }
    _assert_fits_encoder(manifest["similarity"])
    manifest.update(content_hashes(corpus.episodes, corpus.queries))
    return manifest


def _assert_fits_encoder(similarity: dict[str, Any]) -> None:
    """Refuse to freeze a corpus the gate embedder would truncate.

    A truncated trace loses its tail, and the tail is where part of what
    distinguishes it from its siblings lives. Freezing that would mean the whole
    experiment compares vectors that dropped the information under test.
    """
    tokens = similarity.get("embedding", {}).get("episode_tokens")
    if not tokens:
        return
    if tokens["over_window"]:
        raise CorpusValidationError(
            f"{tokens['over_window']} episodes exceed the {tokens['window']} token encoder "
            f"window (longest {tokens['max']}); shorten the episodes before freezing"
        )


def freeze_corpus(
    corpus: Corpus,
    config: GeneratorConfig,
    directory: Path | str | None = None,
    *,
    measure_embeddings: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Write the corpus and its manifest. Returns the manifest."""
    root = ensure_dir(Path(directory) if directory is not None else SYNTHETIC_DIR)
    fresh = content_hashes(corpus.episodes, corpus.queries)
    existing_path = root / MANIFEST_FILE
    created_at: str | None = None
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") == fresh["content_sha256"]:
            # Identical content: keep the original timestamp so a re-run is a no-op.
            created_at = existing.get("created_at")
        elif not force:
            raise CorpusAlreadyFrozen(
                f"{root} already holds a frozen corpus with content_sha256="
                f"{existing.get('content_sha256')}; refusing to overwrite with "
                f"{fresh['content_sha256']}. Pass force=True only on purpose."
            )
    manifest = build_manifest(
        corpus, config, measure_embeddings=measure_embeddings, created_at=created_at
    )
    (root / EPISODES_FILE).write_bytes(serialize_records(corpus.episodes, key="trace_id"))
    (root / QUERIES_FILE).write_bytes(serialize_records(corpus.queries, key="query_id"))
    existing_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _report(manifest: dict[str, Any], root: Path) -> str:
    counts = manifest["counts"]
    lines = [
        f"wrote {root}",
        f"  clusters={counts['clusters']} episodes={counts['episodes']} queries={counts['queries']}",
        f"  cluster sizes: {counts['cluster_size_histogram']}",
        f"  content_sha256={manifest['content_sha256']}",
    ]
    for name in ("lexical", "embedding"):
        block = manifest["similarity"].get(name)
        if not block or "sibling" not in block:
            lines.append(f"  {name}: unavailable")
            continue
        queries = block["queries"]
        lines.append(
            f"  {name}: sibling cos mean={block['sibling']['mean']} "
            f"non-sibling mean={block['non_sibling']['mean']} "
            f"sep_index={block['separation_index']}"
        )
        lines.append(
            f"    untuned disambiguation={queries['disambiguation_accuracy']} "
            f"(chance={queries['chance_accuracy']}, "
            f"verbatim-cue ceiling={block['queries_verbatim_cue']['disambiguation_accuracy']})"
        )
        lines.append(f"    by cluster size {queries['accuracy_by_cluster_size']}")
        fit = block["m_a_fit"]
        collisions = {row["tau"]: row["rho_sibling"] for row in block["collision_rates"]}
        lines.append(
            f"    M-A fit rho={fit['rho']} max_abs_error={fit['max_abs_error']}; "
            f"sibling collision rate by tau {collisions}"
        )
        tokens = block.get("episode_tokens")
        if tokens:
            lines.append(
                f"    episode tokens max={tokens['max']} mean={tokens['mean']} "
                f"window={tokens['window']} over_window={tokens['over_window']}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and freeze the synthetic corpus.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--similarity", type=float, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny 2x3 configuration")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-embed", action="store_true", help="skip the model-based stats")
    parser.add_argument("--force", action="store_true", help="overwrite a differing frozen corpus")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.similarity is not None:
        overrides["similarity"] = args.similarity
    config = default_config(smoke=args.smoke or None, **overrides)
    corpus = generate_corpus(config)
    root = Path(args.out) if args.out else SYNTHETIC_DIR
    manifest = freeze_corpus(
        corpus,
        config,
        root,
        measure_embeddings=not args.no_embed,
        force=args.force,
    )
    print(_report(manifest, root))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
