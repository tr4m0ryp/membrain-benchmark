"""Freeze the relational split: records, manifest, spot-check, content hash.

Same machinery as `brain.corpus.freeze`. A re-freeze over differing content is
refused unless `force` is passed, which is a deliberate and visible act, and the
hash is verified by `brain.corpus.relations.loader` on every load.

Two deliberate omissions:

- **No similarity statistic is computed here.** The existing corpus's manifest
  carries a measured-difficulty block; this one must not, because measuring the class
  geometry at freeze time is the first step towards tuning generation against it
  (`rules.md`, the instrument-bias rule). Task 006 owns that measurement.
- **`spotcheck.jsonl` is outside the content hash.** A reviewer fills in
  `human_verdict` after the freeze (spec T5), and that must not invalidate it.

Freezing asserts that `data/corpus/synthetic/` is byte-identical, so an additive
claim can never be made without evidence for it. **Task 007 adds a second split** at
`data/corpus/relations-v2/`, written by `--scale`, additive over BOTH predecessors:
`main` re-derives and pins the v1 hash too, and the record shape, the action map and
the conventions block are reused unchanged, so one frozen gate measures both.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from brain.config import CORPUS_DIR, ensure_dir, get_rng
from brain.corpus.loader import CorpusHashMismatch
from brain.corpus.relations.generate import (
    RelationsConfig, default_relations_config, generate_relations, scaled_relations_config,
)
from brain.corpus.relations.loader import (
    EPISODES_FILE, MANIFEST_FILE, PAIRS_FILE, RELATIONS_DIR, SPOTCHECK_FILE,
    SYNTHETIC_CONTENT_SHA256, assert_synthetic_corpus_unchanged, content_hashes, load_relations,
)
from brain.corpus.relations.types import (
    ACTIONS, CLASSES, REGIMES, SCHEMA_VERSION, RelationCorpus, RelationEpisode,
    classify, serialize_records, validate_relations,
)

__all__ = ["RelationsAlreadyFrozen", "RELATIONS_V2_DIR", "RELATIONS_V1_CONTENT_SHA256",
           "build_manifest", "spotcheck_rows", "freeze_relations",
           "assert_relations_v1_unchanged", "main"]

#: The task 007 power split: a SEPARATE directory with its own manifest and hash,
#: because `data/corpus/relations/` is immutable exactly like the synthetic one.
#: The v1 hash beneath it is pinned because F22 is stated against it.
RELATIONS_V2_DIR: Path = CORPUS_DIR / "relations-v2"
RELATIONS_V1_CONTENT_SHA256 = "aba780bd15ffeb303b065b1812c0f716e75397cae99112ad73b81ee2197b6e6d"


class RelationsAlreadyFrozen(RuntimeError):
    """A frozen relational split exists at this path with different content."""


def _word_stats(episodes: Sequence[RelationEpisode]) -> dict[str, Any]:
    words = sorted(len(e.text.split()) for e in episodes)
    if not words:
        return {"n": 0}
    return {"n": len(words), "min": words[0], "max": words[-1],
            "mean": round(sum(words) / len(words), 3), "median": words[len(words) // 2]}


def _counts(corpus: RelationCorpus) -> dict[str, Any]:
    by_regime, roles = corpus.episodes_by_regime(), sorted({e.facts.get("role", "") for e in corpus.episodes})
    return {
        "episodes": len(corpus.episodes),
        "episodes_by_regime": {r: len(v) for r, v in by_regime.items()},
        "episodes_by_role": {role: sum(1 for e in corpus.episodes if e.facts.get("role") == role)
                             for role in roles},
        "pairs": len(corpus.pairs),
        "pairs_by_class": {cls: len(v) for cls, v in corpus.pairs_by_class().items()},
        "pairs_by_class_and_regime": corpus.class_regime_counts(),
        "topics": len(corpus.clusters()),
        "pair_clusters": len({p.cluster_id for p in corpus.pairs}),
        "referents": len({e.slot_id for e in corpus.episodes}),
        "distinct_values": len({e.value_id for e in corpus.episodes}),
        "episode_words_by_regime": {r: _word_stats(v) for r, v in by_regime.items()},
    }


def _same_topic(corpus: RelationCorpus) -> dict[str, dict[str, dict[str, Any]]]:
    """Structural diagnostic, not a similarity measurement.

    Shared context is the `g * c` term Theorem 1 turns on, so the rate at which each
    class shares a workstream is the one number a reader needs to judge whether the
    instrument was staged. Interference is same-topic by definition; supersession is a
    mixture, and that mixture is `home_topic_prob` showing through rather than a
    per-class decision. The per-cell COUNTS ship beside the rate so the ceiling can be
    run on the same-topic stratum, where the asymmetry is removed.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cls, pairs in corpus.pairs_by_class().items():
        row: dict[str, dict[str, Any]] = {}
        for regime in REGIMES:
            subset = [p for p in pairs if p.regime == regime]
            shared = sum(p.same_topic for p in subset)
            row[regime] = {"n": len(subset), "same_topic": shared,
                           "cross_topic": len(subset) - shared,
                           "rate": round(shared / len(subset), 6) if subset else 0.0}
        out[cls] = row
    return out


def _allocation(corpus: RelationCorpus, total: int) -> dict[tuple[str, str], int]:
    """Largest-remainder proportional allocation over (class, regime) strata."""
    counts = corpus.class_regime_counts()
    strata = [(cls, regime) for cls in CLASSES for regime in REGIMES if counts[cls][regime]]
    n_pairs = len(corpus.pairs)
    exact = {key: total * counts[key[0]][key[1]] / n_pairs for key in strata}
    alloc = {key: min(int(exact[key]), counts[key[0]][key[1]]) for key in strata}
    while sum(alloc.values()) < min(total, n_pairs):
        key = max((k for k in strata if alloc[k] < counts[k[0]][k[1]]),
                  key=lambda k: (exact[k] - alloc[k], k))
        alloc[key] += 1
    return alloc


def spotcheck_rows(corpus: RelationCorpus, *, n: int, seed: int) -> list[dict[str, Any]]:
    """A seeded sample of labelled pairs with full text, for human review.

    Stratified proportionally over (class, regime) so every cell is actually looked
    at. Spec T5 makes this the check on synthetic labels and ships it with the
    artifact, so it must not be able to miss a class.
    """
    by_id, grouped = corpus.episode_by_id(), corpus.pairs_by_class()
    rng = get_rng(seed, stream="relations-spotcheck")
    rows: list[dict[str, Any]] = []
    for (cls, regime), want in sorted(_allocation(corpus, n).items()):
        pool = sorted((p for p in grouped[cls] if p.regime == regime), key=lambda p: p.pair_id)
        for index in sorted(int(x) for x in rng.choice(len(pool), size=want, replace=False)):
            pair = pool[index]
            a, b = by_id[pair.a], by_id[pair.b]
            rows.append({**pair.to_dict(), "action": ACTIONS[pair.cls],
                         "slot_a": a.slot_id, "slot_b": b.slot_id,
                         "value_a": a.value, "value_b": b.value,
                         "date_a": a.date, "date_b": b.date,
                         "text_a": a.text, "text_b": b.text,
                         "human_verdict": "", "reviewer_note": ""})
    return sorted(rows, key=lambda row: str(row["pair_id"]))


def build_manifest(corpus: RelationCorpus, config: RelationsConfig, *,
                   created_at: str | None = None) -> dict[str, Any]:
    """Seed, params, composition, conventions, hashes. No similarity statistic."""
    stamp = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": stamp,
        "seed": config.seed,
        "generator": config.to_dict(),
        "files": {"episodes": EPISODES_FILE, "pairs": PAIRS_FILE, "spotcheck": SPOTCHECK_FILE},
        "counts": _counts(corpus),
        "action_map": [
            {"rho": rho, "nu": nu, "cls": classify(rho, nu), "action": ACTIONS[classify(rho, nu)]}
            for rho, nu in ((1, 1), (1, 0), (0, 1), (0, 0))
        ],
        "same_topic": _same_topic(corpus),
        "spotcheck": {"file": SPOTCHECK_FILE, "n": min(config.spotcheck_n, len(corpus.pairs)),
                      "seed": config.spotcheck_seed, "stream": "relations-spotcheck",
                      "strata": "class x regime, largest-remainder proportional",
                      "in_content_hash": False},
        "conventions": _conventions(),
        "immutability": {
            "synthetic_content_sha256": SYNTHETIC_CONTENT_SHA256,
            "relations_v1_content_sha256": RELATIONS_V1_CONTENT_SHA256,
            "verified": True,
            "note": "this split is additive; data/corpus/synthetic and data/corpus/relations "
                    "are both untouched",
        },
        "hash_scope": "episodes.jsonl and pairs.jsonl only; manifest metadata excluded",
    }
    manifest.update(content_hashes(corpus.episodes, corpus.pairs))
    return manifest


def _conventions() -> dict[str, str]:
    """The choices a reader has to know to interpret the split correctly."""
    return {
        "rho": "exact equality of the referent (entity, attribute)",
        "nu": "exact equality of value_id",
        "interference_selection":
            "same-workstream pairs only: a merge policy fires above tau, so only colliding pairs "
            "ever reach a classifier. Cross-workstream (rho=0, nu=0) pairs exist in the store and "
            "stay relabellable from episodes.jsonl",
        "coincidence_selection": "any same-value pair with a different referent, no context constraint",
        "pairs_are_a_labelled_sample":
            "pairs.jsonl is the released sample at the Evaluation Plan composition, not the "
            "exhaustive relation. Any pair can be labelled from episodes.jsonl via "
            "brain.corpus.relations.types.pair_of, so absence from this file is not a negative",
        "resampling_unit": "cluster_id, the workstream of the earlier episode; slot_id is the finer alternative",
        "regime_pairing": "both regimes render the SAME plan, so per-regime comparisons are paired",
        "no_similarity_measured": "by design: task 006 owns the geometry, this split must not be tuned against it",
    }


def freeze_relations(corpus: RelationCorpus, config: RelationsConfig,
                     directory: Path | str | None = None, *, force: bool = False,
                     verify_synthetic: bool = True) -> dict[str, Any]:
    """Write the split, its manifest and its spot-check. Returns the manifest."""
    validate_relations(corpus)
    if verify_synthetic:
        assert_synthetic_corpus_unchanged()
    root = ensure_dir(Path(directory) if directory is not None else RELATIONS_DIR)
    fresh, manifest_path = content_hashes(corpus.episodes, corpus.pairs), root / MANIFEST_FILE
    created_at: str | None = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") == fresh["content_sha256"]:
            # Identical content: keep the original timestamp so a re-run is a no-op.
            created_at = existing.get("created_at")
        elif not force:
            raise RelationsAlreadyFrozen(
                f"{root} already holds a frozen split with content_sha256="
                f"{existing.get('content_sha256')}; refusing to overwrite with "
                f"{fresh['content_sha256']}. Pass force=True only on purpose.")
    manifest = build_manifest(corpus, config, created_at=created_at)
    (root / EPISODES_FILE).write_bytes(serialize_records(corpus.episodes, key="trace_id"))
    (root / PAIRS_FILE).write_bytes(serialize_records(corpus.pairs, key="pair_id"))
    rows = spotcheck_rows(corpus, n=config.spotcheck_n, seed=config.spotcheck_seed)  # outside the hash
    (root / SPOTCHECK_FILE).write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def assert_relations_v1_unchanged(directory: Path | str | None = None) -> str:
    """Recompute the v1 relational hash from disk and pin it. Returns the hash.

    v2 is additive over v1 exactly as v1 is over the synthetic corpus, and F22 is
    stated against the v1 hash, so writing v2 without proving v1 byte-identical would
    break the side-by-side comparison task 007 rests on.
    """
    corpus = load_relations(directory if directory is not None else RELATIONS_DIR, verify=True)
    if corpus.content_hash != RELATIONS_V1_CONTENT_SHA256:
        raise CorpusHashMismatch("the v1 relational split is NOT byte-identical: expected "
                                 f"{RELATIONS_V1_CONTENT_SHA256}, found {corpus.content_hash}")
    return corpus.content_hash


def _report(manifest: dict[str, Any], root: Path, priors: dict[str, str]) -> str:
    counts = manifest["counts"]
    lines = [f"wrote {root}",
             f"  episodes={counts['episodes']} by regime {counts['episodes_by_regime']} "
             f"by role {counts['episodes_by_role']}",
             f"  topics={counts['topics']} referents={counts['referents']} distinct values="
             f"{counts['distinct_values']} pair clusters={counts['pair_clusters']}",
             f"  pairs={counts['pairs']} {counts['pairs_by_class']}"]
    for cls in sorted(counts["pairs_by_class_and_regime"]):
        rates = {r: cell["rate"] for r, cell in sorted(manifest["same_topic"][cls].items())}
        lines.append(f"    {cls:<13} {counts['pairs_by_class_and_regime'][cls]}  "
                     f"same-topic rate {rates}")
    gen = manifest["generator"]
    lines += [
        f"  episode words {counts['episode_words_by_regime']}",
        f"  spotcheck n={manifest['spotcheck']['n']} seed={manifest['spotcheck']['seed']}",
        f"  generator={gen['label']} n_slots={gen['n_slots']} n_background={gen['n_background']} "
        f"states_per_slot={gen['states_per_slot']}",
        f"  content_sha256={manifest['content_sha256']}",
    ] + [f"  existing {name} content_sha256={sha} (UNCHANGED)" for name, sha in sorted(priors.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and freeze the relational corpus.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny configuration")
    parser.add_argument("--scale", type=int, default=1,
                        help="task 007: multiply every COUNT by this integer. Anything but 1 "
                             "writes a NEW split and leaves data/corpus/relations alone")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite a differing frozen split")
    args = parser.parse_args(argv)
    if args.smoke and args.scale != 1:
        parser.error("--smoke and --scale are exclusive: the smoke config is not a scaled v1")

    overrides: dict[str, Any] = {} if args.seed is None else {"seed": args.seed}
    scaled = int(args.scale) != 1  # a scaled run NEVER targets the v1 directory by default
    config = (scaled_relations_config(int(args.scale), **overrides) if scaled
              else default_relations_config(smoke=args.smoke or None, **overrides))
    corpus = generate_relations(config)
    root = Path(args.out) if args.out else (RELATIONS_V2_DIR if scaled else RELATIONS_DIR)
    priors = {"synthetic": assert_synthetic_corpus_unchanged()}
    if root.resolve() != RELATIONS_DIR.resolve():
        # Additive by construction: never write a second split without first
        # proving the split every published number is stated against is intact.
        priors["relations v1"] = assert_relations_v1_unchanged()
    manifest = freeze_relations(corpus, config, root, force=args.force)
    print(_report(manifest, root, priors))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
