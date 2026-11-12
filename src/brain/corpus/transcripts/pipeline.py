"""The driver: inventory, extract, scan, pseudonymise, mine, measure, write.

The step order is a safety property. Scanning happens on the RAW extracted text,
quarantine happens before pseudonymisation, and the surviving set is re-scanned
after pseudonymisation with a hard refusal to write if anything still hits. The
re-scan exists because pseudonymisation rewrites text, and a rewrite that
reintroduced a credential shape would otherwise reach disk unnoticed.

The encoder is optional. Without it the pipeline still produces the inventory,
the scan, the pseudonymised episodes and the lexical statistics; it cannot mine
pairs, because pairs are selected by similarity. That degradation is reported
rather than silently swallowed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.config import REPO_ROOT, SEED
from brain.config import results_dir as results_home
from brain.corpus.transcripts.emit import (
    TRANSCRIPTS_DIR,
    render_markdown,
    write_artifacts,
    write_json,
)
from brain.corpus.transcripts.extract import episodes_of
from brain.corpus.transcripts.inventory import summarize, take_inventory
from brain.corpus.transcripts.mine import MiningConfig, mine_pairs, mining_summary
from brain.corpus.transcripts.protocol import effort_estimate
from brain.corpus.transcripts.scrub import aggregate, scan_text
from brain.corpus.transcripts.stages import (
    embed,
    log,
    pseudonymise,
    scan_all,
    synthetic_profiles,
)
from brain.corpus.transcripts.stats import compare, corpus_profile
from brain.corpus.transcripts.types import REGIMES, SCHEMA_VERSION, CandidateEpisode

__all__ = ["DEFAULT_ROOT", "ScanNotClean", "run_pipeline"]

#: Where the session store is looked for. It is NOT distributed with this
#: repository: transcripts are unscrubbed input to the scanner below and carry
#: whatever the recorded sessions carried. Pass --root to point at a local
#: Claude Code store and the experiment mines those sessions instead.
DEFAULT_ROOT: Path = REPO_ROOT / "sessions"


class ScanNotClean(RuntimeError):
    """The post-pseudonymisation re-scan still hits. Nothing is written."""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rel(path: Path | str) -> str:
    """Repo-relative path.

    `results/` IS tracked, so an absolute path written into the report would
    commit the operator's home directory, which the scan classifies as an
    identifier everywhere else. Relative paths also make the result reproducible
    from a different checkout.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return Path(path).name


def run_pipeline(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    results_dir: Path | None = None,
    limit: int | None = None,
    mining: MiningConfig | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run every stage and return the result payload."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    out_dir = Path(out_dir) if out_dir is not None else TRANSCRIPTS_DIR
    results_dir = Path(results_dir) if results_dir is not None else results_home("transcripts")
    run_id = _utc_stamp()
    cache_dir = out_dir / ".embed-cache"

    log(f"inventory over {_rel(root)}")
    inventory = take_inventory(root, limit=limit)
    inventory.root = _rel(inventory.root)
    log(f"{len(inventory)} sessions, {len(inventory.episode_bearing)} episode-bearing")
    raw: list[CandidateEpisode] = []
    for session in inventory.episode_bearing:
        raw.extend(episodes_of(session))
    log(f"{len(raw)} candidate episodes extracted, scanning")

    survivors, quarantined, scan_report = scan_all(raw)
    log(f"scan: {len(survivors)} survive, {len(quarantined)} quarantined")
    scrubbed, scrubber = pseudonymise(survivors, out_dir)

    recheck = aggregate(scan_text(e.text, redact=False)[1] for e in scrubbed)
    if recheck.n_blocked or recheck.n_flagged:
        raise ScanNotClean(
            f"re-scan after pseudonymisation still hits: {recheck.n_blocked} blocked, "
            f"{recheck.n_flagged} flagged. Nothing written."
        )

    by_regime = {r: [e for e in scrubbed if e.regime == r] for r in REGIMES}
    real_profiles: dict[str, Any] = {}
    pairs = []
    audits: dict[str, Any] = {}
    for regime, episodes in by_regime.items():
        if not episodes:
            continue
        texts = [e.text for e in episodes]
        clusters = [e.referent.get("entity", "") for e in episodes]
        log(f"real {regime}: {len(texts)} episodes")
        vectors = embed(texts, cache_dir=cache_dir)
        real_profiles[regime] = corpus_profile(texts, clusters, vectors)
        log(f"real {regime}: profile done")
        if vectors is not None:
            mined, audit = mine_pairs(episodes, vectors, config=mining)
            pairs.extend(mined)
            audits[regime] = audit
            log(f"real {regime}: {len(mined)} pairs mined")

    synthetic = synthetic_profiles(cache_dir)
    comparison = {
        regime: compare(real_profiles[regime], synthetic.get(regime, {}))
        for regime in sorted(real_profiles)
    }

    payload: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "unvalidated": True,
        "ground_truth": None,
        "warning": (
            "CANDIDATE corpus. No pair carries a class. No external-validity claim is "
            "supported by anything in this file until a human has labelled the pairs "
            "under the protocol and an inter-annotator agreement figure is reported."
        ),
        "inventory": summarize(inventory),
        "extraction": {
            "n_extracted": len(raw),
            "n_events": len({e.episode_id.rsplit(":", 1)[0] for e in raw}),
            "n_entities": len({e.referent.get("entity", "") for e in raw}),
            "by_regime": {r: sum(1 for e in raw if e.regime == r) for r in REGIMES},
            "n_survivors": len(scrubbed),
            "n_quarantined": len(quarantined),
        },
        "scan": {
            "pre_pseudonymisation": scan_report.to_dict(),
            "post_pseudonymisation": recheck.to_dict(),
        },
        "pseudonymisation": {
            "n_mapped": len(scrubber.mapping),
            "categories": dict(sorted(scrubber.categories.items())),
            "map_path": _rel(out_dir / "pseudonym-map.json"),
        },
        "mining": mining_summary(pairs),
        "mining_audit": audits,
        "real_profiles": real_profiles,
        "synthetic_profiles": synthetic,
        "comparison": comparison,
        "labelling_effort": effort_estimate(pairs),
        "encoder_available": bool(real_profiles) and any("cosine" in p for p in real_profiles.values()),
    }

    if write:
        manifest = {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "unvalidated": True,
            "source_root": _rel(root),
            "n_episodes": len(scrubbed),
            "n_pairs": len(pairs),
            "n_quarantined": len(quarantined),
            "scan": payload["scan"],
        }
        written = write_artifacts(
            out_dir,
            episodes=scrubbed,
            pairs=pairs,
            quarantined=quarantined,
            manifest=manifest,
        )
        payload["artifacts"] = {name: _rel(path) for name, path in sorted(written.items())}
        scrubber.write_map(out_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        write_json(results_dir / f"transcripts-{run_id}.json", payload)
        (results_dir / f"transcripts-{run_id}.md").write_text(
            render_markdown(payload), encoding="utf-8"
        )
    return payload
