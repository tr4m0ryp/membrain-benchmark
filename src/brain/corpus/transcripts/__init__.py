"""Turn real Claude Code transcripts into a CANDIDATE annotated corpus.

**Read this first: what comes out of here is unvalidated and carries no ground
truth.** The four write classes are a function of `(rho, nu)`, and neither bit is
observable in a transcript. This package therefore stops one step short of a
corpus: it produces episodes that are safe to look at and pairs that are worth a
human's time, plus the protocol for labelling them. Calling its output a corpus
before that human has finished would be exactly the overclaim the paper's
external-validity limitation warns about.

Pipeline, in the order the safety argument requires:

    inventory  ->  extract  ->  SCAN  ->  quarantine  ->  pseudonymise
               ->  RE-SCAN  ->  mine pairs  ->  measure  ->  write

Everything derived lands in `data/transcripts/`, which is verified gitignored at
write time. Scan findings are reported as categories and counts; no code path in
this package can place a detected secret value into a record, a report or a log.

Entry point: `python -m brain.corpus.transcripts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # keep import cost off the package root
    from brain.corpus.transcripts.inventory import Inventory, take_inventory
    from brain.corpus.transcripts.mine import MiningConfig, mine_pairs
    from brain.corpus.transcripts.pipeline import DEFAULT_ROOT, ScanNotClean, run_pipeline
    from brain.corpus.transcripts.protocol import PROTOCOL, effort_estimate
    from brain.corpus.transcripts.stats import compare, corpus_profile, implied_beta
    from brain.corpus.transcripts.types import (
        SCHEMA_VERSION,
        CandidateEpisode,
        CandidatePair,
        ScanReport,
    )

_ORIGIN = {
    "Inventory": "inventory",
    "take_inventory": "inventory",
    "MiningConfig": "mine",
    "mine_pairs": "mine",
    "DEFAULT_ROOT": "pipeline",
    "ScanNotClean": "pipeline",
    "run_pipeline": "pipeline",
    "PROTOCOL": "protocol",
    "effort_estimate": "protocol",
    "compare": "stats",
    "corpus_profile": "stats",
    "implied_beta": "stats",
    "SCHEMA_VERSION": "types",
    "CandidateEpisode": "types",
    "CandidatePair": "types",
    "ScanReport": "types",
}

__all__ = sorted(_ORIGIN)


def __getattr__(name: str) -> Any:
    module_name = _ORIGIN.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(__all__)
