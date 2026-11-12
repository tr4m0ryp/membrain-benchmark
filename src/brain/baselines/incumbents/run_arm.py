"""Score Mem0 as a retrieval arm on the frozen synthetic corpus.

    python -m brain.baselines.incumbents.run_arm

Separate from the write-path probes because it answers a different question and
costs a different amount: this is the number that goes in the same table as
`plain_vector`, `bm25` and `full_context`, computed by the same
`brain.eval.runner.run_arm` with the same k, the same seeds and the same cluster
bootstrap.

Read the arm's docstring before reading its number. It is Mem0's storage and
retrieval layer with the LLM replaced by a pass-through judge, which is the
FAVOURABLE reading of Mem0: no fact rewriting, so none of the addressability
loss F23 attributes to rewriting. A weaker number under a rewriting judge would
not be evidence against Mem0's design; a weak number under this one is.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from brain.config import results_dir
from brain.corpus.loader import load_corpus
from brain.corpus.relations.loader import assert_synthetic_corpus_unchanged
from brain.eval.results import jsonable
from brain.eval.runner import DEFAULT_K, DEFAULT_SEEDS, run_arm

from brain.baselines.incumbents.arm import Mem0Retriever

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Mem0 as a scored retrieval arm")
    parser.add_argument("--judge", default="identity", choices=("identity", "oracle"))
    parser.add_argument("--venv", default="mem0")
    parser.add_argument("--family", default="additive", choices=("additive", "legacy"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--results-dir", default=str(results_dir("incumbents")))
    arguments = parser.parse_args(argv)

    synthetic_sha = assert_synthetic_corpus_unchanged()
    corpus = load_corpus()

    name = f"mem0-{arguments.family}-{arguments.judge}"
    arm = Mem0Retriever(venv=arguments.venv, judge=arguments.judge, family=arguments.family, name=name)
    reason = arm.blocked_reason
    payload: dict[str, Any]
    if reason:
        payload = {"arm": name, "status": "BLOCKED", "blocked_reason": reason}
    else:
        # Mem0 is deterministic given a fixed judge and a fixed embedder, and
        # every store is rebuilt from scratch, so one pass is the whole result.
        result = run_arm(
            lambda: Mem0Retriever(
                venv=arguments.venv, judge=arguments.judge, family=arguments.family, name=name
            ),
            corpus,
            seeds=DEFAULT_SEEDS,
            k=int(arguments.k),
            name=name,
            seed_independent=True,
            write=False,
        )
        payload = {"arm": name, "status": "RAN", "result": result.to_dict()}

    payload["corpus_content_sha256"] = synthetic_sha
    payload["reading"] = (
        "Mem0's storage and retrieval layer on our corpus, with its LLM replaced by a "
        "pass-through judge. NOT Mem0 as shipped: Mem0 as shipped cannot write without a paid "
        "key. The substitution removes fact rewriting, so this is the favourable reading."
    )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(arguments.results_dir) / f"incumbents-{stamp}-arm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}: {payload['status']}")
    if payload["status"] == "RAN":
        interval = payload["result"]["disambiguation_cluster_bootstrap"]
        print(
            f"  disambiguation {interval['mean']:.4f} "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
