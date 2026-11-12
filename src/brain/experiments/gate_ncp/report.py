"""Task 008 artifacts: the whole (gamma, alpha) surface, its baselines, its verdict.

    python -m brain.experiments.gate_ncp.report --seeds 5 --k 5 [--smoke]

Three things this file refuses to let a reader miss.

1. **The C-A test.** A1 died because a symmetric transform on finished vectors
   cannot reorder a colliding pair: 0.8056 at every strength, identical to four
   decimals. So the verdict computes whether disambiguation is FLAT in gamma and
   whether any sibling pair actually reordered. Flat plus zero reorderings means
   the mechanism was rebuilt under a new name, and that is reported as the result.
2. **Sibling cosine never travels alone.** C-A also cut sibling cosine 0.824 to
   0.704 and moved disambiguation by nothing, so every cell prints both, and a
   cosine drop with a flat or negative accuracy delta is counted as a failure.
3. **The incumbent is in the table.** Weighted pooling is Arora et al's, so
   global-df weighting (the exact one-variable ablation) and SIF proper are
   baseline arms. Without them "we improved weighted pooling" is an assertion.

`BRAIN_LLM_BACKEND=none` is set before any arm is built, so the paid arms can
only replay from their on-disk cache and a cold call is not reachable. Spend is
asserted to be exactly zero at the end rather than assumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from brain.config import CORPUS_DIR, ensure_dir, results_dir
from brain.corpus import load_corpus
from brain.corpus.types import Corpus
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.eval.results import corpus_provenance, environment, jsonable
from brain.eval.runner import DEFAULT_K
from brain.experiments.gate_ncp.arms import cost_probe, run_baselines
from brain.experiments.gate_ncp.markdown import render_markdown
from brain.experiments.gate_ncp.runner import (
    SUBSAMPLE_N, GridRun, preregistration, run_grid, subsample_ids,
)
from brain.experiments.gate_ncp.verdict import comparisons, verdict
from brain.experiments.gate_veto.report import directory_sha

__all__ = ["SCHEMA", "CORPUS_SHAS", "corpus_state_report", "build_payload",
           "write_report", "main"]

SCHEMA = "ncp-result-v1"
#: Every frozen split this task must leave byte-identical. Asserted before and
#: after the run; the relational splits are read-only here and only pinned.
CORPUS_SHAS = {
    "synthetic": "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb",
    "relations": "aba780bd",
    "relations-v2": "24ff2689",
}


def corpus_state_report() -> dict[str, Any]:
    """Content hash and a directory digest for every frozen split, from disk."""
    from brain.corpus.relations.loader import load_relations

    out: dict[str, Any] = {}
    for name in ("synthetic", "relations", "relations-v2"):
        root = CORPUS_DIR / name
        if not root.is_dir():
            out[name] = {"present": False}
            continue
        loaded = load_corpus(root) if name == "synthetic" else load_relations(root)
        content = str(loaded.manifest.get("content_sha256", ""))
        expected = CORPUS_SHAS[name]
        out[name] = {"present": True, "content_sha256": content,
                     "directory_sha256": directory_sha(root),
                     "expected_prefix": expected,
                     "matches": content.startswith(expected)}
    return out


def build_payload(run: GridRun, *, command: str, shas_before: dict[str, Any],
                  shas_after: dict[str, Any], n_resamples: int) -> dict[str, Any]:
    """Every number this run produced, in one JSON-safe record."""
    decision = verdict(run)
    deltas = comparisons(run, n_resamples=n_resamples)
    for arm in run.baselines.values():  # cluster indicators are working state, not a result
        arm.pop("clusters", None)
        arm.pop("subsample_clusters", None)
    spend = sum(float((a.get("stats", {}) or {}).get("cost_usd", 0.0) or 0.0)
                for a in run.baselines.values())
    calls = sum(int((a.get("stats", {}) or {}).get("api_calls", 0) or 0)
                for a in run.baselines.values())
    return {
        "schema": SCHEMA, "command": command,
        "experiment": "Task 008 -- neighbourhood-conditional pooling (NCP)",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "verdict_line": f"VERDICT: NCP {decision['status'].upper()}",
        "verdict": decision, "preregistration": preregistration(),
        "paired_deltas": deltas,
        "config": {"k": run.k, "seeds": list(run.seeds), "bootstrap_resamples": int(n_resamples),
                   "resampling_unit": "interference cluster",
                   "seed_pooling": "mean over seeds per query, then resample clusters",
                   "subsample_n": SUBSAMPLE_N,
                   "subsample_note": ("llm_pick's response cache was warmed on this 180-query "
                                      "stratified subsample (F24), so the paid arm is compared "
                                      "on the queries it actually answered and no cold call is "
                                      "made")},
        "cells": run.cells, "baselines": run.baselines, "blocked_arms": run.blocked,
        "diagnostics": run.diagnostics, "cost": run.cost,
        "llm_accounting": {"api_calls": calls, "cost_usd": round(spend, 6),
                           "backend": os.environ.get("BRAIN_LLM_BACKEND"),
                           "per_query_llm_calls_on_the_ncp_path": 0},
        "corpus": {**corpus_provenance(run.corpus), "shas_before": shas_before,
                   "shas_after": shas_after,
                   "unchanged": shas_before == shas_after},
        "wall_clock_seconds": round(float(run.wall_clock), 2), "environment": environment(),
    }


def write_report(payload: dict[str, Any], directory: Path | str) -> tuple[Path, Path]:
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem, suffix = f"ncp-{stamp}", 1
    while (root / f"{stem}.json").exists() or (root / f"{stem}.md").exists():
        stem, suffix = f"ncp-{stamp}-{suffix}", suffix + 1
    json_path, md_path = root / f"{stem}.json", root / f"{stem}.md"
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m brain.experiments.gate_ncp.report")
    for flag, kw in (("--seeds", {"type": int, "default": 5}),
                     ("--k", {"type": int, "default": DEFAULT_K}),
                     ("--resamples", {"type": int, "default": DEFAULT_RESAMPLES}),
                     ("--smoke", {"action": "store_true", "help": "tiny N, no baselines"}),
                     ("--no-cost-probe", {"action": "store_true"}),
                     ("--results-dir", {"default": str(results_dir("ncp"))})):
        parser.add_argument(flag, **kw)
    return parser.parse_args(argv)


def _cost(corpus: Corpus) -> dict[str, Any]:
    """Measured cost at both corpus sizes. Nothing here touches an API."""
    from brain.baselines.base import as_traces
    from brain.corpus.relations.loader import load_relations

    queries = [q.text for q in sorted(corpus.queries, key=lambda q: str(q.query_id))]
    small = cost_probe(as_traces(sorted(corpus.episodes, key=lambda e: str(e.trace_id))), queries)
    out = {"synthetic_120": small}
    root = CORPUS_DIR / "relations-v2"
    if root.is_dir():
        relations = load_relations(root)
        traces = as_traces(sorted(relations.episodes, key=lambda e: str(e.trace_id)))
        out["relations_v2_870"] = cost_probe(traces, queries)
    return out


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parse(raw)
    # Before any arm exists: a cold LLM call must be unreachable, not merely unwanted.
    os.environ["BRAIN_LLM_BACKEND"] = "none"
    command = "python -m brain.experiments.gate_ncp.report " + " ".join(map(shlex.quote, raw))
    shas_before = corpus_state_report()
    print(json.dumps({"corpus_shas_before": shas_before}, indent=2, sort_keys=True), flush=True)
    for name, state in shas_before.items():
        if state.get("present") and not state.get("matches"):
            raise RuntimeError(f"frozen corpus {name} changed: {state}")
    corpus = load_corpus()
    if args.smoke:
        keep = sorted(corpus.queries, key=lambda q: str(q.query_id))[:12]
        corpus = Corpus(corpus.episodes, keep, corpus.manifest)
    seeds = tuple(range(max(1, int(args.seeds))))
    print("PRE-REGISTERED GRID (fixed before the run):", flush=True)
    print(json.dumps(preregistration(), indent=2, sort_keys=True), flush=True)
    run = run_grid(corpus, seeds=seeds, k=int(args.k), n_resamples=int(args.resamples),
                   progress=lambda m: print(f"  {m}", flush=True))
    subset = set(subsample_ids(corpus.queries, SUBSAMPLE_N))
    if not args.smoke:
        run.baselines, run.blocked = run_baselines(
            corpus, seeds=seeds, k=int(args.k), subset=subset,
            n_resamples=int(args.resamples), progress=lambda m: print(f"  {m}", flush=True))
        if not args.no_cost_probe:
            run.cost = _cost(corpus)
    shas_after = corpus_state_report()
    if shas_before != shas_after:
        raise RuntimeError("a frozen corpus directory changed during the run")
    payload = build_payload(run, command=command, shas_before=shas_before,
                            shas_after=shas_after, n_resamples=int(args.resamples))
    if float(payload["llm_accounting"]["cost_usd"]) != 0.0:
        raise RuntimeError(f"this task must cost nothing, spent {payload['llm_accounting']}")
    print("\n" + render_markdown(payload), flush=True)
    json_path, md_path = write_report(payload, args.results_dir)
    print(json.dumps({"corpus_shas_after": shas_after}, indent=2, sort_keys=True), flush=True)
    for name, reason in sorted(run.blocked.items()):
        print(f"BLOCKED {name}: {reason}", flush=True)
    print(f"{json_path}\n{md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
