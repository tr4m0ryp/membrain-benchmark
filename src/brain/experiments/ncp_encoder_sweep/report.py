"""Sweep the winner: NCP over encoders and neighbourhood sizes.

    python -m brain.experiments.ncp_encoder_sweep.report --seeds 5 [--smoke]

For each encoder in the pre-registered `ENCODERS` list, runs the full
(gamma, alpha, query-side) grid at every pre-registered `K_NEIGHBOURS_GRID`
value, against the global-df and SIF incumbents. Reports, per encoder: the
best local/global/SIF cells (max-over-cells, labelled selection-optimistic),
the primary-cell paired delta (not a hindsight pick), the strict one-variable
local-vs-global ablation with BOTH win directions counted, and the sibling-
cosine floor. Then answers the one question this task exists to answer: does
the local-vs-global separation become conclusive under any (encoder, k)?

Everything here is offline. No LLM backend is touched; nothing in this module
can bill an API call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from brain.config import ensure_dir, results_dir
from brain.corpus import load_corpus
from brain.corpus.types import Corpus
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.eval.results import environment, jsonable
from brain.experiments.gate_ncp.report import corpus_state_report
from brain.experiments.ncp_encoder_sweep.analysis import encoder_summary, matched_table_all_k
from brain.experiments.ncp_encoder_sweep.markdown import render_markdown
from brain.experiments.ncp_encoder_sweep.preregistration import (
    ENCODERS,
    K_NEIGHBOURS_GRID,
    PRIMARY_ALPHA,
    PRIMARY_GAMMA,
    PRIMARY_K,
    PRIMARY_QUERY_SIDE,
    RUNTIME_BUDGET_SECONDS,
    preregistration,
)
from brain.experiments.ncp_encoder_sweep.runner import run_encoder

__all__ = ["SCHEMA", "build_payload", "write_report", "main"]

SCHEMA = "ncp-encoder-sweep-result-v1"


def _global_verdict(all_matched: list[dict[str, Any]]) -> dict[str, Any]:
    local_wins = [m for m in all_matched if float(m["delta"]["lower"]) > 0.0]
    global_wins = [m for m in all_matched if float(m["delta"]["upper"]) < 0.0]
    any_local = bool(local_wins)
    any_global = bool(global_wins)
    if any_local and not any_global:
        headline = ("The local-vs-global separation becomes conclusive in the LOCAL direction "
                    "for at least one (encoder, k), and never in the global direction.")
    elif any_global and not any_local:
        headline = ("The local-vs-global separation is conclusive but points AWAY from NCP: "
                    "global beats local for at least one (encoder, k), never the reverse.")
    elif any_local and any_global:
        headline = ("The separation is conclusive in BOTH directions depending on cell: neither "
                    "family dominates, the claim does not survive as a general statement.")
    else:
        headline = ("No (encoder, k, gamma, alpha, query-side) cell in this sweep separates local "
                    "from global conclusively. F26's non-result generalises across every encoder "
                    "and neighbourhood size tested: this is now closed by measurement, not by a "
                    "single frozen configuration.")
    return {
        "total_matched_cells": len(all_matched),
        "local_conclusive_win_count": len(local_wins),
        "global_conclusive_win_count": len(global_wins),
        "any_local_conclusive_win": any_local,
        "any_global_conclusive_win": any_global,
        "headline": headline,
    }


def build_payload(*, command: str, encoder_summaries: list[dict[str, Any]],
                  all_matched: list[dict[str, Any]], corpus: Corpus,
                  shas_before: dict[str, Any], shas_after: dict[str, Any],
                  encoders_run: list[str], encoders_skipped: list[str],
                  seeds: list[int], n_resamples: int, wall_clock: float,
                  budget_seconds_used: float) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "command": command,
        "experiment": "Sweep the NCP winner: encoder x neighbourhood size (F26 -> lost a comparison)",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "preregistration": preregistration(),
        "config": {"seeds": seeds, "bootstrap_resamples": int(n_resamples),
                   "resampling_unit": "interference cluster", "retrieval_k": 5,
                   "budget_seconds_used_this_run": budget_seconds_used,
                   "budget_note": ("the PRE-REGISTERED budget is preregistration.runtime_budget_"
                                   "seconds (3600s); this field records what was actually enforced "
                                   "for this invocation, which may be larger if stated up front to "
                                   "account for shared-machine CPU contention observed at launch")},
        "encoder_summaries": encoder_summaries,
        "verdict": _global_verdict(all_matched),
        "encoders_run": encoders_run,
        "encoders_skipped": encoders_skipped,
        "corpus": {"content_sha256": corpus.manifest.get("content_sha256"),
                   "n_episodes": len(corpus.episodes), "n_queries": len(corpus.queries),
                   "n_clusters": len(corpus.clusters()) if hasattr(corpus, "clusters") else None,
                   "shas_before": shas_before, "shas_after": shas_after,
                   "unchanged": shas_before == shas_after},
        "llm_accounting": {"api_calls": 0, "cost_usd": 0.0,
                           "note": "no LLM-consuming arm exists on this experiment's path"},
        "wall_clock_seconds": round(wall_clock, 2),
        "environment": environment(),
    }


def write_report(payload: dict[str, Any], directory: Path | str) -> tuple[Path, Path]:
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem, suffix = f"ncp-encoder-{stamp}", 1
    while (root / f"{stem}.json").exists() or (root / f"{stem}.md").exists():
        stem, suffix = f"ncp-encoder-{stamp}-{suffix}", suffix + 1
    json_path, md_path = root / f"{stem}.json", root / f"{stem}.md"
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m brain.experiments.ncp_encoder_sweep.report")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--smoke", action="store_true", help="tiny query subset, single encoder")
    parser.add_argument("--results-dir", default=str(results_dir("ncp")))
    parser.add_argument("--encoders", nargs="*", default=None,
                        help="override the pre-registered encoder list (testing only)")
    parser.add_argument("--k-grid", nargs="*", type=int, default=None,
                        help="override the pre-registered k_neighbours grid (testing only)")
    parser.add_argument("--budget-seconds", type=float, default=RUNTIME_BUDGET_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parse(raw)
    command = "python -m brain.experiments.ncp_encoder_sweep.report " + " ".join(map(shlex.quote, raw))

    print("PRE-REGISTRATION (fixed before any number below is read):", flush=True)
    print(json.dumps(preregistration(), indent=2, sort_keys=True), flush=True)

    shas_before = corpus_state_report()
    for name, state in shas_before.items():
        if state.get("present") and not state.get("matches"):
            raise RuntimeError(f"frozen corpus {name} changed before the run started: {state}")

    corpus = load_corpus()
    if args.smoke:
        keep = sorted(corpus.queries, key=lambda q: str(q.query_id))[:12]
        corpus = Corpus(corpus.episodes, keep, corpus.manifest)

    seeds = tuple(range(max(1, int(args.seeds))))
    encoders = list(args.encoders) if args.encoders else (list(ENCODERS[:1]) if args.smoke else list(ENCODERS))
    k_grid = list(args.k_grid) if args.k_grid else (list(K_NEIGHBOURS_GRID[:2]) if args.smoke else list(K_NEIGHBOURS_GRID))

    started = time.perf_counter()
    encoder_summaries: list[dict[str, Any]] = []
    all_matched: list[dict[str, Any]] = []
    encoders_run: list[str] = []
    encoders_skipped: list[str] = []
    for name in encoders:
        elapsed = time.perf_counter() - started
        if elapsed >= args.budget_seconds:
            print(f"BUDGET EXCEEDED ({elapsed:.1f}s >= {args.budget_seconds}s): "
                 f"stopping cleanly before {name}", flush=True)
            encoders_skipped.append(name)
            continue
        print(f"== encoder: {name} ==", flush=True)
        run = run_encoder(name, corpus, seeds=seeds, k_grid=k_grid, n_resamples=int(args.resamples),
                          progress=lambda m: print(f"  {m}", flush=True))
        if not run.diagnostics.get("gamma_zero_identity_ok"):
            raise RuntimeError(f"{name}: gamma=0 did not reproduce mean pooling byte-identically: "
                               f"{run.diagnostics.get('gamma_zero_identity_failures')}")
        matched = matched_table_all_k(run, n_resamples=int(args.resamples))
        summary = encoder_summary(run, primary_gamma=PRIMARY_GAMMA, primary_alpha=PRIMARY_ALPHA,
                                  primary_k=PRIMARY_K if PRIMARY_K in k_grid else k_grid[0],
                                  primary_query_side=PRIMARY_QUERY_SIDE,
                                  n_resamples=int(args.resamples), matched_all=matched)
        encoder_summaries.append(summary)
        all_matched.extend(matched)
        encoders_run.append(name)

    shas_after = corpus_state_report()
    if shas_before != shas_after:
        raise RuntimeError("a frozen corpus directory changed during the run")

    wall_clock = time.perf_counter() - started
    payload = build_payload(command=command, encoder_summaries=encoder_summaries,
                            all_matched=all_matched, corpus=corpus, shas_before=shas_before,
                            shas_after=shas_after, encoders_run=encoders_run,
                            encoders_skipped=encoders_skipped, seeds=list(seeds),
                            n_resamples=int(args.resamples), wall_clock=wall_clock,
                            budget_seconds_used=float(args.budget_seconds))
    print("\n" + render_markdown(payload), flush=True)
    json_path, md_path = write_report(payload, args.results_dir)
    print(f"{json_path}\n{md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
