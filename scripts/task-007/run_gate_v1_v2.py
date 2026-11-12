"""Task 007: run the FROZEN K1 gate on the v1 and v2 relational splits.

This is a thin driver, not a measurement. Every number comes from
`brain.experiments.gate_threshold`, which is imported unchanged: the module's own
`main()` hardcodes `data/corpus/relations/`, so the only thing here is pointing
`run_experiment` at a second frozen split and writing both reports side by side.

    .venv/bin/python run_gate_v1_v2.py [--seeds 5] [--resamples 10000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from brain.config import CORPUS_DIR, RESULTS_DIR
from brain.corpus.relations.freeze import (
    RELATIONS_V1_CONTENT_SHA256,
    RELATIONS_V2_DIR,
    assert_relations_v1_unchanged,
)
from brain.corpus.relations.loader import (
    RELATIONS_DIR,
    SYNTHETIC_CONTENT_SHA256,
    assert_synthetic_corpus_unchanged,
    load_relations,
)
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.experiments.gate_threshold import distributions as D
from brain.experiments.gate_threshold.report import build_payload, run_experiment, write_report
from brain.experiments.gate_veto.report import directory_sha

SPLITS = {"v1": RELATIONS_DIR, "v2": RELATIONS_V2_DIR}
SYNTHETIC_DIR = CORPUS_DIR / "synthetic"


def shas() -> dict[str, str]:
    return {name: directory_sha(path) for name, path in
            (("synthetic", SYNTHETIC_DIR), ("relations_v1", RELATIONS_DIR),
             ("relations_v2", RELATIONS_V2_DIR))}


def cell(payload: dict, stratum: str, regime: str) -> dict:
    return payload["ceilings"]["derivable"][stratum][regime]


def line(name: str, payload: dict, stratum: str, regime: str) -> str:
    block = cell(payload, stratum, regime)
    ceiling, boot, k1 = block["ceiling"], block["bootstrap"], block["k1"]
    lo, hi = k1["ci_lower"], k1["ci_upper"]
    half = 0.5 * (hi - lo)
    loco = block["cross_validated"]
    cv = f"{loco['balanced_accuracy']:.4f}" if loco.get("available") else "n/a"
    return (f"{name:<3} {stratum:<10} {regime:<7} {ceiling['balanced_accuracy']:.4f} "
            f"[{lo:.4f}, {hi:.4f}] half={half:.4f} tau*={ceiling['tau_star']:.4f} "
            f"n_S={ceiling['n_supersession']:<5} n_I={ceiling['n_interference']:<5} "
            f"clusters={boot['n_clusters']:<3} loco={cv} "
            f"{'FIRES' if k1['fires'] else 'does not fire'}"
            f"{' CONCLUSIVE' if k1['interval_conclusive'] else ' STRADDLES 0.90'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=str(Path(RESULTS_DIR) / "task-007"))
    args = parser.parse_args(argv)

    before = shas()
    synthetic_sha = assert_synthetic_corpus_unchanged()
    v1_sha = assert_relations_v1_unchanged()
    assert synthetic_sha == SYNTHETIC_CONTENT_SHA256 and v1_sha == RELATIONS_V1_CONTENT_SHA256
    print("BEFORE")
    print(f"  synthetic    content_sha256={synthetic_sha}")
    print(f"  relations v1 content_sha256={v1_sha}")
    for name, sha in sorted(before.items()):
        print(f"  dir sha {name:<13} {sha}")

    seeds = tuple(range(max(1, int(args.seeds))))
    payloads: dict[str, dict] = {}
    for name, path in SPLITS.items():
        corpus = load_relations(path)  # verifies the content hash, raises on drift
        run_on = D.subsample(corpus) if args.smoke else corpus
        print(f"\n=== {name}: {path} sha={corpus.content_hash[:12]} "
              f"episodes={len(run_on.episodes)} sample_pairs={len(run_on.pairs)}", flush=True)
        result = run_experiment(run_on, n_resamples=int(args.resamples), seeds=seeds,
                                progress=lambda m: print(f"  {m}", flush=True))
        after_split = shas()
        payload = build_payload(
            result, corpus=run_on, synthetic_sha=synthetic_sha, smoke=bool(args.smoke),
            command=f"task-007 driver, split={name}, seeds={list(seeds)}, "
                    f"resamples={args.resamples}",
            dir_shas={"directory_sha_before": before, "directory_sha_after": after_split,
                      "directories_unchanged": after_split == before,
                      "split": name, "split_dir": str(path),
                      "relations_v1_content_sha256": v1_sha,
                      "measured_content_sha256": corpus.content_hash},
            n_resamples=int(args.resamples), seeds=seeds)
        assert payload["corpus"]["directories_unchanged"], "a corpus directory changed mid-run"
        payload["task_007"] = {
            "split": name, "split_dir": str(path),
            "measured_relations_content_sha256": corpus.content_hash,
            "relations_v1_content_sha256": v1_sha,
            "synthetic_content_sha256": synthetic_sha,
            "gate_module_modified": False,
            "note": "the gate module is byte-identical to task 006; only the corpus it reads moved",
        }
        payloads[name] = payload
        json_path, md_path = write_report(payload, args.results_dir)
        print(f"  wrote {json_path}\n  wrote {md_path}")

    after = shas()
    print("\nAFTER")
    for name, sha in sorted(after.items()):
        flag = "UNCHANGED" if after[name] == before[name] else "*** CHANGED ***"
        print(f"  dir sha {name:<13} {sha} {flag}")
    assert after == before, "a corpus directory changed during the run"
    assert assert_synthetic_corpus_unchanged() == SYNTHETIC_CONTENT_SHA256
    assert assert_relations_v1_unchanged() == RELATIONS_V1_CONTENT_SHA256

    print("\nSIDE BY SIDE (derivable population, the gate's primary path)")
    for stratum in ("same_topic", "marginal"):
        for regime in ("long", "short", "pooled"):
            for name in ("v1", "v2"):
                print("  " + line(name, payloads[name], stratum, regime))
            print("")
    summary = {name: {s: {r: cell(p, s, r) for r in ("long", "short", "pooled")}
                      for s in ("same_topic", "marginal")} for name, p in payloads.items()}
    out = Path(args.results_dir) / "task-007-side-by-side.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
