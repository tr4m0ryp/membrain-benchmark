"""Run the harness from the command line: `python -m brain.eval [--arms ...]`.

Prints, per arm, the disambiguation accuracy with its CLUSTER bootstrap interval,
the across-seed interval beside it (zero-width for a deterministic arm, which is
the point), and the accuracy-vs-cluster-size curve M-A predicts on.

`BRAIN_SMOKE_TEST=1` switches to the 2-cluster smoke corpus and a small number of
resamples, so the whole pipeline can be validated on CPU in seconds. The
full-context arm is opt-in (`--full-context`) because a cold pass costs real
money; it replays from `data/cache/full_context/` once warmed.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from brain.config import CORPUS_DIR, SMOKE_TEST
from brain.corpus import load_corpus
from brain.eval.runner import DEFAULT_K, DEFAULT_SEEDS, ArmBlocked, run_arm
from brain.eval.verdict import evaluate_kill_criterion

LOCAL_ARMS = ("plain_vector", "bm25", "write_time_mmr")


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m brain.eval")
    parser.add_argument("--arms", default=",".join(LOCAL_ARMS))
    parser.add_argument("--full-context", action="store_true", help="add the LLM arm (costs money cold)")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--resamples", type=int, default=2000 if SMOKE_TEST else 10_000)
    parser.add_argument("--candidate", default=None, help="arm to judge against the others")
    parser.add_argument("--write", action="store_true", help="write results/<stamp>-<arm>.json")
    return parser.parse_args(argv)


def _report(result: Any) -> None:
    ci = result.bootstrap
    seed_ci = result.aggregate["disambiguation"]
    print(
        f"{result.arm:<16} disambig={ci.mean:.4f} "
        f"cluster95=[{ci.lower:.4f}, {ci.upper:.4f}] "
        f"seed95=[{seed_ci.lower:.4f}, {seed_ci.upper:.4f}] "
        f"seeds={list(result.seeds_run)} k={result.k} {result.wall_clock:.1f}s"
    )
    curve = " ".join(f"m={size}:{point.accuracy:.3f}" for size, point in sorted(result.curve.items()))
    decay = result.decay
    print(
        f"{'':<16} curve {curve} | drop={decay.get('drop', float('nan')):.3f} "
        f"decays={decay.get('decays')} rho={decay.get('m_a_fit', {}).get('rho')}"
    )
    if result.path:
        print(f"{'':<16} -> {result.path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse(list(sys.argv[1:] if argv is None else argv))
    directory = CORPUS_DIR / ("synthetic-smoke" if SMOKE_TEST else "synthetic")
    corpus = load_corpus(directory)
    names = [name for name in args.arms.split(",") if name]
    if args.full_context:
        names.append("full_context")
    seeds = tuple(range(max(1, int(args.seeds))))
    print(f"corpus={directory.name} episodes={len(corpus.episodes)} queries={len(corpus.queries)} "
          f"sha={corpus.content_hash[:8]} smoke={SMOKE_TEST}")
    results: dict[str, Any] = {}
    for name in names:
        try:
            results[name] = run_arm(
                name,
                corpus,
                seeds=seeds,
                k=int(args.k),
                name=name,
                n_resamples=int(args.resamples),
                # The LLM arm has no randomness: one pass, replayed, not five.
                seed_independent=name == "full_context",
                write=bool(args.write),
            )
            _report(results[name])
        except ArmBlocked as exc:
            print(f"{name:<16} BLOCKED: {exc}")
    if args.candidate and args.candidate in results:
        candidate = results.pop(args.candidate)
        baseline = results.get("plain_vector")
        verdict = evaluate_kill_criterion(
            candidate.stats(),
            [result.stats() for result in results.values()],
            baseline_curve=None if baseline is None else baseline.decay,
            curve_arm=None if baseline is None else "plain_vector",
            n_resamples=int(args.resamples),
        )
        print(f"verdict({verdict.candidate}) = {verdict.status}")
        for comparison in verdict.comparisons:
            print(
                f"  vs {comparison.baseline:<14} delta={comparison.delta * 100:+.2f}pts "
                f"margin={comparison.meets_margin} disjoint={comparison.intervals_disjoint} "
                f"passes={comparison.passes}"
            )
        for reason in verdict.reasons:
            print(f"  reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
