"""CLI: `python -m brain.experiments.gate_separation --seeds 5 --k 5 [--smoke]`.

Runs every arm on the frozen corpus, sweeps the separation strength, evaluates
the pre-registered H-A kill criterion and writes `results/gate-separation/gate-<stamp>.{json,md}`.

`--smoke` swaps in the 2-cluster smoke corpus and a short sweep so the whole
pipeline can be validated on CPU in well under a minute. It leaves the LLM arm
out, which is why a smoke run correctly reports INCONCLUSIVE: the criterion names
full-context, and a verdict without it is not the pre-registered one.

The full-context arm costs real money on a COLD cache (about 13 USD and an hour
over 360 queries at 6 workers) and replays for free afterwards from
`data/cache/full_context/`. `--no-full-context` skips it, at the price of an
undecided verdict by design.
"""

from __future__ import annotations

import argparse
import shlex
import sys

from brain.config import CORPUS_DIR, results_dir
from brain.corpus import load_corpus
from brain.eval.runner import DEFAULT_K
from brain.experiments.gate_separation.gate import run_gate
from brain.experiments.gate_separation.report import write_gate_report
from brain.experiments.gate_separation.specs import BASELINE_ARMS, S_GRID, SMOKE_S_GRID

__all__ = ["main"]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m brain.experiments.gate_separation")
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds, 0..N-1")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="retrieval depth")
    parser.add_argument("--smoke", action="store_true", help="tiny corpus, short sweep, no LLM arm")
    parser.add_argument("--resamples", type=int, default=None, help="bootstrap resamples")
    parser.add_argument(
        "--no-full-context",
        action="store_true",
        help="skip the LLM arm; the verdict is then undecided by design",
    )
    parser.add_argument("--aggregation", default="sequential", choices=("sequential", "net"))
    parser.add_argument("--no-secondary", action="store_true", help="skip the net diagnostic run")
    parser.add_argument("--write-arms", action="store_true", help="also write per-arm result files")
    parser.add_argument("--results-dir", default=str(results_dir("gate-separation")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(list(sys.argv[1:] if argv is None else argv))
    command = "python -m brain.experiments.gate_separation " + " ".join(
        shlex.quote(a) for a in (sys.argv[1:] if argv is None else argv)
    )
    directory = CORPUS_DIR / ("synthetic-smoke" if args.smoke else "synthetic")
    # Loading through the loader verifies the content hash; a drifted corpus
    # raises here rather than producing an unattributable number.
    corpus = load_corpus(directory)
    seeds = tuple(range(max(1, int(args.seeds))))
    baselines = [
        name
        for name in BASELINE_ARMS
        if not (name == "full_context" and (args.smoke or args.no_full_context))
    ]
    print(
        f"corpus={directory.name} sha={corpus.content_hash[:12]} "
        f"episodes={len(corpus.episodes)} queries={len(corpus.queries)} "
        f"clusters={len(corpus.clusters())} k={args.k} seeds={list(seeds)} "
        f"arms={baselines} aggregation={args.aggregation}",
        flush=True,
    )
    run = run_gate(
        corpus,
        seeds=seeds,
        k=int(args.k),
        n_resamples=int(args.resamples) if args.resamples else (2000 if args.smoke else 10_000),
        baselines=baselines,
        grid=SMOKE_S_GRID if args.smoke else S_GRID,
        aggregation=args.aggregation,
        secondary_aggregation=None if (args.no_secondary or args.smoke) else "net",
        write_arms=bool(args.write_arms),
        results_dir=args.results_dir,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    json_path, md_path, payload = write_gate_report(
        run, command=command, directory=args.results_dir
    )
    print("")
    print(payload["verdict_line"], flush=True)
    for label in ("selected", "preregistered"):
        verdict = run.verdicts.get(label)
        if verdict is None:
            continue
        print(f"  [{label}] {verdict.candidate} -> {verdict.status}")
        for comparison in verdict.comparisons:
            print(
                f"    vs {comparison.baseline:<16} delta={comparison.delta * 100:+6.2f}pts "
                f"margin={comparison.meets_margin} disjoint={comparison.intervals_disjoint} "
                f"passes={comparison.passes}"
            )
        for reason in verdict.reasons:
            print(f"    reason: {reason}")
    for name, reason in sorted(run.blocked.items()):
        print(f"  BLOCKED {name}: {reason}")
    print(f"\n{json_path}\n{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
