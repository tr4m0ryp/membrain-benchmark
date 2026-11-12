"""CLI entry point: run some or all of the four statistics, checkpoint per seed.

The actual work is split by concern (file-discipline rule, not stylistic):
`scratch_store.ScratchStore` is the resume cache, `draw_workers` runs one
statistic's N draws, `payload_builder` turns draws into the result payload.
This module is only the glue: argument parsing and the run order.

`--only` lets independent statistics run as separate CONCURRENT processes
against the same `--scratch-dir` and the same `--stamp` (so they checkpoint
one combined result file instead of one each) -- useful because F21 and F23
each cost tens of minutes per live draw and have no reason to serialize
behind each other. Every process still re-reads every statistic's scratch
files at every checkpoint, so the combined file always reflects the latest
state across all of them, not just the one this process is computing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from brain.config import results_dir
from brain.experiments.draw_variance import integrity
from brain.experiments.draw_variance.draw_workers import (
    F21_BLOCKED_SENTINEL,
    read_f21_blocked,
    run_f17a,
    run_f17b,
    run_f21,
    run_f23,
    run_statistic,
    write_f21_blocked,
)
from brain.experiments.draw_variance.payload_builder import (
    assemble_payload,
    load_all_from_scratch,
    write_report,
)
from brain.experiments.draw_variance.preregistration import SEEDS, STATISTICS
from brain.experiments.draw_variance.scratch_store import ScratchStore

__all__ = ["ScratchStore", "run_statistic", "run_f17a", "run_f17b", "run_f21", "run_f23",
           "assemble_payload", "write_report", "F21_BLOCKED_SENTINEL", "read_f21_blocked", "main"]

_ALL_STATS = ("f23", "f17a", "f17b", "f21")


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m brain.experiments.draw_variance.runner")
    parser.add_argument("--scratch-dir", default="/tmp/brain-draw-variance-scratch")
    parser.add_argument("--results-dir", default=str(results_dir("draw-variance")))
    parser.add_argument("--skip-f21", action="store_true", help="mark F21 BLOCKED without attempting it")
    parser.add_argument(
        "--only", default=",".join(_ALL_STATS),
        help="comma-separated subset of f23,f17a,f17b,f21 to run in this process",
    )
    parser.add_argument(
        "--stamp", default=None,
        help="UTC timestamp naming the shared result file; pass the SAME value to "
             "every concurrent --only process so they checkpoint one combined file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parse(raw)
    command = "python -m brain.experiments.draw_variance.runner " + " ".join(raw)
    say = lambda message: print(message, flush=True)  # noqa: E731
    only = {token.strip() for token in args.only.split(",") if token.strip()}

    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    say(f"pre-registration: N={len(SEEDS)} seeds={list(SEEDS)} statistics="
        f"{[s['id'] for s in STATISTICS]}; this process runs {sorted(only)}")
    before = integrity.snapshot()
    say(f"frozen-corpus directory hashes before: {before}")

    f21_blocked: str | None = None if "f21" in only else read_f21_blocked(args.scratch_dir)

    def checkpoint(label: str) -> None:
        results = load_all_from_scratch(args.scratch_dir)
        current_blocked = f21_blocked if "f21" in only else read_f21_blocked(args.scratch_dir)
        payload = assemble_payload(
            before=before, results=results, f21_blocked=current_blocked,
            started_at=started_at, command=command, verify_integrity=False,
        )
        json_path, _ = write_report(payload, args.results_dir, stamp)
        say(f"[checkpoint {label}] wrote {json_path}")

    # Priority order (coordinator instruction): F23 first -- it is the result
    # that would hurt most if it moved -- then the cheap no-LLM F17 pair, then
    # F21 last. `on_seed=checkpoint` writes the combined file after EVERY seed.
    if "f23" in only:
        say("\n=== F23: consolidation delta (PRIORITY) ===")
        run_f23(SEEDS, ScratchStore(args.scratch_dir, "F23_consolidation_delta"), say,
                on_seed=lambda: checkpoint("F23"))

    if "f17a" in only:
        say("\n=== F17a: merge harm, synthetic corpus ===")
        run_f17a(SEEDS, ScratchStore(args.scratch_dir, "F17a_merge_harm_synthetic"), say,
                 on_seed=lambda: checkpoint("F17a"))

    if "f17b" in only:
        say("\n=== F17b: merge harm, relations-v2 corpus ===")
        run_f17b(SEEDS, ScratchStore(args.scratch_dir, "F17b_merge_harm_relations_v2"), say,
                 on_seed=lambda: checkpoint("F17b"))

    if "f21" in only:
        say("\n=== F21: memory precision ===")
        if args.skip_f21:
            f21_blocked = "--skip-f21 passed"
            write_f21_blocked(args.scratch_dir, f21_blocked)
        else:
            _, f21_blocked = run_f21(
                SEEDS, ScratchStore(args.scratch_dir, "F21_memory_precision"), say,
                scratch_dir=args.scratch_dir, on_seed=lambda: checkpoint("F21"),
            )
        if f21_blocked:
            say(f"F21 BLOCKED: {f21_blocked}")
        checkpoint("F21 (final)")

    results = load_all_from_scratch(args.scratch_dir)
    f21_blocked = f21_blocked if "f21" in only else read_f21_blocked(args.scratch_dir)
    payload = assemble_payload(
        before=before, results=results, f21_blocked=f21_blocked,
        started_at=started_at, command=command, verify_integrity=True,
    )
    json_path, md_path = write_report(payload, args.results_dir, stamp)
    say(f"\nFINAL (this process's view): wrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
