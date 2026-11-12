"""CLI: `python -m brain.experiments.gate_k_sweep`.

Runs the full pre-registered grid (plain_vector and bm25 x k in
{1,2,3,5,10,20,50}, 5 seeds, cluster-BCa B=10000) on the frozen synthetic
corpus and writes `results/ksweep-<UTC>.{json,md}`. Takes no flags that would
let the grid, the metric list or the corpus be changed at the command line --
those are pre-registered in `registration.py`, not runtime options, which is
the point.
"""

from __future__ import annotations

import sys

from brain.config import results_dir

from brain.experiments.gate_k_sweep.report import write_sweep_report
from brain.experiments.gate_k_sweep.sweep import run

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = "python -m brain.experiments.gate_k_sweep " + " ".join(args)

    result = run(progress=lambda message: print(f"  {message}", flush=True))
    json_path, md_path, payload = write_sweep_report(result, command=command, directory=results_dir("k-sweep"))

    integrity = payload["corpus_integrity"]
    print(f"\ncorpus sha unchanged: {integrity['unchanged']}")
    for arm, row in payload["answers"]["q1_recall_saturation"].items():
        print(
            f"Q1 {arm}: ceiling={row['ceiling_at_max_k']:.4f} "
            f"plateau_k={row['plateau_k_within_1pt']}"
        )
    for arm, row in payload["answers"]["q2_disambiguation_monotonicity"].items():
        print(
            f"Q2 {arm}: non_increasing={row['non_increasing']} "
            f"total_drop={row['total_drop']:.4f}"
        )
    print(f"\n{json_path}\n{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
