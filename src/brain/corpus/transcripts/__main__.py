"""CLI: `python -m brain.corpus.transcripts [--root DIR] [--limit N]`.

Prints counts only. Deliberately never prints episode text, so a run captured in
a terminal log or a CI transcript cannot leak what it just scrubbed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brain.corpus.transcripts.mine import DEFAULT_PER_STRATUM, MiningConfig
from brain.corpus.transcripts.pipeline import DEFAULT_ROOT, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="brain.corpus.transcripts", description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="session store to read (default: <repo>/sessions)")
    parser.add_argument("--limit", type=int, default=None, help="read at most N sessions")
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM,
                        help="review budget per stratum per regime")
    parser.add_argument("--dry-run", action="store_true", help="measure but write nothing")
    args = parser.parse_args()

    payload = run_pipeline(
        args.root,
        limit=args.limit,
        mining=MiningConfig(per_stratum=args.per_stratum),
        write=not args.dry_run,
    )
    summary = {
        "run_id": payload["run_id"],
        "UNVALIDATED": payload["warning"],
        "sessions": payload["inventory"]["n_sessions"],
        "genres": payload["inventory"]["genres"],
        "episodes_extracted": payload["extraction"]["n_extracted"],
        "episodes_surviving": payload["extraction"]["n_survivors"],
        "quarantined": payload["extraction"]["n_quarantined"],
        "scan_blocked": payload["scan"]["pre_pseudonymisation"]["n_blocked"],
        "scan_flagged": payload["scan"]["pre_pseudonymisation"]["n_flagged"],
        "pairs_for_review": payload["mining"]["n_pairs"],
        "labelling_hours": payload["labelling_effort"]["hours_total"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
