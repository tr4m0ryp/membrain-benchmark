"""`python -m brain.experiments.convention_external`.

`--plan` prices the whole sweep without calling a model, which is the mandated
first step: a run whose projection passes the budget is reported, not spent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from brain.baselines.vectors import shared_vector_cache
from brain.config import ensure_dir, results_dir
from brain.consolidate.grouping import plan_groups
from brain.consolidate.prefetch import warm_cache
from brain.corpus.external import load_locomo
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.eval.results import jsonable
from brain.experiments.convention_external.design import THRESHOLDS
from brain.experiments.convention_external.report import render
from brain.experiments.convention_external.runner import CACHE_DIR, measure, store_for

__all__ = ["main", "plan"]


def plan(thresholds: tuple[float, ...] = THRESHOLDS) -> dict[str, Any]:
    """Merge groups per threshold, per conversation. No model call, no spend."""
    cache = shared_vector_cache()
    convs = load_locomo()
    cache.ensure([e.text for c in convs for e in c.episodes])
    out: dict[str, Any] = {"per_conversation": {}, "groups_per_threshold": {}}
    for conversation in convs:
        store = store_for(conversation, cache)
        try:
            groups = plan_groups(store, thresholds=thresholds)
        finally:
            store.close()
        out["per_conversation"][conversation.sample_id] = {
            f"{tau:g}": len(items) for tau, items in groups.items()
        }
        for tau, items in groups.items():
            key = f"{tau:g}"
            out["groups_per_threshold"][key] = out["groups_per_threshold"].get(key, 0) + len(items)
    out["calls_total"] = sum(out["groups_per_threshold"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brain.experiments.convention_external")
    parser.add_argument("--plan", action="store_true", help="price the sweep and exit")
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(THRESHOLDS))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--conversations", type=int, default=0,
                        help="0 means all ten; any other value must be pre-registered")
    parser.add_argument("--warm", type=int, default=0, metavar="WORKERS",
                        help="fill the merge cache with N concurrent calls before measuring")
    parser.add_argument("--warm-only", action="store_true",
                        help="stop after warming; the measurement is never run on a subset")
    parser.add_argument("--warm-reverse", action="store_true",
                        help="warm the conversations last-first, so a second process can "
                             "meet a first one in the middle of a shared cache")
    args = parser.parse_args(argv)
    taus = tuple(sorted({float(t) for t in args.thresholds}))

    if args.plan:
        print(json.dumps(plan(taus), indent=2, sort_keys=True))
        return 0

    convs = load_locomo()
    if args.conversations:
        convs = convs[: int(args.conversations)]
    cache = shared_vector_cache()
    cache.ensure([e.text for c in convs for e in c.episodes])
    if args.warm:
        # Warming changes the wall clock and nothing else: the cache is keyed by
        # the exact request, so a warmed merge is the one the pass would make.
        order = list(reversed(convs)) if args.warm_reverse else list(convs)
        for conversation in order:
            store = store_for(conversation, cache)
            try:
                summary = warm_cache(store, thresholds=taus, cache_dir=CACHE_DIR,
                                     workers=int(args.warm))
            finally:
                store.close()
            print(f"warmed {conversation.sample_id}: {summary}", flush=True)
        if args.warm_only:
            return 0
    report = measure(thresholds=taus, n_resamples=int(args.resamples), conversations=convs,
                     cache=cache)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ensure_dir(results_dir("convention-external"))
    path = root / f"convention-external-{stamp}.json"
    path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = path.with_suffix(".md")
    markdown.write_text(render(report), encoding="utf-8")
    print(render(report))
    print(f"wrote {path}\nwrote {markdown}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
