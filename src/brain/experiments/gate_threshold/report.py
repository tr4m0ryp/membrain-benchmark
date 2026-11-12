"""Task 006 -- the threshold-ceiling gate (K1). Runs it and writes both artifacts.

CLI: `.venv/bin/python -m brain.experiments.gate_threshold.report [--smoke]`

The headline is the SAME-TOPIC-STRATIFIED ceiling on the derivable population,
per regime. It is primary for the reason task 001 recorded: interference pairs are
100 percent same-workstream while supersession pairs are only 53 percent, and both
effects push toward overlap, which is the direction that favours our own claim.
Stratifying removes that help, so the stratified number is the adversarial one and
the marginal one is a companion, never a substitute.

Long and short are never collapsed into the headline. Task 001 predicted the short
regime would separate cleanly and called the long regime genuinely open, so a
pooled number would hide exactly that split. The pooled cell is computed and
printed, labelled as the number that hides the finding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

from brain.config import CORPUS_DIR, ensure_dir, results_dir
from brain.corpus.relations import INTERFERENCE, REGIMES, SUPERSESSION, load_relations
from brain.corpus.relations.loader import (
    RELATIONS_DIR,
    SYNTHETIC_CONTENT_SHA256,
    assert_synthetic_corpus_unchanged,
)
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.eval.results import environment, jsonable
from brain.experiments.gate_threshold import ceiling as C
from brain.experiments.gate_threshold import distributions as D
from brain.experiments.gate_threshold.markdown import render_markdown
from brain.experiments.gate_veto.report import directory_sha

__all__ = ["SCHEMA", "STRATA", "CELLS", "RELATIONS_SHA", "PAIR_SET_NOTE", "run_experiment",
           "build_payload", "k1_verdict", "write_report", "main"]

SCHEMA = "gate-threshold-result-v1"
STRATA: tuple[str, ...] = ("marginal", "same_topic")
CELLS: tuple[str, ...] = (*REGIMES, "pooled")
RELATIONS_SHA = "aba780bd15ffeb303b065b1812c0f716e75397cae99112ad73b81ee2197b6e6d"
#: Float representation slack only; K1 is a strict "exceeds 0.90".
_TOL = 1e-12

PAIR_SET_NOTE = {
    "comparison": "SUPERSESSION versus INTERFERENCE only. The claim is about those two "
    "operations, so COINCIDENCE and DUPLICATE never enter the ceiling; they appear in the harm "
    "table, where a merge policy does act on them.",
    "primary": "derivable / same_topic, per regime",
    "interference_population": "conventions.interference_selection: same-workstream pairs only, "
    "because a merge policy fires above tau and only colliding pairs ever reach a classifier",
    "coincidence_population": "conventions.coincidence_selection: any same-value, "
    "different-referent pair, no context constraint",
    "sample_vs_derivable": "pairs.jsonl is a labelled SAMPLE; every derivable number relabels "
    "pairs from episodes.jsonl through brain.corpus.relations.pair_of, so absence from the "
    "sample is never treated as a negative label",
    "confound": "task 001: interference pairs are 100 percent same-workstream, supersession only "
    "53 percent. Both effects push toward overlap, which favours our own claim, so the "
    "same-topic-stratified ceiling is primary and the marginal one is a companion",
}


def _subset(rows: Sequence[D.PairRow], cell: str) -> list[D.PairRow]:
    return list(rows) if cell == "pooled" else [r for r in rows if r.regime == cell]


def _cell(rows: Sequence[D.PairRow], *, n_resamples: int, seeds: Sequence[int]) -> dict[str, Any]:
    sup, inter = D.by_class(rows, SUPERSESSION), D.by_class(rows, INTERFERENCE)
    if sup.size == 0 or inter.size == 0:
        return {"available": False, "reason": "a class is empty in this cell"}
    found = C.best_threshold(sup, inter)
    bootstrap = C.ceiling_interval(rows, n_resamples=n_resamples, seeds=seeds)
    return {
        "available": True, "ceiling": found.to_dict(), "bootstrap": bootstrap,
        "bayes_error_any_f": C.overlap_bayes_error(sup, inter),
        "cross_validated": C.loco_balanced_accuracy(rows),
        "overlap": D.support_overlap(sup, inter),
        "k1": k1_verdict("", found, bootstrap),
    }


def run_experiment(
    corpus: Any, *, embedder: Any | None = None, n_resamples: int = DEFAULT_RESAMPLES,
    seeds: Sequence[int] = (0, 1, 2, 3, 4), progress: Any = None,
) -> dict[str, Any]:
    """Encode once, build every population, compute every reported number."""
    say = progress or (lambda _message: None)
    say("encoding the relational corpus once, in sorted trace_id order")
    ids, vectors = D.encode_corpus(corpus, embedder)
    gram = D.gram_matrix(vectors)
    labelled = D.all_pair_rows(corpus, ids, gram)
    sets = {
        "derivable": D.population(labelled, "derivable"),
        "all_interference": D.population(labelled, "all_interference"),
        "sample": D.sample_rows(corpus, ids, gram),
    }
    cells: dict[str, Any] = {}
    verdicts: list[dict[str, Any]] = []
    for name in ("derivable", "sample"):
        cells[name] = {}
        for stratum in STRATA:
            rows = D.stratify(sets[name], same_topic_only=(stratum == "same_topic"))
            cells[name][stratum] = {}
            for cell in CELLS:
                subset = _subset(rows, cell)
                say(f"ceiling: {name}/{stratum}/{cell} ({len(subset)} pairs)")
                block = _cell(subset, n_resamples=n_resamples, seeds=seeds)
                cells[name][stratum][cell] = block
                if block["available"]:
                    block["k1"] |= {"label": f"{name}/{stratum}/{cell}", "population": name,
                                    "stratum": stratum, "cell": cell}
                    verdicts.append(block["k1"])
    say("sensitivity: the literal action map, cross-workstream interference included")
    sensitivity: dict[str, Any] = {}
    for cell in CELLS:
        rows = _subset(sets["all_interference"], cell)
        sup, inter = D.by_class(rows, SUPERSESSION), D.by_class(rows, INTERFERENCE)
        if sup.size and inter.size:
            found = C.best_threshold(sup, inter)
            sensitivity[cell] = {**found.to_dict(),
                                 "fires_k1": bool(found.balanced_accuracy > C.K1_THRESHOLD)}
    say("harm table and expected loss")
    strat = {s: D.stratify(sets["derivable"], same_topic_only=(s == "same_topic")) for s in STRATA}
    return {
        "distributions": {n: D.distribution_table(r) for n, r in sorted(sets.items())},
        "direction": {s: D.direction_report(sets["derivable"], same_topic_only=(s == "same_topic"))
                      for s in STRATA},
        "ceilings": cells, "k1_verdicts": verdicts,
        "sensitivity_all_interference": sensitivity,
        "harm_table": {c: C.harm_table(_subset(sets["derivable"], c)) for c in CELLS},
        "expected_loss": {s: {c: C.expected_loss(_subset(strat[s], c)) for c in CELLS}
                          for s in STRATA},
        "population_sizes": {
            name: {"total": len(rows)} | {
                f"{cls}/{regime}": int(D.by_class(rows, cls, regime).size)
                for cls in sorted({r.cls for r in rows}) for regime in REGIMES
            }
            for name, rows in sorted(sets.items())
        },
    }


def k1_verdict(label: str, ceiling: C.Ceiling, interval: dict[str, Any]) -> dict[str, Any]:
    """K1, computed from the number. A fired criterion IS the result, not a failure."""
    value = float(ceiling.balanced_accuracy)
    ci = interval.get("interval", {})
    lower, upper = float(ci.get("lower", math.nan)), float(ci.get("upper", math.nan))
    fires = bool(value > C.K1_THRESHOLD + _TOL)
    return {
        "label": label, "criterion": C.K1_TEXT, "threshold": C.K1_THRESHOLD,
        "balanced_accuracy": value, "fires": fires,
        "status": "K1 FIRES" if fires else "K1 does not fire",
        "ci_lower": lower, "ci_upper": upper,
        "interval_conclusive": bool(
            math.isfinite(lower) and math.isfinite(upper)
            and (lower > C.K1_THRESHOLD or upper < C.K1_THRESHOLD)
        ),
        "note": "the point estimate against the pre-registered bar, as written. The interval "
                "sits beside it and is not part of the criterion.",
    }


def _headline(verdicts: Sequence[dict[str, Any]]) -> str:
    primary = [v for v in verdicts if v["population"] == "derivable"
               and v["stratum"] == "same_topic" and v["cell"] in REGIMES]
    fired = sorted(v["cell"] for v in primary if v["fires"])
    if not fired:
        return "VERDICT: K1 DOES NOT FIRE on the primary same-topic-stratified ceiling"
    if len(fired) == len(primary):
        return "VERDICT: K1 FIRES on the primary same-topic-stratified ceiling in every regime"
    return ("VERDICT: K1 FIRES on the primary same-topic-stratified ceiling in the "
            f"{', '.join(fired)} regime only")


def build_payload(
    result: dict[str, Any], *, corpus: Any, command: str, synthetic_sha: str,
    dir_shas: dict[str, Any], n_resamples: int, seeds: Sequence[int], smoke: bool = False,
) -> dict[str, Any]:
    """`corpus` is what was MEASURED; under --smoke that is a subset of the frozen
    split, so the counts describe the subset while the SHA still names the source."""
    return {
        "schema": SCHEMA,
        "experiment": "Task 006 -- the threshold-ceiling gate (K1, the minimal falsifying "
                      "experiment)",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": command, "verdict_line": _headline(result["k1_verdicts"]), **result,
        "pair_set_used": PAIR_SET_NOTE,
        "config": {
            "bootstrap_resamples": int(n_resamples), "seeds": list(seeds),
            "resampling_unit": "cluster_id, never the pair",
            "determinism": "the embedder and the exact threshold sweep are deterministic, so the "
            "point estimates are seed independent; seeds vary the bootstrap draw only",
            "sweep": "exact: the optimum is attained at an observed score, so there is no grid",
            "llm_calls": 0, "cost_usd": 0.0, "smoke_subset": bool(smoke),
        },
        "corpus": {
            "relations_content_sha256": corpus.content_hash,
            "relations_sha_expected": RELATIONS_SHA,
            "relations_sha_matches": corpus.content_hash == RELATIONS_SHA,
            "synthetic_content_sha256": synthetic_sha,
            "synthetic_sha_expected": SYNTHETIC_CONTENT_SHA256,
            "synthetic_sha_matches": synthetic_sha == SYNTHETIC_CONTENT_SHA256,
            "n_episodes": len(corpus.episodes), "n_sample_pairs": len(corpus.pairs), **dir_shas,
        },
        "environment": environment(),
    }


def write_report(payload: dict[str, Any], directory: Path | str) -> tuple[Path, Path]:
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = root / f"threshold-{stamp}.json", root / f"threshold-{stamp}.md"
    suffix = 1
    while json_path.exists() or md_path.exists():
        json_path = root / f"threshold-{stamp}-{suffix}.json"
        md_path = root / f"threshold-{stamp}-{suffix}.md"
        suffix += 1
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m brain.experiments.gate_threshold.report")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--smoke", action="store_true", help="tiny-N: 8 pairs per class per regime")
    parser.add_argument("--results-dir", default=str(results_dir("threshold-ceiling")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parse(raw)
    command = "python -m brain.experiments.gate_threshold.report " + " ".join(
        shlex.quote(a) for a in raw)
    synthetic_dir = CORPUS_DIR / "synthetic"
    before = {"relations": directory_sha(RELATIONS_DIR), "synthetic": directory_sha(synthetic_dir)}
    corpus = load_relations()  # verifies the content hash, raises on drift
    synthetic_sha = assert_synthetic_corpus_unchanged()
    if corpus.content_hash != RELATIONS_SHA:
        raise RuntimeError(f"relational corpus SHA changed: {corpus.content_hash}")
    run_on = D.subsample(corpus) if args.smoke else corpus
    seeds = tuple(range(max(1, int(args.seeds))))
    print(f"relations sha={corpus.content_hash[:12]} synthetic sha={synthetic_sha[:12]} "
          f"episodes={len(run_on.episodes)} sample_pairs={len(run_on.pairs)} seeds={list(seeds)} "
          f"resamples={args.resamples} smoke={bool(args.smoke)}", flush=True)
    result = run_experiment(run_on, n_resamples=int(args.resamples), seeds=seeds,
                            progress=lambda message: print(f"  {message}", flush=True))
    after = {"relations": directory_sha(RELATIONS_DIR), "synthetic": directory_sha(synthetic_dir)}
    payload = build_payload(
        result, corpus=run_on, command=command, synthetic_sha=synthetic_sha, smoke=bool(args.smoke),
        dir_shas={"directory_sha_before": before, "directory_sha_after": after,
                  "directories_unchanged": before == after},
        n_resamples=int(args.resamples), seeds=seeds,
    )
    if not payload["corpus"]["directories_unchanged"]:
        raise RuntimeError("a corpus directory changed during the run")
    json_path, md_path = write_report(payload, args.results_dir)
    print("")
    print(render_markdown(payload))
    print(f"{json_path}\n{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
