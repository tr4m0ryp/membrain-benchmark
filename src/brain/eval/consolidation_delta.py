"""Consolidation delta (I5b): memory accuracy on a fixed store, before vs after.

No vendor publishes an accuracy measurement for its consolidation pass, so
nobody has shown consolidation is net positive for memory at all. Measured here
on the frozen corpus through the plain-vector arm, everything except the pass
held constant: `delta(tau, mode) = acc(after ONE pass) - acc(before)`.

**Merges are credited, never scored as an id change.** Task 003 returns
`replacements`, original id -> the consolidated episode standing for it, and
every query's correct trace AND its named siblings are read through it, so an
answer now inside a merged episode is found, not missed by definition.

**Three readings, because the crediting convention moves the number more than
the pass does.** After a merge a query's answer and one of its own distractors
can be the SAME record, so "the correct trace outranks every sibling" has no
single honest reading. `credited` treats an absorbed sibling as gone (upper
bound, generous to the baseline), `strict` treats it as a loss (lower bound),
and the headline `expected` scores `1/(1+a)` over the `a+1` events that record
conflates: the accuracy of a reader who must pick one. `answer_recall@1` is a
fourth metric needing no crediting convention at all.

**Two confounds controlled, not noted.** *Store size*: destructive shrinks the
store and non-destructive grows it, and difficulty moves with size, so every
metric is read at k=5 and at k=|store|. At full depth the order of a query's
answer against its own siblings cannot depend on which other episodes exist, so
`@full` is store-size free by construction and `@5 - @full` is the displacement
the size change caused. *Length*: merged episodes run 1.7x to 3.4x longer and
the embedder truncates at 256 word pieces, so the placebo arms rewrite the
episodes a merge would have consumed to that length WITHOUT merging content and
at unchanged store size (`placebo_shared` pads with neutral boilerplate,
`placebo_self` repeats the episode): their delta is the length artifact alone.

The two modes attribute the loss: non-destructive only adds the summary and
destructive also retires the originals, so the difference is merging alone. ONE
pass per condition, never iterated: the shipped default is deliberately not
idempotent and repairing it would stop it being what is measured. Run with
`.venv/bin/python -m brain.eval.consolidation_delta`, offline and free.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Mapping, NamedTuple, Sequence

from brain.baselines.plain_vector import PlainVectorRetriever
from brain.baselines.vectors import VectorCache
from brain.config import SEED, SMOKE_TEST, get_rng, results_dir, seed_everything
from brain.consolidate import run_consolidation
from brain.consolidate.smoke import FROZEN_CORPUS_SHA, source_store
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci, paired_delta_ci
from brain.eval.results import corpus_provenance, write_result
from brain.store import Trace

__all__ = ["DELTA_SCHEMA", "METRICS", "MODES", "SEEDS", "THRESHOLDS", "TOP_K", "Condition",
           "answer_tokens", "condition_scores", "conditions_for", "measure",
           "placebo_traces", "query_indicators"]

DELTA_SCHEMA = "consolidation-delta-v1"
#: Task 003's sweep. The delta is a curve over this, never a single point.
THRESHOLDS: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.95)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
#: Depth for the primary reading; `@full` re-reads at k = |store|, where the
#: store-size confound cannot act.
TOP_K = 5
MODES: tuple[str, ...] = ("destructive", "non_destructive", "placebo_shared", "placebo_self")
METRICS: tuple[str, ...] = ("expected@5", "credited@5", "strict@5", "expected@full",
                            "credited@full", "strict@full", "answer_recall@1")
#: Content-free padding: the shared-boilerplate term that dominates surface
#: similarity in Theorem 1, carrying no cluster-specific fact.
FILLER = ("Session log padding. Routine notes captured automatically during the working "
          "session and kept verbatim for later recall. No new finding is recorded here; "
          "it repeats the boilerplate every capture already carries.")
_WORD = re.compile(r"[a-z0-9#/._-]+")
_STOP = frozenset(("the a an of to in on for and or is was were be been it this that with "
                   "at from by as its into not no had has have but when which what why how "
                   "did do").split())


def _tokens(text: str) -> set[str]:
    """Lowercased content words; stop words and single characters dropped."""
    return {w for w in _WORD.findall(str(text).lower()) if len(w) > 1 and w not in _STOP}


def answer_tokens(query: Any) -> set[str]:
    """The answer's content words, its `axis:` labels stripped off."""
    return _tokens(" ".join(p.split(":", 1)[-1] for p in str(query.answer).split(";")))


class Condition(NamedTuple):
    """One store to score: episodes, merge crediting map, provenance.

    `name` is `mode@tau`, which is what the report and the curve are keyed on."""
    name: str
    traces: list[Trace]
    replacements: dict[str, str]
    stats: dict[str, Any]


def query_indicators(query: Any, ranked: Sequence[str],
                     replacements: Mapping[str, str], k: int) -> dict[str, float]:
    """The three readings for one query, generalising disambiguation accuracy.

    A record stands for an event when it IS that episode or is the consolidated
    episode replacing it, so in non-destructive mode both survive and the arm
    keeps the credit its surviving originals earn. Walking down the ranking, the
    first record standing only for a sibling ends the query; the best answer
    record above it wins and `absorbed` is how many of the query's own siblings
    that record also conflates. Without merges every record stands for one event
    and this is exactly `score_query` (asserted in the tests).
    """
    correct = str(query.correct_trace_id)
    distractors = {str(d) for d in query.distractor_ids}
    stands_for: dict[str, set[str]] = {}
    for event in sorted(distractors | {correct}):
        for record in (event, str(replacements.get(event, event))):
            stands_for.setdefault(record, set()).add(event)
    best: int | None = None
    for record in list(ranked)[:k]:
        held = stands_for.get(str(record))
        if not held:
            continue
        if correct not in held:
            break  # a sibling-only record: nothing below it can win the query
        absorbed = len(held & distractors)
        best = absorbed if best is None else min(best, absorbed)
        if best == 0:
            break
    if best is None:  # eligibility is pinned to the before store: a loss, not a drop
        return {"credited": 0.0, "expected": 0.0, "strict": 0.0}
    return {"credited": 1.0, "expected": 1.0 / (1.0 + best), "strict": 0.0 if best else 1.0}


def condition_scores(condition: Condition, queries: Sequence[Any],
                     answers: Mapping[str, set[str]], *, k: int = TOP_K,
                     cache: VectorCache | None = None) -> dict[str, dict[str, float]]:
    """Index, answer every query, return metric -> query id -> value.

    The full ranking is computed once and `@5` is its first k entries."""
    text_of = {t.id: t.text for t in condition.traces}
    arm = PlainVectorRetriever(cache=cache)
    try:
        arm.index(condition.traces)
        ranks = [arm.retrieve(q.text, len(condition.traces)) for q in queries]
    finally:
        arm.close()
    scores: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    for query, ranked in zip(queries, ranks):
        qid, want = str(query.query_id), answers[str(query.query_id)]
        for depth, label in ((k, "5"), (len(ranked), "full")):
            for n, v in query_indicators(query, ranked, condition.replacements, depth).items():
                scores[f"{n}@{label}"][qid] = v
        top = _tokens(text_of.get(ranked[0], "")) if ranked else set()
        scores["answer_recall@1"][qid] = len(want & top) / len(want) if want else math.nan
    return scores


def _pad(text: str, target: int, filler: str | None) -> str:
    """Lengthen `text` to `target` chars with content carrying no fact.

    `filler=None` repeats the episode, which can neither add nor remove
    information; a shared filler adds boilerplate. The two bracket the effect."""
    if target <= len(text):
        return text
    tail = text if filler is None else filler
    out = text
    while len(out) < target:
        out = f"{out}\n{tail}"
    return out[:target].rsplit(" ", 1)[0]


def placebo_traces(traces: Sequence[Trace], result: Any,
                   *, filler: str | None = FILLER) -> list[Trace]:
    """The same episodes rewritten to the merged length, with nothing merged."""
    size = {m: len(getattr(result.store.get(c), "text", "") or "")
            for m, c in result.replacements.items()}
    return [replace(t, text=_pad(t.text, size[t.id], filler), content_sha="")
            if t.id in size else t for t in traces]


def conditions_for(store: Any, *, thresholds: Sequence[float] = THRESHOLDS,
                   seed: int = SEED, **kwargs: Any) -> list[Condition]:
    """Before, then four arms per threshold. Exactly ONE pass per (tau, mode)."""
    base = store.all_traces(valid_only=True)
    out = [Condition("before", list(base), {}, {"n_after": len(base)})]
    for tau in sorted({float(t) for t in thresholds}):
        runs = {m: run_consolidation(store, seed, threshold=tau, **kwargs,
                destructive=(m == "destructive")) for m in ("destructive", "non_destructive")}
        hard = runs["destructive"]
        lens = sorted(len(getattr(hard.store.get(i), "text", "") or "") for i in hard.merged_ids)
        for mode, res in runs.items():
            out.append(Condition(f"{mode}@{tau:g}", list(res.traces), dict(res.replacements),
                                 {**res.stats, "spend": res.spend.to_dict(),
                                  "merged_chars_median": lens[len(lens) // 2] if lens else 0}))
        for mode, filler in (("placebo_shared", FILLER), ("placebo_self", None)):
            padded = placebo_traces(base, hard, filler=filler)
            out.append(Condition(f"{mode}@{tau:g}", padded, {},
                {"n_after": len(padded), "n_rewritten": len(hard.replacements),
                 "chars_median": sorted(len(t.text) for t in padded)[len(padded) // 2]}))
        for result in runs.values():
            result.store.close()
    return out


def _cluster_groups(seed_runs: Sequence[Mapping[str, Mapping[str, float]]],
                    cluster_of: Mapping[str, str]) -> dict[str, dict[str, list[float]]]:
    """Metric -> cluster -> indicators, seeds averaged per query first, which
    keeps the CLUSTER the resampling unit (never the query, F11)."""
    out: dict[str, dict[str, list[float]]] = {}
    for metric in seed_runs[0]:
        groups: dict[str, list[float]] = {}
        for qid in sorted(seed_runs[0][metric]):
            value = math.fsum(r[metric][qid] for r in seed_runs) / len(seed_runs)
            if math.isfinite(value):  # answer recall is NaN on an empty answer
                groups.setdefault(cluster_of[qid], []).append(value)
        out[metric] = {c: groups[c] for c in sorted(groups)}
    return out


def measure(*, thresholds: Sequence[float] = THRESHOLDS, seeds: Sequence[int] = SEEDS,
            k: int = TOP_K, n_resamples: int = DEFAULT_RESAMPLES, corpus: Any = None,
            store: Any = None, cache: VectorCache | None = None,
            **kwargs: Any) -> dict[str, Any]:
    """Run the sweep and return the report. Paired, cluster-bootstrapped, >= 5 seeds."""
    if corpus is None:
        from brain.corpus.loader import load_corpus
        corpus = load_corpus()
    if corpus.content_hash and corpus.content_hash != FROZEN_CORPUS_SHA:
        raise RuntimeError(f"frozen corpus changed: {corpus.content_hash}")
    source = store if store is not None else source_store(corpus)
    queries = sorted(corpus.queries, key=lambda q: str(q.query_id))
    answers = {str(q.query_id): answer_tokens(q) for q in queries}
    cluster_of = {str(q.query_id): str(q.cluster_id) for q in queries}
    runs: dict[str, list[dict[str, dict[str, float]]]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for seed in sorted({int(s) for s in seeds}):
        seed_everything(seed)
        for cond in conditions_for(source, thresholds=thresholds, seed=seed, **kwargs):
            runs.setdefault(cond.name, []).append(
                condition_scores(cond, queries, answers, k=k, cache=cache))
            stats[cond.name] = dict(cond.stats)
    groups = {n: _cluster_groups(r, cluster_of) for n, r in runs.items()}
    base = groups["before"]

    def cell(name: str, metric: str) -> dict[str, Any]:
        # One rng stream per METRIC, not per condition: every condition then
        # sees the same resampled corpora, so the curve compares arms, not draws.
        return {"accuracy": cluster_bootstrap_ci(
                    groups[name][metric], n_resamples=n_resamples,
                    rng=get_rng(SEED, stream=f"d.{metric}")).to_dict(),
                "delta_vs_before": paired_delta_ci(
                    groups[name][metric], base[metric], n_resamples=n_resamples,
                    rng=get_rng(SEED, stream=f"d.paired.{metric}")).to_dict()}

    return {
        "schema": DELTA_SCHEMA, "corpus": corpus_provenance(corpus),
        "config": {
            "thresholds": sorted({float(t) for t in thresholds}),
            "seeds": sorted({int(s) for s in seeds}), "k": int(k), "arm": "plain_vector",
            "passes_per_condition": 1, "linkage": kwargs.get("linkage", "complete"),
            "bootstrap_resamples": int(n_resamples),
            "resampling_unit": "interference cluster",
            "pairing": "same clusters, same queries, eligibility pinned to the before store",
            "seeds_identical": all(all(r == s[0] for r in s) for s in runs.values())},
        "conditions": {n: {"stats": stats[n], "metrics": {m: cell(n, m) for m in METRICS}}
                       for n in runs}}


def main() -> dict[str, Any]:
    """Verify the corpus, run the sweep, print the curve, write the result file."""
    from brain.corpus.loader import load_corpus
    corpus = load_corpus()
    print(f"frozen corpus SHA {corpus.content_hash} (verified, read-only)")
    report = measure(corpus=corpus, seeds=(0,) if SMOKE_TEST else SEEDS,
                     thresholds=(0.85,) if SMOKE_TEST else THRESHOLDS,
                     n_resamples=200 if SMOKE_TEST else DEFAULT_RESAMPLES)
    conditions = report["conditions"]
    taus = sorted({float(name.split("@")[1]) for name in conditions if "@" in name})
    print("\nepisodes after the pass: "
          f"{ {n: c['stats'].get('n_after') for n, c in conditions.items()} }")
    for metric in METRICS:
        print(f"\ndelta {metric} (paired cluster BCa, 95 percent)")
        print("   tau" + "".join(f"{mode:>27}" for mode in MODES))
        for tau in taus:
            cells = [conditions.get(f"{m}@{tau:g}", {}).get("metrics", {}).get(metric)
                     for m in MODES]
            print(f"  {tau:<4g}" + "".join("".rjust(27) if c is None else
                "{mean:+.3f} [{lower:+.3f},{upper:+.3f}]".format(**c["delta_vs_before"]).rjust(27)
                for c in cells))
    spend = [c["stats"].get("spend", {}) for c in conditions.values()]
    print(f"\nlive api calls {sum(int(s.get('api_calls', 0)) for s in spend)}, spend "
          f"{sum(float(s.get('cost_usd', 0.0)) for s in spend):.4f} USD, seeds identical: "
          f"{report['config']['seeds_identical']}\nwrote "
          f"{write_result(report, arm='consolidation_delta', directory=results_dir("consolidation"))}")
    return report


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
