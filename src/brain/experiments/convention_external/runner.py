"""Run the pre-registered design over LoCoMo and return the report.

The machinery is imported, never reimplemented: `conditions_for` builds the same
before / destructive / non-destructive / placebo arms the synthetic run used,
`query_indicators` applies the same three crediting conventions, and
`paired_delta_ci` produces the same paired BCa cluster bootstrap. What is new
here is only the loop over ten independent per-conversation stores and the two
pre-registered distractor scopes.

One deviation from `brain.eval.consolidation_delta.condition_scores`, made for
cost and not for the result: the ranking is computed ONCE per condition and both
scopes are scored off it, because a distractor set changes what a ranking means
and never what the ranking is.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping, Sequence

from brain.baselines.base import as_traces
from brain.baselines.plain_vector import PlainVectorRetriever
from brain.baselines.vectors import VectorCache, shared_vector_cache
from brain.config import SEED, get_rng, seed_everything
from brain.corpus.external import load_locomo, locomo_provenance, scoped_queries
from brain.corpus.external.locomo import EXTERNAL_DIR
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci, paired_delta_ci
from brain.eval.consolidation_delta import answer_tokens, conditions_for, query_indicators, _tokens
from brain.experiments.convention_external.design import (
    CONVENTIONS, PRE_REGISTRATION, PRIMARY, SCOPES, THRESHOLDS, classify_outcome,
)
from brain.store import EpisodicStore

__all__ = ["CACHE_DIR", "SCHEMA", "METRICS", "measure", "store_for"]

SCHEMA = "convention-external-v1"
METRICS: tuple[str, ...] = tuple(
    f"{c}@{d}" for d in ("5", "full") for c in CONVENTIONS
) + ("answer_recall@1",)
TOP_K = 5
#: LoCoMo merges are cached NEXT TO the download, not in `data/cache/consolidate`.
#: That directory is tracked in git and holds the frozen corpus's merges; a
#: benchmark's responses do not belong in it and must not be committed.
CACHE_DIR = EXTERNAL_DIR / "cache" / "consolidate-locomo"


def store_for(conversation: Any, cache: VectorCache) -> EpisodicStore:
    """One conversation's turns in a store, built the way every arm builds one."""
    traces = as_traces(conversation.episodes)
    store = EpisodicStore(":memory:", cache.dim)
    store.add_many(traces, cache.matrix([t.text for t in traces]))
    if store.count() != len(traces):
        raise RuntimeError(f"{conversation.sample_id}: {len(traces)} traces, {store.count()} rows")
    return store


def _score(condition: Any, queries_by_scope: Mapping[str, Sequence[Any]],
           answers: Mapping[str, set[str]], *, k: int, cache: VectorCache,
           ) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, int]]]:
    """Rank once, then read the ranking under every scope and every convention.

    Returns `(scope -> metric -> query id -> value, scope -> query id -> a)`,
    where `a` is the number of the query's own distractors the winning record
    also absorbed, recovered exactly from the expected convention's `1/(1+a)`.
    """
    text_of = {t.id: t.text for t in condition.traces}
    arm = PlainVectorRetriever(cache=cache)
    try:
        arm.index(condition.traces)
        first = next(iter(queries_by_scope.values()))
        ranks = [arm.retrieve(q.text, len(condition.traces)) for q in first]
    finally:
        arm.close()
    scores: dict[str, dict[str, dict[str, float]]] = {
        scope: {metric: {} for metric in METRICS} for scope in queries_by_scope
    }
    absorbed: dict[str, dict[str, int]] = {scope: {} for scope in queries_by_scope}
    for scope, queries in queries_by_scope.items():
        for query, ranked in zip(queries, ranks):
            qid = str(query.query_id)
            for depth, label in ((k, "5"), (len(ranked), "full")):
                values = query_indicators(query, ranked, condition.replacements, depth)
                for name, value in values.items():
                    scores[scope][f"{name}@{label}"][qid] = value
                if label == "5":
                    absorbed[scope][qid] = (
                        int(round(1.0 / values["expected"] - 1.0)) if values["credited"] else -1
                    )
            want = answers[qid]
            top = _tokens(text_of.get(ranked[0], "")) if ranked else set()
            scores[scope]["answer_recall@1"][qid] = (
                len(want & top) / len(want) if want else math.nan
            )
    return scores, absorbed


def _grouped(values: Mapping[str, float], unit_of: Mapping[str, str]) -> dict[str, list[float]]:
    """Query indicators bucketed by resampling unit, ids sorted so the draw is pinned."""
    out: dict[str, list[float]] = {}
    for qid in sorted(values):
        value = float(values[qid])
        if math.isfinite(value):  # answer recall is NaN on an empty answer
            out.setdefault(unit_of[qid], []).append(value)
    return {key: out[key] for key in sorted(out)}


def _cells(groups: Mapping[str, dict[str, list[float]]],
           base: Mapping[str, dict[str, list[float]]], *, metric: str,
           n_resamples: int) -> dict[str, Any]:
    return {
        "accuracy": cluster_bootstrap_ci(
            groups[metric], n_resamples=n_resamples,
            rng=get_rng(SEED, stream=f"ce.{metric}")).to_dict(),
        "delta_vs_before": paired_delta_ci(
            groups[metric], base[metric], n_resamples=n_resamples,
            rng=get_rng(SEED, stream=f"ce.paired.{metric}")).to_dict(),
    }


def _merge_stats(per_conversation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold ten per-conversation stat blocks into one.

    Counters (`n_merged`, `n_retired`, ...) are summed because the ten stores are
    disjoint. A float (a median character count) is averaged, which is a summary
    and not a median of the pooled population; it is reported as a scale, never
    used in a comparison. Nested values such as `spend` are dropped here and
    accounted for by `_spend`.
    """
    out: dict[str, Any] = {}
    for key in sorted({k for s in per_conversation for k in s}):
        values = [s.get(key) for s in per_conversation if isinstance(s.get(key), (int, float))]
        if values and all(isinstance(v, int) for v in values):
            out[key] = sum(values)
        elif values:
            out[key] = round(sum(float(v) for v in values) / len(values), 4)
    return out


def measure(*, thresholds: Sequence[float] = THRESHOLDS, seed: int = SEED,
            k: int = TOP_K, n_resamples: int = DEFAULT_RESAMPLES,
            conversations: Iterable[Any] | None = None,
            cache: VectorCache | None = None, **kwargs: Any) -> dict[str, Any]:
    """The whole pre-registered run. One pass per (conversation, tau, mode)."""
    started = time.time()
    kwargs.setdefault("cache_dir", CACHE_DIR)
    convs = list(conversations) if conversations is not None else load_locomo()
    vectors = cache if cache is not None else shared_vector_cache()
    # The consolidation pass must embed its summaries with the SAME cache the
    # arms read, or a merged record would be placed by a different embedder.
    kwargs.setdefault("vectors", vectors)
    vectors.ensure([e.text for c in convs for e in c.episodes])
    seed_everything(seed)

    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    absorbed: dict[str, dict[str, dict[str, int]]] = {}
    stats: dict[str, list[dict[str, Any]]] = {}
    unit_of: dict[str, dict[str, str]] = {"conversation": {}, "session": {}}
    answers: dict[str, set[str]] = {}
    for conversation in convs:
        by_scope = {scope: scoped_queries(conversation, scope) for scope in SCOPES}
        for query in conversation.queries:
            qid = str(query.query_id)
            answers[qid] = answer_tokens(query)
            unit_of["conversation"][qid] = conversation.sample_id
            unit_of["session"][qid] = str(query.cluster_id)
        store = store_for(conversation, vectors)
        try:
            for condition in conditions_for(store, thresholds=thresholds, seed=seed, **kwargs):
                block, taken = _score(condition, by_scope, answers, k=k, cache=vectors)
                stats.setdefault(condition.name, []).append(dict(condition.stats))
                for scope in SCOPES:
                    target = scores.setdefault(condition.name, {}).setdefault(
                        scope, {m: {} for m in METRICS})
                    for metric, values in block[scope].items():
                        target[metric].update(values)
                    absorbed.setdefault(condition.name, {}).setdefault(scope, {}).update(
                        taken[scope])
        finally:
            store.close()

    results: dict[str, Any] = {}
    for scope in SCOPES:
        for unit in ("conversation", "session"):
            grouped = {
                name: {m: _grouped(block[scope][m], unit_of[unit]) for m in METRICS}
                for name, block in scores.items()
            }
            base = grouped["before"]
            results.setdefault(scope, {})[unit] = {
                name: {m: _cells(grouped[name], base, metric=m, n_resamples=n_resamples)
                       for m in METRICS}
                for name in sorted(grouped)
            }

    primary = f"{PRIMARY['mode']}@{float(PRIMARY['threshold']):g}"
    diagnostics = _diagnostics(scores, absorbed, primary)
    cell = results[PRIMARY["scope"]][PRIMARY["resampling_unit"]][primary]
    verdict = {
        "cell": primary, **{k2: v for k2, v in PRIMARY.items()},
        "credited": cell[f"credited@{PRIMARY['depth']}"]["delta_vs_before"],
        "expected": cell[f"expected@{PRIMARY['depth']}"]["delta_vs_before"],
        "strict": cell[f"strict@{PRIMARY['depth']}"]["delta_vs_before"],
        "conventions_coincide_fraction":
            diagnostics[PRIMARY["scope"]]["conventions_coincide_fraction"],
        "outcome": classify_outcome(
            cell[f"credited@{PRIMARY['depth']}"]["delta_vs_before"]["mean"],
            cell[f"strict@{PRIMARY['depth']}"]["delta_vs_before"]["mean"],
            credited_ci=cell[f"credited@{PRIMARY['depth']}"]["delta_vs_before"],
            strict_ci=cell[f"strict@{PRIMARY['depth']}"]["delta_vs_before"],
            zero_absorbed_fraction=diagnostics[PRIMARY["scope"]]["conventions_coincide_fraction"],
        ),
    }
    return {
        "schema": SCHEMA,
        "pre_registration": PRE_REGISTRATION,
        "corpus": locomo_provenance(convs),
        "config": {
            "thresholds": sorted({float(t) for t in thresholds}), "seed": int(seed), "k": int(k),
            "arm": "plain_vector", "passes_per_condition": 1,
            "linkage": kwargs.get("linkage", "complete"),
            "bootstrap_resamples": int(n_resamples), "scopes": list(SCOPES),
            "resampling_units": ["conversation", "session"],
            "stores": "one per conversation",
        },
        "conditions": {name: {"stats": _merge_stats(items)} for name, items in sorted(stats.items())},
        "results": results,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "spend": _spend(stats),
        "wall_clock_seconds": round(time.time() - started, 2),
    }


def _spend(stats: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Live calls and money, summed over every consolidation pass in the run."""
    out = {"api_calls": 0, "cache_hits": 0, "live_prompt_tokens": 0,
           "live_output_tokens": 0, "cost_usd": 0.0, "temperature_enforced": True}
    for items in stats.values():
        for item in items:
            spend = item.get("spend") or {}
            for key in ("api_calls", "cache_hits", "live_prompt_tokens", "live_output_tokens"):
                out[key] = int(out[key]) + int(spend.get(key, 0))
            out["cost_usd"] = float(out["cost_usd"]) + float(spend.get("cost_usd", 0.0))
            if not spend.get("temperature_enforced", True):
                out["temperature_enforced"] = False
    out["cost_usd"] = round(float(out["cost_usd"]), 6)
    return out


def _diagnostics(scores: Mapping[str, Any], absorbed: Mapping[str, Any],
                 primary: str) -> dict[str, Any]:
    """What the conventions had to work with: the absorbed-count distribution."""
    out: dict[str, Any] = {}
    for scope in SCOPES:
        taken = absorbed.get(primary, {}).get(scope, {})
        histogram: dict[str, int] = {}
        for value in taken.values():
            key = "lost" if value < 0 else str(value)
            histogram[key] = histogram.get(key, 0) + 1
        block = scores.get(primary, {}).get(scope, {})
        coincide = sum(
            1 for qid in block.get("credited@5", {})
            if block["credited@5"][qid] == block["expected@5"][qid] == block["strict@5"][qid]
        )
        total = max(len(block.get("credited@5", {})), 1)
        out[scope] = {
            "absorbed_histogram_at_primary": {k: histogram[k] for k in sorted(histogram)},
            "n_scored": total,
            "conventions_coincide_fraction": round(coincide / total, 6),
            "fraction_absorbing_at_least_one": round(
                sum(v for k, v in histogram.items() if k not in ("0", "lost")) / total, 6),
        }
    return out
