"""Run the memory-precision audit: validate the judge, then measure a pipeline.

    .venv/bin/python -m brain.eval.precision [--transcripts N] [--budget USD]

Order is fixed by the cost rules and by what a reader should be able to trust:

1. **Smoke at N = 10** on the control set, then print the measured per-call cost
   and the extrapolated cost of the whole pass. **Above the budget the run stops
   and reports the estimate instead of spending it.**
2. **Validate the judge** against the 36 hand-labelled control assertions and
   report agreement. Below `AGREEMENT_BAR` the metric is reported as
   UNCALIBRATED, because an unvalidated judge is not an instrument.
3. **Measure a real pipeline**: one bounded Haiku call per transcript stores
   assertions (the capture step the spec fixes in T2), and every stored
   assertion is judged against the transcript it came from. Transcripts are one
   episode per interference cluster of the frozen corpus, read-only, so the
   resampling units are independent of each other as well as of the assertions.

Everything is cached on disk, so a second identical run makes zero API calls and
reproduces the same numbers offline.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from brain.config import SMOKE_TEST, ensure_dir, results_dir
from brain.corpus.loader import load_corpus
from brain.eval.precision.control import CONTROL_SET, CONTROL_TRANSCRIPTS
from brain.baselines.llm import LlmUnavailable
from brain.eval.precision.accounting import extrapolate_cost
from brain.eval.precision.entailment import EntailmentJudge
from brain.eval.precision.metric import DEFAULT_SEEDS, precision_report
from brain.eval.precision.validation import AGREEMENT_BAR, validate_judge

#: rules.md: above this the task reports the estimate under Stop Conditions
#: rather than spending it. The orchestrator decides.
MAX_SPEND_USD = 25.0
_SMOKE_N = 10
#: Extraction asks for 4 to 8 assertions, so its output cap is about twice the
#: judge's. The projection scales the measured per-call cost accordingly rather
#: than assuming every call is a cheap one.
_EXTRACT_COST_FACTOR = 2.0
_ASSERTIONS_PER_TRANSCRIPT = 8


def _arg(name: str, default: int | float) -> Any:
    for token in sys.argv[1:]:
        if token.startswith(f"--{name}="):
            return type(default)(token.split("=", 1)[1])
    return default


def _corpus_transcripts(n: int) -> tuple[dict[str, str], str]:
    """One episode per interference cluster, ids sorted. The corpus is read-only."""
    corpus = load_corpus()  # verifies the frozen content hash and refuses on drift
    clusters = corpus.clusters()
    chosen = {}
    for cluster_id in sorted(clusters)[: max(int(n), 0)]:
        episode = clusters[cluster_id][0]
        chosen[episode.trace_id] = episode.text
    return chosen, corpus.content_hash


def main() -> int:
    n_transcripts = int(_arg("transcripts", 2 if SMOKE_TEST else 30))
    budget = float(_arg("budget", MAX_SPEND_USD))
    judge = EntailmentJudge()
    if judge.blocked_reason:
        print(f"BLOCKED: {judge.blocked_reason}")
        return 1

    # 1. Smoke first, then extrapolate the full pass before paying for it.
    smoke = validate_judge(judge=judge, control=CONTROL_SET[:_SMOKE_N], workers=6)
    print(f"[smoke] {smoke.n} control items: binary={smoke.binary_agreement:.3f} "
          f"strict={smoke.strict_agreement:.3f}")
    print(f"[smoke] judge stats: {json.dumps(judge.stats.to_dict(), sort_keys=True)}")
    judge_calls = (len(CONTROL_SET) - _SMOKE_N) + n_transcripts * _ASSERTIONS_PER_TRANSCRIPT
    projection = extrapolate_cost(judge.stats, judge_calls)
    extraction = extrapolate_cost(judge.stats, n_transcripts)
    projected = projection["projected_usd"] + extraction["projected_usd"] * _EXTRACT_COST_FACTOR
    total = judge.stats.cost_usd + projected
    print(f"[cost] spent so far {judge.stats.cost_usd:.4f} USD over {judge.stats.api_calls} live calls; "
          f"projected remaining {projected:.4f} USD over {judge_calls + n_transcripts} calls "
          f"({projection['source']} at {projection['per_call_usd']:.5f} USD per call); "
          f"full pass about {total:.4f} USD")
    if total > budget:
        print(f"STOP: projected full-pass cost {total:.2f} USD exceeds the {budget:.2f} USD budget. "
              "Nothing further was spent.")
        return 2

    # 2. The judge is an instrument only once it has been checked.
    validation = validate_judge(judge=judge, control=CONTROL_SET, workers=6)
    print(f"[judge] agreement over {validation.n} hand-labelled assertions: "
          f"binary={validation.binary_agreement:.3f} strict={validation.strict_agreement:.3f} "
          f"bar={AGREEMENT_BAR} calibrated={validation.calibrated}")
    for kind, (correct, count) in sorted(validation.by_kind.items()):
        print(f"         {kind:14s} {correct}/{count}")
    for (gold, got), count in sorted(validation.confusion.items()):
        if gold != got:  # where the two agreement rates come apart
            print(f"         confusion {gold} -> {got}: {count}")
    for item in validation.disagreements:
        print(f"         BINARY MISS {item.assertion_id} -> {item.verdict}: {item.assertion[:60]}")

    # 3. The pipeline under audit: capture, then judge what it stored.
    transcripts, content_hash = _corpus_transcripts(n_transcripts)
    stored = judge.extract_many(list(transcripts.values()), max_assertions=_ASSERTIONS_PER_TRANSCRIPT)
    items = {
        trace_id: (text, assertions)
        for (trace_id, text), assertions in zip(transcripts.items(), stored)
    }
    report = precision_report(items, judge=judge, seeds=DEFAULT_SEEDS)
    print(f"[metric] memory_precision = {report.score:.4f} over {report.n_assertions} assertions "
          f"from {report.n_transcripts} transcripts")
    print(f"[metric] cluster bootstrap over source transcripts, {len(report.seeds)} seeds: "
          f"lower {report.lower.mean:.4f} upper {report.upper.mean:.4f} "
          f"(seed 0: [{report.seed_intervals[0].lower:.4f}, {report.seed_intervals[0].upper:.4f}])")
    print(f"[metric] counts {json.dumps(report.counts, sort_keys=True)} "
          f"ambiguity_rate={report.ambiguity_rate:.4f}")
    print(f"[metric] invented_rate={report.invented_rate:.4f} of which strict fabrication "
          f"(NOT_ENTAILED only) = {report.fabrication_rate:.4f}")
    print(f"[cost] actual spend this run: {judge.stats.cost_usd:.4f} USD "
          f"({judge.stats.api_calls} live calls, {judge.stats.cache_hits} cache hits, "
          f"{judge.stats.prompt_tokens} prompt tokens, {judge.stats.output_tokens} output tokens)")
    print(f"[corpus] frozen content hash unchanged: {content_hash}")

    payload = {
        "validation": validation.to_dict(),
        "precision": report.to_dict(),
        "corpus_content_sha256": content_hash,
        "calibrated": validation.calibrated,
        "note": "" if validation.calibrated else "UNCALIBRATED: judge agreement below the bar",
    }
    path = ensure_dir(results_dir("precision")) / "memory_precision.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wrote] {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LlmUnavailable as exc:  # pragma: no cover - depends on the environment
        print(f"BLOCKED: {exc}")
        raise SystemExit(1) from exc
