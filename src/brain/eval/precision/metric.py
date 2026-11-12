"""Memory precision (I5a): the rate at which a pipeline INVENTS what it stored.

```
memory_precision = 1 - |{ a in A : transcript does not entail a }| / |A|
```

`A` is the assertion set a pipeline stored from one transcript. No published
benchmark measures this: LoCoMo, LongMemEval and ImplicitMemBench all ask
whether a system recalls what it learned, never whether it was ever said.

Three choices fix the number and are all visible here:

1. **Two labels count as invented**, `NOT_ENTAILED` and
   `UNSUPPORTED_BUT_PLAUSIBLE`, because a plausible invention is still one. All
   three counts are reported, so the stricter reading stays recoverable.
2. **The empty assertion set is NaN, never 1.0.** `1 - 0/0` is undefined, and the
   convenient reading would hand a perfect score to a pipeline that stores
   nothing, which is the degenerate way to game a precision metric.
3. **The resampling unit is the source transcript.** Assertions from one
   transcript are not independent draws: they share its wording, its facts and
   whatever the extractor misread. Resampling assertions would report an
   interval far too narrow, the failure mode that manufactured a false win on
   this project once already (`brain.eval.bootstrap`).

An unparseable verdict is `UNPARSED`, left out of the denominator rather than
coerced into a label, and its rate is reported: a failing judge must never read
as an inventing pipeline.

**Is "not entailed" the right reading of "invented"?** The metric's main
validity threat, so it is measured rather than asserted, and it turns on two
kinds of unstated assertion. A **deductive consequence** ("under 6 per ten
thousand" stored as "below one tenth of one percent") is not an invention, and
counting it as one would make this a paraphrase detector; the control set
carries six such rows so the judge's treatment of them is a reported number. A
**pragmatic inference** the transcript merely suggests ("the slow
acknowledgement came from a database lock") is counted as invented, because a
store cannot afterwards tell such a belief apart from something that was
actually said: it is retrieved with the same standing as the transcript's own
facts. That second call is a choice, so `PrecisionReport.fabrication_rate`
reports the stricter reading from the same verdicts and both numbers travel
together.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from brain.config import get_rng
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci
from brain.eval.precision.entailment import (
    ENTAILED,
    NOT_ENTAILED,
    UNPARSED,
    UNSUPPORTED_BUT_PLAUSIBLE,
    VERDICTS,
    EntailmentJudge,
)
from brain.eval.stats import Interval, mean_ci

__all__ = [
    "AssertionVerdict",
    "PrecisionReport",
    "DEFAULT_SEEDS",
    "judge_pairs",
    "memory_precision",
    "precision_report",
]

#: "A single run is not a result" (rules.md). Applies to the interval, since the
#: point estimate is a deterministic function of the cached verdicts.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class AssertionVerdict:
    """One stored assertion, judged against the transcript it came from."""

    assertion_id: str
    transcript_id: str
    assertion: str
    verdict: str
    reason: str = ""

    @property
    def judged(self) -> bool:
        return self.verdict in VERDICTS

    @property
    def supported(self) -> bool:
        return self.verdict == ENTAILED

    @property
    def invented(self) -> bool:
        """The headline failure: not entailed, whether or not it sounds right."""
        return self.verdict in (NOT_ENTAILED, UNSUPPORTED_BUT_PLAUSIBLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "transcript_id": self.transcript_id,
            "assertion": self.assertion,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def judge_pairs(judge: Any, pairs: Sequence[tuple[str, str]], workers: int) -> list[tuple[str, str]]:
    """Judge every pair, preferring a batched judge and falling back to one call each.

    The worker count is passed by signature inspection, not by catching
    TypeError: a TypeError raised *inside* a batch would otherwise be swallowed
    and the whole batch silently re-run, paying for every call twice.
    """
    if not pairs:
        return []
    batched = getattr(judge, "judge_many", None)
    if callable(batched):
        takes_workers = "workers" in inspect.signature(batched).parameters
        results = list(batched(pairs, workers=workers) if takes_workers else batched(pairs))
    else:
        results = [judge.judge(transcript, assertion) for transcript, assertion in pairs]
    if len(results) != len(pairs):
        raise ValueError(f"judge returned {len(results)} verdicts for {len(pairs)} pairs")
    return results


def memory_precision(
    assertions: Sequence[str],
    transcript: str,
    *,
    judge: Any | None = None,
    transcript_id: str = "t0",
    workers: int = 6,
) -> tuple[float, list[AssertionVerdict]]:
    """`(score, per_item_verdicts)` for one transcript's stored assertions.

    NaN when nothing was judged (an empty set, or a judge that failed on every
    item): a precision with no observations behind it is not a score.
    """
    items = [str(a) for a in assertions]
    if judge is None:
        judge = EntailmentJudge()
    results = judge_pairs(judge, [(transcript, a) for a in items], workers)
    verdicts = [
        AssertionVerdict(
            assertion_id=f"{transcript_id}#{index:02d}",
            transcript_id=str(transcript_id),
            assertion=assertion,
            verdict=verdict,
            reason=reason,
        )
        for index, (assertion, (verdict, reason)) in enumerate(zip(items, results))
    ]
    judged = [v for v in verdicts if v.judged]
    if not judged:
        return math.nan, verdicts
    return 1.0 - sum(v.invented for v in judged) / len(judged), verdicts


def _counts(verdicts: Iterable[AssertionVerdict]) -> dict[str, int]:
    tally = {label: 0 for label in (*VERDICTS, UNPARSED)}
    for verdict in verdicts:
        tally[verdict.verdict if verdict.judged else UNPARSED] += 1
    return tally


@dataclass(frozen=True, slots=True)
class PrecisionReport:
    """Memory precision over many transcripts, with a transcript-clustered interval."""

    score: float
    counts: dict[str, int]
    per_item: tuple[AssertionVerdict, ...]
    groups: dict[str, list[float]]
    seeds: tuple[int, ...]
    seed_intervals: tuple[Interval, ...]
    lower: Interval
    upper: Interval
    judge_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_assertions(self) -> int:
        return sum(len(values) for values in self.groups.values())

    @property
    def n_transcripts(self) -> int:
        return len(self.groups)

    @property
    def invented_rate(self) -> float:
        return math.nan if math.isnan(self.score) else 1.0 - self.score

    @property
    def fabrication_rate(self) -> float:
        """The stricter reading of invention: `NOT_ENTAILED` only, no plausible guesses.

        Sits next to `invented_rate` because which of the two is "invention" is a
        definitional choice (see the module docstring), and both come from the
        same verdicts, so neither costs an extra call.
        """
        judged = sum(self.counts.get(label, 0) for label in VERDICTS)
        return (self.counts.get(NOT_ENTAILED, 0) / judged) if judged else math.nan

    @property
    def ambiguity_rate(self) -> float:
        """Fraction of assertions the judge could not label at all."""
        total = sum(self.counts.values())
        return (self.counts.get(UNPARSED, 0) / total) if total else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_precision": self.score,
            "invented_rate": self.invented_rate,
            "fabrication_rate": self.fabrication_rate,
            "counts": dict(self.counts),
            "n_assertions": self.n_assertions,
            "n_transcripts": self.n_transcripts,
            "ambiguity_rate": self.ambiguity_rate,
            "seeds": list(self.seeds),
            "interval_lower_over_seeds": self.lower.to_dict(),
            "interval_upper_over_seeds": self.upper.to_dict(),
            "seed_intervals": [i.to_dict() for i in self.seed_intervals],
            "resampling_unit": "source transcript",
            "judge_stats": dict(self.judge_stats),
            "per_item": [v.to_dict() for v in self.per_item],
        }


def precision_report(
    items: Mapping[str, tuple[str, Sequence[str]]],
    *,
    judge: Any | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    workers: int = 6,
) -> PrecisionReport:
    """Score every transcript's assertions and bootstrap over TRANSCRIPTS.

    `items` maps a transcript id to `(transcript_text, stored_assertions)`. The
    point estimate is a deterministic function of the verdicts, so seeds move
    only the resampling draw; the spread of the endpoints across seeds is what
    the >= 5 seed rule buys here, and it is reported rather than hidden.
    """
    if judge is None:
        judge = EntailmentJudge()
    pairs: list[tuple[str, str]] = []
    index: list[tuple[str, int, str]] = []
    for transcript_id in sorted(items):  # sorted: ids drive every downstream order
        transcript, assertions = items[transcript_id]
        for position, assertion in enumerate(assertions):
            pairs.append((transcript, str(assertion)))
            index.append((transcript_id, position, str(assertion)))
    results = judge_pairs(judge, pairs, workers)

    per_item: list[AssertionVerdict] = []
    groups: dict[str, list[float]] = {}
    for (transcript_id, position, assertion), (verdict, reason) in zip(index, results):
        item = AssertionVerdict(
            assertion_id=f"{transcript_id}#{position:02d}",
            transcript_id=transcript_id,
            assertion=assertion,
            verdict=verdict,
            reason=reason,
        )
        per_item.append(item)
        if item.judged:
            groups.setdefault(transcript_id, []).append(float(item.supported))
    groups = {key: groups[key] for key in sorted(groups)}

    stats = getattr(judge, "stats", None)
    wanted = tuple(sorted({int(seed) for seed in seeds}))
    intervals = tuple(
        cluster_bootstrap_ci(
            groups,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=get_rng(seed, stream="eval.precision.bootstrap"),
        )
        for seed in wanted
    ) if groups else ()
    score = intervals[0].mean if intervals else math.nan
    return PrecisionReport(
        score=score,
        counts=_counts(per_item),
        per_item=tuple(per_item),
        groups=groups,
        seeds=wanted,
        seed_intervals=intervals,
        lower=mean_ci([i.lower for i in intervals], alpha) if intervals else Interval(math.nan, math.nan, math.nan),
        upper=mean_ci([i.upper for i in intervals], alpha) if intervals else Interval(math.nan, math.nan, math.nan),
        judge_stats=stats.to_dict() if hasattr(stats, "to_dict") else {},
    )


# The tiny-N path for this package is `python -m brain.eval.precision`: it smokes
# at N = 10 and gates on cost, and `BRAIN_SMOKE_TEST=1` shrinks it to 2 transcripts.
