"""Validating the judge against the hand-labelled control set.

An unvalidated LLM judge is not an instrument, so this runs before any number
from `metric` is believed, and its result is carried next to that number rather
than assumed.

Two agreement rates are reported, and the difference between them is
informative rather than cosmetic:

* **strict** over all three labels. The harder number, and the one that says
  whether the judge shares our reading of "plausible but unestablished".
* **binary**, supported versus invented. This is what the metric is actually
  defined on, since it folds both invention labels together, so this is the
  figure held to `AGREEMENT_BAR`.

A judge that reliably separates supported from invented while shuffling the two
invention labels is a usable instrument for this metric and an unreliable one
for a claim about *which kind* of invention occurred. Reporting both keeps that
distinction visible instead of averaging it away.

Disagreements are returned in full, not just counted: with 36 control items the
individual misses are readable, and they are the only evidence about which way
the judge is biased.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from brain.eval.precision.control import CONTROL_SET, CONTROL_TRANSCRIPTS, ControlItem
from brain.eval.precision.metric import AssertionVerdict, judge_pairs

__all__ = ["AGREEMENT_BAR", "ValidationReport", "validate_judge"]

#: Below this the judge is reported as uncalibrated and the metric is not claimed
#: to be calibrated either (task contract, Stop Conditions).
AGREEMENT_BAR = 0.85


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The judge measured against the hand-labelled control set."""

    n: int
    strict_agreement: float
    binary_agreement: float
    by_kind: dict[str, tuple[int, int]]
    confusion: dict[tuple[str, str], int]
    disagreements: tuple[AssertionVerdict, ...]

    @property
    def calibrated(self) -> bool:
        """Binary agreement carries the bar: it is what the metric rests on."""
        return self.binary_agreement >= AGREEMENT_BAR

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "strict_agreement": round(self.strict_agreement, 4),
            "binary_agreement": round(self.binary_agreement, 4),
            "bar": AGREEMENT_BAR,
            "calibrated": self.calibrated,
            "by_kind": {k: {"correct": c, "n": n} for k, (c, n) in sorted(self.by_kind.items())},
            "confusion": {f"{gold}->{got}": n for (gold, got), n in sorted(self.confusion.items())},
            "disagreements": [v.to_dict() for v in self.disagreements],
        }


def validate_judge(
    *,
    judge: Any | None = None,
    control: Sequence[ControlItem] = CONTROL_SET,
    transcripts: Mapping[str, str] = CONTROL_TRANSCRIPTS,
    workers: int = 6,
) -> ValidationReport:
    """Agreement with the control set, strict over three labels and binary.

    `by_kind` is the diagnostic that matters most: the `inference` rows are
    assertions the transcript forces, so a judge that marks them invented is
    over-counting correct inference rather than fabrication, which is this
    metric's main validity threat.
    """
    if judge is None:
        from brain.eval.precision.entailment import EntailmentJudge

        judge = EntailmentJudge()
    items = list(control)
    results = judge_pairs(
        judge, [(transcripts[item.transcript_id], item.assertion) for item in items], workers
    )
    strict = binary = 0
    by_kind: dict[str, list[int]] = {}
    confusion: dict[tuple[str, str], int] = {}
    disagreements: list[AssertionVerdict] = []
    for item, (verdict, reason) in zip(items, results):
        got = AssertionVerdict(item.item_id, item.transcript_id, item.assertion, verdict, reason)
        confusion[(item.verdict, verdict)] = confusion.get((item.verdict, verdict), 0) + 1
        tally = by_kind.setdefault(item.kind, [0, 0])
        tally[1] += 1
        if verdict == item.verdict:
            strict += 1
            tally[0] += 1
        if got.judged and got.supported == item.supported:
            binary += 1
        else:
            disagreements.append(got)
    n = len(items)
    return ValidationReport(
        n=n,
        strict_agreement=(strict / n) if n else math.nan,
        binary_agreement=(binary / n) if n else math.nan,
        by_kind={kind: (correct, count) for kind, (correct, count) in by_kind.items()},
        confusion=confusion,
        disagreements=tuple(disagreements),
    )
