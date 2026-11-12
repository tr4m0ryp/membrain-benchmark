"""Memory precision (I5a): the fraction of stored assertions the transcript never said.

The first of the two missing measurements in the proposal. Nothing published
measures it: every agent-memory benchmark asks whether a system recalls or
applies what it learned, never whether what it learned was fabricated.

    from brain.eval.precision import EntailmentJudge, memory_precision, validate_judge

    judge = EntailmentJudge()                 # cached on disk, claude-haiku-4-5
    print(validate_judge(judge=judge))        # do this before believing a number
    score, verdicts = memory_precision(assertions, transcript, judge=judge)

Three layers:

* `entailment` + `accounting` -- the three-way judge, its prompts, its response
  cache, and the token/cost accounting that sums all three prompt-token buckets.
* `control` + `validation` -- 36 hand-labelled assertions over 6 transcripts,
  half of them deliberate inventions, and the agreement report they produce.
* `metric` -- the metric itself plus a transcript-clustered BCa interval. The
  resampling unit is the source transcript, never the assertion.

`python -m brain.eval.precision` runs the whole audit: smoke, cost projection,
judge validation, then the measurement.
"""

from brain.eval.precision.accounting import (
    PRICE_USD_PER_MTOK,
    JudgeStats,
    extrapolate_cost,
)
from brain.eval.precision.control import (
    CONTROL_SET,
    CONTROL_TRANSCRIPTS,
    ControlItem,
    control_by_transcript,
)
from brain.eval.precision.entailment import (
    DEFAULT_CACHE_DIR,
    ENTAILED,
    NOT_ENTAILED,
    PROMPT_VERSION,
    UNPARSED,
    UNSUPPORTED_BUT_PLAUSIBLE,
    VERDICTS,
    EntailmentJudge,
)
from brain.eval.precision.metric import (
    DEFAULT_SEEDS,
    AssertionVerdict,
    PrecisionReport,
    memory_precision,
    precision_report,
)
from brain.eval.precision.validation import AGREEMENT_BAR, ValidationReport, validate_judge

__all__ = [
    "ENTAILED",
    "NOT_ENTAILED",
    "UNSUPPORTED_BUT_PLAUSIBLE",
    "UNPARSED",
    "VERDICTS",
    "PROMPT_VERSION",
    "DEFAULT_CACHE_DIR",
    "PRICE_USD_PER_MTOK",
    "JudgeStats",
    "EntailmentJudge",
    "extrapolate_cost",
    "ControlItem",
    "CONTROL_SET",
    "CONTROL_TRANSCRIPTS",
    "control_by_transcript",
    "AssertionVerdict",
    "PrecisionReport",
    "ValidationReport",
    "AGREEMENT_BAR",
    "DEFAULT_SEEDS",
    "memory_precision",
    "precision_report",
    "validate_judge",
]
