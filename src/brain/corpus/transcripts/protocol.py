"""The labelling protocol, as text a human follows and as the effort it costs.

Kept in code rather than in a loose markdown file so that the decision rule and
the artifact that carries it cannot drift apart: the report is rendered from this
module, and changing the rule changes the report.

The rate constants are the honest part. They are an ESTIMATE from the shape of
the task (read two short records, decide two bits, occasionally open the diff),
not a measurement, and the report says so wherever it uses them.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["SECONDS_PER_PAIR", "DOUBLE_LABEL_FRACTION", "PROTOCOL", "effort_estimate"]

#: Seconds per pair, by regime. `short` is two one-line assertions; `long` is two
#: session notes and sometimes the underlying diff. Estimates, not measurements.
SECONDS_PER_PAIR: dict[str, int] = {"short": 25, "long": 60}

#: Fraction double-labelled to get an inter-annotator agreement figure. Without
#: one, a corpus of this kind has no evidence its labels mean anything.
DOUBLE_LABEL_FRACTION = 0.20

PROTOCOL = """\
# Labelling protocol: transcript-derived relational pairs

## What you are deciding

For each pair `(a, b)`, with `a` earlier in time, decide two INDEPENDENT bits and
let the action map produce the class. Never pick the class directly.

- `rho = 1` iff `a` and `b` speak about the SAME REFERENT: the same
  `(entity, attribute)`. Entity is the artifact the change lands on, attribute is
  the property of it being asserted. Same file is NOT sufficient and not
  necessary.
- `nu = 1` iff the VALUE asserted about that referent is the same. If `rho = 0`,
  set `nu` by asking whether the two assertions would be the same claim had they
  been about the same referent; when that question is meaningless, mark UNCLEAR.

| rho | nu | class | action |
|---|---|---|---|
| 1 | 1 | DUPLICATE | increment the observation count, keep one |
| 1 | 0 | SUPERSESSION | set `invalid_at` on the old, link forward, keep queryable |
| 0 | 1 | COINCIDENCE | independent, keep both, no link |
| 0 | 0 | INTERFERENCE | keep both, mark siblings, never merge |

## Rules that are not negotiable

1. **Do not look at the cosine.** It is recorded on every pair so the SELECTION is
   auditable. A label derived from it makes the corpus a restatement of the
   encoder, which is the quantity under test. Same for `stratum`: it is a
   sampling bucket, not a hint.
2. **Order is fixed.** `a` precedes `b`. SUPERSESSION means `b` replaces `a`,
   never the reverse. If the timestamps look wrong, mark UNCLEAR and move on.
3. **Judge the assertion, not the wording.** Two records saying the same thing in
   different words are `nu = 1`. Two records with near-identical wording asserting
   different values are `nu = 0`. This is the case the whole corpus exists to
   contain, so it is the case to be slowest about.
4. **UNCLEAR is a valid label and is not failure.** Use it rather than guessing.
   The rate of UNCLEAR is itself a reportable property of real transcripts.
5. **Label `short` and `long` renderings of the same event independently**, in
   separate passes, ideally on different days. They carry the same underlying
   relation by construction, so labelling them together leaks the answer.

## The ambiguous cases, called out in advance

- **Refinement versus replacement.** `b` rewords or expands `a` without changing
  what is claimed. That is DUPLICATE, not SUPERSESSION. Supersession requires the
  value to have CHANGED, not merely been restated better.
- **Partial supersession.** `b` revises one of several claims in `a`. Label on the
  claim named by the referent; if `a` carries several referents at once, mark
  UNCLEAR rather than choosing one.
- **Repeated events.** The same work recorded twice (a retry, an amend, a
  re-run). Same referent, same value: DUPLICATE. Do not use "it happened twice"
  as a reason to call it supersession.
- **Same file, different section.** Different attributes means `rho = 0`. Two
  edits to unrelated parts of one file that share heavy boilerplate are the
  canonical INTERFERENCE case, and calling them supersession because the file
  matches is the single most likely error in this task.
- **Boilerplate-dominated pairs.** In the `long` regime most of the text is shared
  context. Find the asserted value first, then decide. If you cannot find an
  asserted value, mark UNCLEAR.
- **Mechanical changes.** Formatting, renames, version bumps. Usually UNCLEAR:
  they assert nothing about a referent.

## Procedure

1. Two labellers work independently. `DOUBLE_LABEL_FRACTION` of pairs go to both.
2. Report Cohen's kappa on the doubly-labelled subset BEFORE reporting anything
   computed from the labels. A corpus that cannot show its own agreement rate
   supports no external-validity claim.
3. Adjudicate disagreements in a third pass. Record adjudicated pairs as such.
4. Freeze. The frozen split gets a content hash and never moves again.
"""


def effort_estimate(
    pairs: Sequence[Any], *, double_fraction: float = DOUBLE_LABEL_FRACTION
) -> dict[str, Any]:
    """Hours a human would spend, per regime and in total.

    Includes the double-labelled overhead, because a single-pass corpus has no
    agreement figure and therefore no defensible claim.
    """
    by_regime: dict[str, int] = {}
    for pair in pairs:
        by_regime[pair.regime] = by_regime.get(pair.regime, 0) + 1
    detail: dict[str, Any] = {}
    total_seconds = 0.0
    for regime, count in sorted(by_regime.items()):
        rate = SECONDS_PER_PAIR.get(regime, 45)
        seconds = count * rate * (1.0 + double_fraction)
        total_seconds += seconds
        detail[regime] = {
            "n_pairs": count,
            "seconds_per_pair": rate,
            "hours": round(seconds / 3600.0, 2),
        }
    return {
        "basis": "estimated from task shape, not measured on a pilot",
        "double_label_fraction": double_fraction,
        "per_regime": detail,
        "n_pairs_total": sum(by_regime.values()),
        "n_judgements_total": int(round(sum(by_regime.values()) * (1.0 + double_fraction))),
        "hours_total": round(total_seconds / 3600.0, 2),
        "days_at_5h": round(total_seconds / 3600.0 / 5.0, 2),
    }
