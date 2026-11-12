"""The hand-labelled control set that validates the judge (36 assertions, 6 transcripts).

An unvalidated LLM judge is not an instrument, so this set exists before any
number is reported. It is written by hand rather than sampled from the frozen
corpus for two reasons: the corpus is immutable and its episodes share one
template, and a control set drawn from the same distribution the metric is run
on would measure the judge's fit to a template instead of its grasp of
entailment.

Composition, fixed before any judging (18 supported, 18 invented, as the task
contract requires):

| kind | verdict | n | what it tests |
|---|---|---|---|
| literal | ENTAILED | 8 | the easy case |
| paraphrase | ENTAILED | 4 | wording must not matter |
| inference | ENTAILED | 6 | a conclusion that *must* hold, incl. unit restatement |
| contradiction | NOT_ENTAILED | 5 | the transcript says otherwise |
| absent | NOT_ENTAILED | 4 | a different subject entirely |
| plausible | UNSUPPORTED_BUT_PLAUSIBLE | 9 | the interesting failure: a good guess |

The `inference` rows are the deliberate probe of this metric's main validity
threat. If the judge marks a conclusion the transcript forces as invented, then
"not entailed" is over-counting correct inference rather than fabrication, and
the headline number is wrong in a knowable direction.

The boundary between `absent` and `plausible` is genuinely soft: an unmentioned
detail can be both off-topic and believable. That is why agreement is reported
twice, strict over three labels and binary over supported-versus-invented, and
why the binary figure is the one the metric rests on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from brain.eval.precision.entailment import (
    ENTAILED,
    NOT_ENTAILED,
    UNSUPPORTED_BUT_PLAUSIBLE,
)

__all__ = ["ControlItem", "CONTROL_TRANSCRIPTS", "CONTROL_SET", "control_by_transcript"]


@dataclass(frozen=True, slots=True)
class ControlItem:
    """One hand-labelled (transcript, assertion) pair with its known verdict."""

    item_id: str
    transcript_id: str
    assertion: str
    verdict: str
    kind: str

    @property
    def supported(self) -> bool:
        """Gold binary label: True when the transcript establishes the assertion."""
        return self.verdict == ENTAILED


CONTROL_TRANSCRIPTS: dict[str, str] = {
    "k1": (
        "[2026-02-03] platform/featureflags/eval.go. Spent the morning on the flapping flag "
        "in the flag evaluator. Repro: evaluate the same flag for one user repeatedly and the "
        "decision alternates between calls. It only turned up for users sitting in a negative "
        "UTC offset. Root cause: a cache entry that was never invalidated on write. Fix: "
        "invalidated the entry inside the write transaction, shipped as PR #9166. After the "
        "deploy the error rate dropped from 1.9 percent to under 6 per ten thousand. Stack is "
        "Go and the covering check is eval_test.go."
    ),
    "k2": (
        "[2026-03-11] payments/webhooks/dispatch.py. Duplicate charge webhooks were reaching "
        "merchants: the same event id was delivered two or three times within a minute. It "
        "only happened for payloads above roughly two megabytes. Cause was an unbounded retry "
        "loop with no backoff, so a slow acknowledgement looked like a failure. Capped attempts "
        "at five and added exponential backoff with jitter, on branch fix/dup-webhooks-2. "
        "Duplicate deliveries went to zero over the next 24 hours. Python, and dispatch_test.py "
        "covers it."
    ),
    "k3": (
        "[2026-04-19] search/indexer/segment.rs. The indexer kept getting OOM-killed after "
        "about nine hours. Heap grew steadily even with a fixed corpus. Each segment merge "
        "cloned the term dictionary and the old copy stayed reachable through a stats handle. "
        "Dropped the handle before the merge and reused the buffer instead of cloning. Resident "
        "set is now flat at 2.1 GB across a 40 hour soak. Rust; segment_bench.rs is the "
        "covering benchmark."
    ),
    "k4": (
        "[2026-05-02] mobile/ios/Startup.swift. Crash on cold start, but only on devices that "
        "had been offline since the previous release. The startup path read a cached feature "
        "manifest and force-unwrapped a field that older manifests do not carry. Made the field "
        "optional with a default and added a migration for stored manifests, PR #4102. "
        "Crash-free sessions went from 98.4 percent to 99.97 percent in two days. Swift; "
        "StartupTests.swift covers the migration."
    ),
    "k5": (
        "[2026-05-28] etl/warehouse/load_orders.sql. The nightly order load started dropping "
        "about 4 percent of rows without failing. The upstream export had added a nullable "
        "currency column and our COPY was positional, so every row after the new column shifted "
        "by one and failed the type check on quantity. Switched the COPY to name the columns "
        "explicitly and backfilled the three affected nights, ticket DW-812. Row counts match "
        "the source again. SQL and dbt; the freshness test lives in schema.yml."
    ),
    "k6": (
        "[2026-06-14] auth/session/refresh.ts. Users were being signed out mid-session, roughly "
        "one in two hundred requests. Two concurrent requests could both see an expiring token "
        "and both call refresh; the second refresh invalidated the first one's new token. Added "
        "a per-session mutex around refresh and a five second grace window on the old token, PR "
        "#7731. Sign-out reports stopped entirely in the following week. TypeScript; "
        "refresh.spec.ts covers the concurrent case."
    ),
}

#: (transcript_id, assertion, verdict, kind), in a fixed order so the report is stable.
_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("k1", "The flapping decisions came from a cache entry that was never invalidated on write.", ENTAILED, "literal"),
    ("k1", "The fix shipped as PR #9166.", ENTAILED, "literal"),
    ("k1", "After the fix the error rate was below one tenth of one percent.", ENTAILED, "inference"),
    ("k1", "The flag evaluator was rewritten in Rust.", NOT_ENTAILED, "contradiction"),
    ("k1", "The same session also migrated the flag store to Postgres.", NOT_ENTAILED, "absent"),
    ("k1", "The stale entry survived because two replicas disagreed about the write.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),

    ("k2", "Duplicate webhook deliveries were caused by a retry loop with no backoff.", ENTAILED, "literal"),
    ("k2", "Retries are now capped at five attempts.", ENTAILED, "literal"),
    ("k2", "The duplicates were only seen on large payloads.", ENTAILED, "paraphrase"),
    ("k2", "Merchants were refunded for the duplicate charges.", NOT_ENTAILED, "absent"),
    ("k2", "The slow acknowledgements came from a lock held by the downstream database.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),
    ("k2", "The backoff jitter is drawn from a uniform distribution.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),

    ("k3", "The indexer ran out of memory because an old term dictionary stayed reachable after each merge.", ENTAILED, "literal"),
    ("k3", "Memory use stayed flat over a soak longer than a day.", ENTAILED, "inference"),
    ("k3", "The fix stopped the merge from cloning the term dictionary.", ENTAILED, "paraphrase"),
    ("k3", "The out-of-memory kills were caused by a leak in the memory allocator.", NOT_ENTAILED, "contradiction"),
    ("k3", "The service was restarted nightly as a workaround before the fix landed.", NOT_ENTAILED, "absent"),
    ("k3", "The stats handle had been added by an earlier observability change.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),

    ("k4", "The cold start crash came from force-unwrapping a field that older cached manifests do not carry.", ENTAILED, "literal"),
    ("k4", "The share of crash-free sessions improved after the change.", ENTAILED, "inference"),
    ("k4", "Manifests already stored on the device were migrated.", ENTAILED, "paraphrase"),
    ("k4", "The session was spent upgrading the CI runners.", NOT_ENTAILED, "absent"),
    ("k4", "The force-unwrap had been in the startup path for several releases.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),
    ("k4", "The default value chosen for the missing field is false.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),

    ("k5", "Rows were dropped because a positional COPY shifted columns after an upstream schema change.", ENTAILED, "literal"),
    ("k5", "The load now names its columns explicitly.", ENTAILED, "literal"),
    ("k5", "Three nights of data were backfilled.", ENTAILED, "literal"),
    ("k5", "The upstream export removed a column.", NOT_ENTAILED, "contradiction"),
    ("k5", "The nightly load failed loudly while it was dropping rows.", NOT_ENTAILED, "contradiction"),
    ("k5", "The dropped rows were concentrated in international orders.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),

    ("k6", "Two concurrent refreshes could invalidate each other's tokens.", ENTAILED, "literal"),
    ("k6", "A per-session mutex now guards the refresh path.", ENTAILED, "paraphrase"),
    ("k6", "Fewer than one percent of requests were affected.", ENTAILED, "inference"),
    ("k6", "The old token keeps a sixty second grace window.", NOT_ENTAILED, "contradiction"),
    ("k6", "The race was more likely on mobile clients that retry aggressively.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),
    ("k6", "The mutex is held across the network call to the identity provider.", UNSUPPORTED_BUT_PLAUSIBLE, "plausible"),
)

CONTROL_SET: tuple[ControlItem, ...] = tuple(
    ControlItem(
        item_id=f"{transcript_id}#{index:02d}",
        transcript_id=transcript_id,
        assertion=assertion,
        verdict=verdict,
        kind=kind,
    )
    for index, (transcript_id, assertion, verdict, kind) in enumerate(_ROWS)
)


def control_by_transcript() -> Mapping[str, tuple[ControlItem, ...]]:
    """Control items grouped by transcript, ids sorted (the bootstrap's unit)."""
    grouped: dict[str, list[ControlItem]] = {}
    for item in CONTROL_SET:
        grouped.setdefault(item.transcript_id, []).append(item)
    return {key: tuple(grouped[key]) for key in sorted(grouped)}
