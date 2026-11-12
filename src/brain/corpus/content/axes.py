"""Discriminative axes: the ONLY thing that separates siblings in a cluster.

Each axis value carries two phrasings:

- `episode`: how the fact is written into the stored episode.
- `cue`: how the query refers to the same fact, deliberately PARAPHRASED with
  near-zero lexical overlap.

The paraphrase is load-bearing. If the query quoted the episode verbatim, BM25
would win by string match and the corpus would test lexical overlap rather than
interference between similar-but-distinct memories, which is the hypothesis
under test (spec F2, M-A). Numbers stay OUT of the cue for the same reason: an
outcome cue names the shape of the result, never its digits.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AxisValue", "Diagnosis", "TRIGGERS", "DIAGNOSES", "OUTCOMES"]


@dataclass(frozen=True, slots=True)
class AxisValue:
    value_id: str
    episode: str
    cue: str


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Root cause and the fix that addressed it, kept as one coherent pair."""

    value_id: str
    cause: str
    cause_cue: str
    fix: str


_TRIGGERS: tuple[tuple[str, str], ...] = (
    ("only when two requests raced on the same key",
     "just in the case where a pair of calls landed on an identical record at once"),
    ("only on the first request after a cold start",
     "exclusively on the very first call following a fresh boot"),
    ("only for sessions that arrived through the OAuth provider",
     "purely for logins issued by the third-party identity flow"),
    ("only for payloads above roughly two megabytes",
     "solely once the body grew past a couple of million bytes"),
    ("only for users sitting in a negative UTC offset",
     "just for accounts located west of Greenwich"),
    ("only after the client had already retried once",
     "exclusively once the caller had made a second attempt"),
    ("only on the staging read replica",
     "purely on the pre-production follower database"),
    ("only when the input list came in empty",
     "solely if the incoming collection held no entries"),
    ("only for records carrying non-ASCII names",
     "just for rows whose names contained accented characters"),
    ("only for accounts in the five percent rollout bucket",
     "exclusively for tenants inside the small canary slice"),
    ("only for jobs scheduled across the midnight boundary",
     "purely for runs that straddled the change of day"),
    ("only for traffic arriving through the corporate proxy",
     "solely on requests relayed by the office egress node"),
    ("only after a rolling restart of the fleet",
     "just following a staged reboot of every node"),
    ("only for clients still pinned to the v1 SDK",
     "exclusively for callers held back on the older library major"),
    ("only for tenants with more than ten thousand distinct keys",
     "purely for customers holding a very large set of unique identifiers"),
    ("only for deeply nested inputs beyond six levels",
     "solely when the structure recursed past half a dozen layers"),
)

_DIAGNOSES: tuple[tuple[str, str, str], ...] = (
    ("a missing composite index on (tenant_id, created_at)",
     "an absent multi-column lookup structure on the table",
     "added the composite index and backfilled in batches"),
    ("connection pool exhaustion under burst load",
     "the pool of open connections running dry during a burst",
     "raised the pool ceiling and put a bounded wait queue in front of it"),
    ("a cache entry that was never invalidated on write",
     "a cached value that survived a write it should not have",
     "invalidated the entry inside the write transaction"),
    ("clock skew between the two hosts",
     "the two machines disagreeing about the current time",
     "switched the deadline to a monotonic clock"),
    ("a mutable default argument shared across calls",
     "a shared mutable value hiding in a parameter default",
     "made the default empty and built the value per call"),
    ("an off-by-one in the slice bound",
     "a boundary that sat one position wrong in a slice",
     "corrected the bound and added a property test around it"),
    ("an unbounded retry loop with no backoff",
     "retries that repeated forever without ever slowing down",
     "capped the attempts and added exponential backoff with jitter"),
    ("floating point rounding in the money path",
     "decimal drift in a code path that handles money",
     "moved the arithmetic to integer minor units"),
    ("a race in lazy initialisation",
     "two threads building the same deferred value at once",
     "guarded the init with a double-checked lock"),
    ("the config loader reading the wrong environment file",
     "settings being picked up from the wrong environment file",
     "made environment selection explicit and failed closed on ambiguity"),
    ("missing Unicode normalisation before hashing",
     "text that was digested before it was normalised",
     "normalised to NFC ahead of the digest"),
    ("an error swallowed by a bare except",
     "a failure quietly absorbed by an overly broad handler",
     "narrowed the handler and let the failure propagate"),
    ("an N+1 query hidden behind a lazy relation",
     "one query per row issued behind a deferred relation",
     "prefetched the relation in a single query"),
    ("a listener registered but never removed",
     "a subscriber that was attached and never detached",
     "removed the listener on teardown and asserted it in a test"),
    ("schema drift between the migration and the model",
     "the migration and the model no longer agreeing on shape",
     "regenerated the model from the migration and pinned the checksum"),
    ("a naive datetime compared against an aware one",
     "a timestamp without a zone compared to one that had a zone",
     "made every boundary timezone aware at the edge"),
    ("two writes that were never wrapped in a transaction",
     "a pair of writes that were not committed atomically",
     "wrapped both writes in one transaction"),
    ("the serialiser dropping fields it did not know",
     "the encoder discarding fields outside its known set",
     "switched to a strict schema that rejects unknown fields"),
)

#: Slots: a = 2..99, b = 0..9 (decimal), c = 10..990, d = 1..9, e = 1..9, m = 3..90.
#: Numbers live in the episode only. Keeping them out of the cue is what stops a
#: query from being solvable by matching a rare literal token.
_OUTCOMES: tuple[tuple[str, str], ...] = (
    ("p99 latency fell from {d}.{b} s to {c} ms",
     "the tail latency collapsing to a fraction of a second"),
    ("the error rate dropped from {d}.{b} percent to under {e} per ten thousand",
     "the failure share falling to almost nothing"),
    ("the job finished in {m} minutes instead of {d} hours",
     "the batch completing in minutes rather than hours"),
    ("memory settled at {c} MB instead of climbing past {d} GB",
     "the resident footprint levelling off instead of growing"),
    ("{a} duplicate rows a day stopped being written",
     "repeated rows no longer being appended each day"),
    ("the flaky test went from {a} failures per hundred runs to zero",
     "an unreliable test becoming perfectly stable"),
    ("the backlog cleared {m} minutes after the deploy",
     "the pending queue draining shortly after release"),
    ("throughput rose {a} percent, to {c} requests per second",
     "the served volume per second going up sharply"),
    ("the nightly rollup matched the source counts for {a} days running",
     "the aggregate agreeing with the raw counts day after day"),
    ("cold start dropped {a} percent, to {c} ms",
     "the first-call warmup shrinking substantially"),
    ("the alert stayed quiet for {a} consecutive days",
     "the pager going silent for a run of days"),
    ("we reclaimed {a} GB of blob storage",
     "a sizeable chunk of object storage being freed"),
)

TRIGGERS: tuple[AxisValue, ...] = tuple(
    AxisValue(f"g{index:02d}", episode, cue)
    for index, (episode, cue) in enumerate(_TRIGGERS)
)

DIAGNOSES: tuple[Diagnosis, ...] = tuple(
    Diagnosis(f"d{index:02d}", cause, cause_cue, fix)
    for index, (cause, cause_cue, fix) in enumerate(_DIAGNOSES)
)

OUTCOMES: tuple[AxisValue, ...] = tuple(
    AxisValue(f"o{index:02d}", episode, cue)
    for index, (episode, cue) in enumerate(_OUTCOMES)
)
