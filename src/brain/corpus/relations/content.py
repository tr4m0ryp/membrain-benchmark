"""Content banks and the renderer for the relational corpus.

Split out of `generate.py` under the project's 300 line cap. Nothing here
knows about pair classes:
the renderer takes a topic, a referent, a value, a date and a variant, and the
four classes are read off the plan afterwards. That separation is what keeps the
instrument honest, so it is structural rather than a convention.

Three banks matter, in this order:

- `VALUES` is keyed by TYPE, not by attribute, so two distinct referents of the
  same type can carry a byte-identical value. That is what makes the
  COINCIDENCE cell (rho = 0, nu = 1) occur naturally instead of being staged.
- `_INCIDENTAL` gives every long episode its own chatter. Without it the long
  regime would collapse to a one-sentence diff and the measured similarity would
  be an artifact of the template rather than of the content.
- `_SHORT_ASSERTIONS` renders the short regime as the assertion sentence ALONE.
  That is the limit where the slot phrase dominates, and it is deliberately the
  instantiation that separates the classes most easily, which is the direction
  that hurts our own claim rather than helping it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Entity",
    "Attribute",
    "Workstream",
    "VALUES",
    "WORKSTREAMS",
    "ENTITIES",
    "ATTRIBUTES",
    "TOPICS_BY_AREA",
    "VALID_SLOTS",
    "N_VARIANTS",
    "N_INCIDENTAL",
    "render_text",
]


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    noun: str
    kind: str
    area: str


@dataclass(frozen=True, slots=True)
class Attribute:
    attribute_id: str
    phrase: str
    value_type: str
    kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Workstream:
    topic_id: str
    area: str
    path: str
    focus: str
    activity: str
    check: str


_WORKSTREAMS: tuple[tuple[str, ...], ...] = (
    ("authentication", "services/auth/session_store.py", "the session store rewrite", "moving sessions off the legacy table", "tests/auth/test_session_store.py"),
    ("authentication", "services/auth/token_refresh.py", "the refresh path cleanup", "shortening the refresh window", "tests/auth/test_token_refresh.py"),
    ("authentication", "services/auth/login_flow.py", "the login flow audit", "walking every branch of the login handler", "tests/auth/test_login_flow.py"),
    ("api gateway", "api/gateway/rate_limit.go", "the limiter rework", "replacing the fixed window with a sliding one", "api/gateway/rate_limit_test.go"),
    ("api gateway", "api/gateway/router.go", "the router cleanup", "collapsing two overlapping prefixes", "api/gateway/router_test.go"),
    ("api gateway", "api/gateway/timeouts.go", "the timeout sweep", "auditing every outbound deadline", "api/gateway/timeouts_test.go"),
    ("billing", "billing/invoice_totals.py", "the totals rounding fix", "moving money arithmetic to integer units", "tests/billing/test_invoice_totals.py"),
    ("billing", "billing/webhooks/stripe.py", "the webhook idempotency work", "keying the ledger on the event id", "tests/billing/test_stripe_webhook.py"),
    ("billing", "billing/reports/monthly.py", "the monthly report rebuild", "regenerating the statement templates", "tests/billing/test_monthly.py"),
    ("search", "search/indexer/chunker.ts", "the chunker rewrite", "rebuilding the offset arithmetic", "search/indexer/chunker.test.ts"),
    ("search", "search/query/rewriter.ts", "the rewriter tuning", "reworking how stop words are dropped", "search/query/rewriter.test.ts"),
    ("search", "search/indexer/backfill.ts", "the index backfill", "reindexing the archive partitions", "search/indexer/backfill.test.ts"),
    ("storage", "storage/blob/uploader.rs", "the multipart upload fix", "acknowledging the final part properly", "the module test block in storage/blob/uploader.rs"),
    ("storage", "storage/sql/migrations/0042_add_index.sql", "migration 0042", "adding the composite index online", "the migration dry run in CI"),
    ("storage", "storage/backup/snapshot.py", "the snapshot job", "changing where the nightly dumps land", "tests/storage/test_snapshot.py"),
    ("ingest", "ingest/pipeline/parser.py", "the parser hardening", "handling the mixed encoding sample", "tests/ingest/test_parser.py"),
    ("ingest", "ingest/pipeline/dedupe.py", "the dedupe stage audit", "counting rows in against rows out", "tests/ingest/test_dedupe.py"),
    ("ingest", "ingest/pipeline/batching.py", "the batching change", "resizing the units the pipeline moves", "tests/ingest/test_batching.py"),
    ("web ui", "ui/components/DataTable.tsx", "the data table fix", "restoring the saved column layout", "ui/components/DataTable.test.tsx"),
    ("web ui", "ui/components/DatePicker.tsx", "the date picker fix", "correcting the day boundary", "ui/components/DatePicker.test.tsx"),
    ("web ui", "ui/state/cacheSlice.ts", "the cache slice rework", "invalidating the list after an edit", "ui/state/cacheSlice.test.ts"),
    ("background jobs", "worker/queue/consumer.py", "the consumer restart work", "making redelivery safe", "tests/worker/test_consumer.py"),
    ("background jobs", "worker/scheduler/cron.py", "the scheduler audit", "replaying the schedule against a fake clock", "tests/worker/test_cron.py"),
    ("background jobs", "worker/queue/backpressure.py", "the backpressure change", "bounding how much work stays in flight", "tests/worker/test_backpressure.py"),
    ("ml pipeline", "ml/features/encoder.py", "the encoder fix", "keeping the sparse column finite", "tests/ml/test_encoder.py"),
    ("ml pipeline", "ml/serving/batcher.py", "the batcher tuning", "reshaping how requests are grouped", "tests/ml/test_batcher.py"),
    ("ml pipeline", "ml/train/loop.py", "the training loop review", "watching the loss curve on the small split", "tests/ml/test_train_loop.py"),
    ("platform", "platform/config/loader.py", "the config loader fix", "making environment selection explicit", "tests/platform/test_config_loader.py"),
    ("platform", "platform/featureflags/eval.go", "the flag evaluator fix", "stabilising a decision that flapped", "platform/featureflags/eval_test.go"),
    ("platform", "platform/logging/redact.go", "the redactor audit", "checking every field that should be masked", "platform/logging/redact_test.go"),
)

_ENTITIES: tuple[tuple[str, str, str], ...] = (
    ("the auth service", "service", "authentication"),
    ("the session store", "datastore", "authentication"),
    ("the token issuer", "service", "authentication"),
    ("the gateway", "service", "api gateway"),
    ("the rate limiter", "service", "api gateway"),
    ("the invoicing job", "job", "billing"),
    ("the webhook consumer", "service", "billing"),
    ("the indexer", "service", "search"),
    ("the query rewriter", "service", "search"),
    ("the blob uploader", "service", "storage"),
    ("the primary Postgres cluster", "datastore", "storage"),
    ("the ingest parser", "service", "ingest"),
    ("the dedupe stage", "job", "ingest"),
    ("the dashboard", "client", "web ui"),
    ("the date picker", "client", "web ui"),
    ("the queue consumer", "job", "background jobs"),
    ("the cron scheduler", "job", "background jobs"),
    ("the feature encoder", "job", "ml pipeline"),
    ("the model server", "service", "ml pipeline"),
    ("the config loader", "service", "platform"),
    ("the flag evaluator", "service", "platform"),
    ("the maintainer", "person", "platform"),
    ("the brain repo", "repo", "platform"),
)

_ATTRIBUTES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("retry budget", "count", ("service", "job", "client")),
    ("request timeout", "duration", ("service", "client", "datastore")),
    ("cache TTL", "duration", ("service", "datastore", "client")),
    ("log level", "level", ("service", "job", "client")),
    ("deploy target", "environment", ("service", "job", "client", "repo")),
    ("replica count", "count", ("service", "job", "datastore")),
    ("batch size", "count", ("job", "service")),
    ("rate limit", "rate", ("service",)),
    ("runtime version", "version", ("service", "job", "client", "repo")),
    ("schema version", "version", ("datastore", "service", "repo")),
    ("lock timeout", "duration", ("datastore", "job")),
    ("owning team", "team", ("service", "datastore", "job", "client", "repo")),
    ("test runner", "tool", ("repo", "person")),
    ("preferred editor", "tool", ("person",)),
    ("rollout state", "state", ("service", "job", "client")),
    ("backup target", "path", ("datastore", "service")),
    ("alert channel", "channel", ("service", "job", "datastore")),
)

VALUES: dict[str, tuple[str, ...]] = {
    "count": ("3", "5", "8", "16", "32", "64"),
    "duration": ("5 seconds", "30 seconds", "2 minutes", "15 minutes", "1 hour", "24 hours"),
    "level": ("debug", "info", "warning", "error", "trace", "fatal"),
    "environment": ("staging", "production", "the canary pool", "the eu-west region", "the shared dev cluster", "the on-prem tenant"),
    "rate": ("50 requests per second", "100 requests per second", "500 requests per second", "1000 requests per second", "2000 requests per second", "5000 requests per second"),
    "version": ("3.11", "3.12", "1.4.2", "2.0.0", "0.9.7", "5.1"),
    "team": ("the platform team", "the storage team", "the payments team", "the search team", "the growth team", "the sre rotation"),
    "tool": ("pytest", "nox", "tox", "hatch", "just", "make"),
    "state": ("enabled", "disabled", "enabled for the canary slice only", "disabled outside business hours", "enabled behind a kill switch", "disabled pending review"),
    "path": ("s3://prod-artifacts", "s3://staging-artifacts", "gs://brain-backups", "/mnt/shared/cache", "s3://cold-archive", "gs://nightly-dumps"),
    "channel": ("#eng-alerts", "#oncall", "#billing-ops", "#search-eng", "#platform", "the pager"),
}

_TOPIC_SENTENCES: dict[str, str] = {
    "t1": "Session on {focus} in {path}.",
    "t2": "This is the {area} workstream.",
    "t3": "Context: {activity}.",
    "t4": "The covering check is {check}.",
}

_INCIDENTAL: tuple[str, ...] = (
    "Read through the call sites before touching anything.",
    "Left a TODO where the retry path swallows errors.",
    "Rebased on main first to avoid a merge mess.",
    "The local run needed a cache clear to reproduce.",
    "Paired on this for the second half of the afternoon.",
    "Wrote the note down because it was easy to forget.",
    "The CI run sat queued for a while before it started.",
    "Checked the dashboard afterwards and it looked flat.",
    "Followed up on the review comment from yesterday.",
    "Ran the covering check twice to be sure.",
    "Took a diff against last week to see what moved.",
    "The docs in this folder were out of date again.",
    "Cleaned up two stray print statements while in here.",
    "Timeboxed this to an hour and stopped there.",
)

_LONG_ASSERTIONS: tuple[str, ...] = (
    "Noted that the {attr} for {entity} is {value}.",
    "Recorded the {attr} for {entity} as {value}.",
    "For {entity}, the {attr} is {value}.",
    "The {attr} for {entity} is now {value}.",
)

_SHORT_ASSERTIONS: tuple[str, ...] = (
    "The {attr} for {entity} is {value}.",
    "{Entity}: {attr} {value}.",
    "Set the {attr} for {entity} to {value}.",
    "{Entity} {attr} is {value}.",
)

_VARIANTS: tuple[tuple[str, ...], ...] = (
    ("t1", "t2", "a", "t3", "n1", "t4", "n2"),
    ("t1", "t3", "n1", "a", "t2", "n2", "t4"),
    ("t2", "t1", "n1", "t4", "a", "t3", "n2"),
    ("t1", "t4", "n2", "t3", "a", "n1", "t2"),
)

WORKSTREAMS: tuple[Workstream, ...] = tuple(
    Workstream(f"t{i:02d}", *row) for i, row in enumerate(_WORKSTREAMS)
)
ENTITIES: tuple[Entity, ...] = tuple(Entity(f"e{i:02d}", *row) for i, row in enumerate(_ENTITIES))
ATTRIBUTES: tuple[Attribute, ...] = tuple(
    Attribute(f"a{i:02d}", *row) for i, row in enumerate(_ATTRIBUTES)
)
TOPICS_BY_AREA: dict[str, tuple[Workstream, ...]] = {
    area: tuple(w for w in WORKSTREAMS if w.area == area)
    for area in sorted({w.area for w in WORKSTREAMS})
}
#: Every (entity, attribute) the kind system allows, in a fixed sorted order.
VALID_SLOTS: tuple[tuple[Entity, Attribute], ...] = tuple(
    (entity, attribute)
    for entity in ENTITIES
    for attribute in ATTRIBUTES
    if entity.kind in attribute.kinds
)

N_VARIANTS = len(_VARIANTS)
N_INCIDENTAL = len(_INCIDENTAL)


def render_text(
    *,
    entity: Entity,
    attribute: Attribute,
    value: str,
    topic: Workstream,
    day: str,
    variant: int,
    incidental: tuple[int, int],
    regime: str,
) -> str:
    """Text for one episode in one regime. Blind to the pair class by design."""
    fields = {
        "attr": attribute.phrase,
        "entity": entity.noun,
        "Entity": entity.noun[0].upper() + entity.noun[1:],
        "value": value,
    }
    if regime == "short":
        return _SHORT_ASSERTIONS[variant % len(_SHORT_ASSERTIONS)].format(**fields)
    fields.update(
        focus=topic.focus,
        path=topic.path,
        area=topic.area,
        activity=topic.activity,
        check=topic.check,
    )
    parts: list[str] = []
    for token in _VARIANTS[variant % N_VARIANTS]:
        if token in _TOPIC_SENTENCES:
            parts.append(_TOPIC_SENTENCES[token].format(**fields))
        elif token == "a":
            parts.append(_LONG_ASSERTIONS[variant % len(_LONG_ASSERTIONS)].format(**fields))
        else:
            parts.append(_INCIDENTAL[incidental[0 if token == "n1" else 1]])
    return f"[{day}] {topic.path}\n" + " ".join(parts)
