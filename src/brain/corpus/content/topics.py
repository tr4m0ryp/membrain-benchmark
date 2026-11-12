"""Topic bank: the SHARED surface of an interference cluster.

Every episode in a cluster carries the same topic (same file, same subsystem,
same component, same error family, same repro path, same log signal). That
shared surface is what makes siblings near-duplicates; the per-episode axes in
`axes.py` are the only thing that tells them apart. One topic is used by at most
one cluster, so a cluster is uniquely identified by its file path.

Field order: path, subsystem, component, symptom, language, repro, signal, check.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Topic", "TOPICS"]


@dataclass(frozen=True, slots=True)
class Topic:
    topic_id: str
    path: str
    subsystem: str
    component: str
    symptom: str
    language: str
    repro: str
    signal: str
    check: str


_RAW: tuple[tuple[str, ...], ...] = (
    ("services/auth/session_store.py", "authentication", "the session store", "request timeout",
     "Python", "POST /login twice against a warm cache", "a 30 second deadline in the gateway log",
     "tests/auth/test_session_store.py"),
    ("services/auth/token_refresh.py", "authentication", "the refresh path", "401 storm",
     "Python", "let an access token expire mid-session", "a burst of unauthorized responses in the access log",
     "tests/auth/test_token_refresh.py"),
    ("api/gateway/rate_limit.go", "api gateway", "the limiter", "spurious 429",
     "Go", "replay the saved burst fixture", "throttle counters far above the configured ceiling",
     "api/gateway/rate_limit_test.go"),
    ("api/gateway/router.go", "api gateway", "the router", "route shadowing",
     "Go", "curl the two overlapping prefixes in order", "the wrong handler name in the request span",
     "api/gateway/router_test.go"),
    ("billing/invoice_totals.py", "billing", "the totals calculator", "off-by-one cent",
     "Python", "total an invoice with three prorated lines", "a one cent gap in the reconciliation report",
     "tests/billing/test_invoice_totals.py"),
    ("billing/webhooks/stripe.py", "billing", "the webhook consumer", "duplicate charge",
     "Python", "redeliver the same webhook id twice", "two ledger rows sharing one event id",
     "tests/billing/test_stripe_webhook.py"),
    ("search/indexer/chunker.ts", "search", "the chunker", "truncated documents",
     "TypeScript", "index the long changelog fixture", "chunk offsets that stop short of the document length",
     "search/indexer/chunker.test.ts"),
    ("search/query/rewriter.ts", "search", "the query rewriter", "empty result set",
     "TypeScript", "search for a two word phrase with a stop word", "a rewritten query with no clauses left",
     "search/query/rewriter.test.ts"),
    ("storage/blob/uploader.rs", "storage", "the uploader", "partial upload",
     "Rust", "upload the 40 MB fixture over a throttled link", "a final part that never gets acknowledged",
     "the module test block in storage/blob/uploader.rs"),
    ("storage/sql/migrations/0042_add_index.sql", "storage", "migration 0042", "lock timeout",
     "SQL", "run the migration against a loaded table", "a lock wait that trips the statement timeout",
     "the migration dry run in CI"),
    ("ingest/pipeline/parser.py", "ingest", "the parser", "malformed record",
     "Python", "feed the mixed-encoding sample file", "a decode error pointing at one byte offset",
     "tests/ingest/test_parser.py"),
    ("ingest/pipeline/dedupe.py", "ingest", "the dedupe stage", "dropped rows",
     "Python", "run the pipeline over the overlapping day partitions", "an output count below the input count",
     "tests/ingest/test_dedupe.py"),
    ("ui/components/DataTable.tsx", "web ui", "the data table", "blank render",
     "TypeScript", "open the table with a saved column layout", "an empty tbody with no console error",
     "ui/components/DataTable.test.tsx"),
    ("ui/components/DatePicker.tsx", "web ui", "the date picker", "off-by-one day",
     "TypeScript", "pick the first of the month and reopen", "a selected value one day behind the label",
     "ui/components/DatePicker.test.tsx"),
    ("ui/state/cacheSlice.ts", "web ui", "the cache slice", "stale state",
     "TypeScript", "edit a record then navigate back to the list", "an old value still in the store snapshot",
     "ui/state/cacheSlice.test.ts"),
    ("infra/terraform/vpc.tf", "infrastructure", "the VPC module", "plan drift",
     "HCL", "run plan twice with no changes in between", "a diff on a resource nobody edited",
     "the terraform plan check in CI"),
    ("infra/k8s/deployment.yaml", "infrastructure", "the deployment", "crash loop",
     "YAML", "roll out to the small node pool", "a container restarting a few seconds after start",
     "the manifest lint and dry run in CI"),
    ("worker/queue/consumer.py", "background jobs", "the consumer", "message redelivery",
     "Python", "kill the worker mid-batch and restart it", "the same message id handled more than once",
     "tests/worker/test_consumer.py"),
    ("worker/scheduler/cron.py", "background jobs", "the scheduler", "missed run",
     "Python", "advance the fake clock across the schedule", "a slot with no execution record",
     "tests/worker/test_cron.py"),
    ("ml/features/encoder.py", "ml pipeline", "the feature encoder", "NaN feature",
     "Python", "encode the batch holding the sparse column", "a not-a-number value in one output column",
     "tests/ml/test_encoder.py"),
    ("ml/train/loop.py", "ml pipeline", "the training loop", "loss divergence",
     "Python", "train two epochs on the small split", "a loss curve that turns upward and never returns",
     "tests/ml/test_train_loop.py"),
    ("ml/serving/batcher.py", "ml pipeline", "the batcher", "latency spike",
     "Python", "send the mixed-size request trace", "a tail far above the median in the serving histogram",
     "tests/ml/test_batcher.py"),
    ("analytics/etl/rollup.sql", "analytics", "the daily rollup", "double counting",
     "SQL", "rerun the rollup for a day already loaded", "totals larger than the source table",
     "the rollup reconciliation check in CI"),
    ("analytics/dashboards/funnel.py", "analytics", "the funnel report", "missing step",
     "Python", "build the funnel for the two week window", "a stage with zero users between two populated ones",
     "tests/analytics/test_funnel.py"),
    ("mobile/ios/SyncEngine.swift", "mobile sync", "the sync engine", "conflict loop",
     "Swift", "edit the same note on two devices offline", "two clients rewriting each other on every pass",
     "mobile/ios/SyncEngineTests.swift"),
    ("mobile/android/OfflineCache.kt", "mobile sync", "the offline cache", "data loss",
     "Kotlin", "force stop the app with unsynced edits", "a local row missing after the next launch",
     "mobile/android/OfflineCacheTest.kt"),
    ("platform/logging/redact.go", "platform", "the redactor", "leaked field",
     "Go", "log the nested payload fixture", "a value that should have been masked in the log line",
     "platform/logging/redact_test.go"),
    ("platform/config/loader.py", "platform", "the config loader", "wrong environment",
     "Python", "boot the service with an unset profile variable", "a production endpoint in a staging process",
     "tests/platform/test_config_loader.py"),
    ("platform/featureflags/eval.go", "platform", "the flag evaluator", "flapping flag",
     "Go", "evaluate the same flag for one user repeatedly", "a decision that alternates between calls",
     "platform/featureflags/eval_test.go"),
    ("notifications/email/render.py", "notifications", "the email renderer", "broken layout",
     "Python", "render the digest template with a long subject", "a table that overflows in the plain text part",
     "tests/notifications/test_render.py"),
    ("notifications/push/dispatch.py", "notifications", "the dispatcher", "silent drop",
     "Python", "dispatch to a device token that was rotated", "a send that reports success with no delivery",
     "tests/notifications/test_dispatch.py"),
    ("crypto/keys/rotation.py", "security", "key rotation", "signature mismatch",
     "Python", "rotate the signing key while requests are in flight", "a verification failure against the previous key",
     "tests/crypto/test_rotation.py"),
    ("crypto/tls/handshake.go", "security", "the handshake", "intermittent EOF",
     "Go", "open many short lived connections in a row", "an unexpected end of stream during negotiation",
     "crypto/tls/handshake_test.go"),
    ("cli/commands/deploy.py", "developer tooling", "the deploy command", "hang on exit",
     "Python", "deploy the sample project and wait", "a process that finishes work but never returns",
     "tests/cli/test_deploy.py"),
    ("cli/commands/init.py", "developer tooling", "the init command", "corrupted scaffold",
     "Python", "init into a directory that already has a lockfile", "a generated tree missing one file",
     "tests/cli/test_init.py"),
    ("build/bazel/BUILD.bazel", "build system", "the build graph", "cache miss storm",
     "Starlark", "build twice with no source change", "a rebuild of targets that should have been cached",
     "the remote cache hit rate check in CI"),
    ("build/ci/pipeline.yml", "build system", "the CI pipeline", "flaky job",
     "YAML", "rerun the same commit ten times", "one job failing on a minority of runs",
     "the pipeline replay job"),
    ("graph/traversal/walker.rs", "graph engine", "the walker", "infinite recursion",
     "Rust", "walk the cyclic fixture graph", "a stack that grows until the guard trips",
     "the module test block in graph/traversal/walker.rs"),
    ("graph/store/adjacency.rs", "graph engine", "the adjacency store", "memory blowup",
     "Rust", "load the dense neighbourhood fixture", "resident memory climbing with every batch",
     "the module test block in graph/store/adjacency.rs"),
    ("realtime/ws/session.ts", "realtime", "the websocket session", "dropped frames",
     "TypeScript", "hold a socket open through a network flap", "frames missing from the client sequence",
     "realtime/ws/session.test.ts"),
)

TOPICS: tuple[Topic, ...] = tuple(
    Topic(f"t{index:02d}", *fields) for index, fields in enumerate(_RAW)
)
