"""Public benchmarks mapped onto this harness's corpus interface.

Everything measured in this project so far was measured on a corpus we
generated to test our own claim. A finding that only holds there is a finding
about the generator. This package exists so a claim can be re-run on data that
is not ours, without changing a line of the machinery it is a claim about:
an adapter emits the same `Episode` / `Query` / `Corpus` records
`brain.corpus.loader` emits, and every arm, every metric and every bootstrap
downstream stays byte-identical.

Adapters live one directory per benchmark. `locomo` is the first.
"""

from brain.corpus.external.locomo import (
    LOCOMO_COMMIT,
    LOCOMO_JSON_SHA256,
    LOCOMO_REPO_URL,
    LocomoConversation,
    DISTRACTOR_SCOPES,
    ensure_locomo,
    load_locomo,
    locomo_provenance,
    scoped_queries,
)

__all__ = [
    "DISTRACTOR_SCOPES",
    "LOCOMO_COMMIT",
    "LOCOMO_JSON_SHA256",
    "LOCOMO_REPO_URL",
    "LocomoConversation",
    "ensure_locomo",
    "load_locomo",
    "locomo_provenance",
    "scoped_queries",
]
