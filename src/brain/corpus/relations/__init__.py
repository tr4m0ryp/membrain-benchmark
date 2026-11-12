"""The four-class relational corpus (post-write correctness, task 001, gate B0).

An ADDITIVE split at `data/corpus/relations/`. The existing frozen corpus at
`data/corpus/synthetic/` holds interference pairs only, which is why nothing in
the repo can currently test I1 at all: its whole claim is that supersession and
interference are different operations, and one of the two classes is absent.

Every episode here carries an explicit referent `r = (entity, attribute)` and an
asserted value `v`, so the ground-truth class of ANY pair is the spec's action
map applied to `(rho, nu)`, computable from `episodes.jsonl` alone. `pairs.jsonl`
is the released sample at the Evaluation Plan composition.

Typical use:

    from brain.corpus.relations import load_relations
    corpus = load_relations()      # verifies the content hash, raises on drift

Regenerate and freeze:

    .venv/bin/python -m brain.corpus.relations.freeze
"""

from brain.corpus.relations.content import ATTRIBUTES, ENTITIES, VALID_SLOTS, VALUES, WORKSTREAMS
from brain.corpus.relations.freeze import (
    RelationsAlreadyFrozen,
    build_manifest,
    freeze_relations,
    spotcheck_rows,
)
from brain.corpus.relations.loader import (
    RELATIONS_DIR,
    SYNTHETIC_CONTENT_SHA256,
    assert_synthetic_corpus_unchanged,
    content_hashes,
    load_relations,
)
from brain.corpus.relations.generate import (
    RelationsConfig,
    default_relations_config,
    generate_relations,
)
from brain.corpus.relations.types import (
    ACTIONS,
    ACTION_MAP,
    CLASSES,
    COINCIDENCE,
    DUPLICATE,
    INTERFERENCE,
    REGIMES,
    SCHEMA_VERSION,
    SUPERSESSION,
    LabelledPair,
    RelationCorpus,
    RelationEpisode,
    classify,
    pair_of,
    validate_relations,
)

__all__ = [
    "SCHEMA_VERSION",
    "RELATIONS_DIR",
    "SYNTHETIC_CONTENT_SHA256",
    "SUPERSESSION",
    "INTERFERENCE",
    "DUPLICATE",
    "COINCIDENCE",
    "CLASSES",
    "REGIMES",
    "ACTION_MAP",
    "ACTIONS",
    "ATTRIBUTES",
    "ENTITIES",
    "VALUES",
    "VALID_SLOTS",
    "WORKSTREAMS",
    "RelationEpisode",
    "LabelledPair",
    "RelationCorpus",
    "RelationsConfig",
    "RelationsAlreadyFrozen",
    "classify",
    "pair_of",
    "validate_relations",
    "default_relations_config",
    "generate_relations",
    "content_hashes",
    "build_manifest",
    "spotcheck_rows",
    "freeze_relations",
    "load_relations",
    "assert_synthetic_corpus_unchanged",
]
