"""Row contract and SQL DDL for the episodic trace store (spec decision T2).

The whole brain is one portable SQLite file. Two tables carry a trace:
`traces` holds the symbolic record (text, salience, bi-temporal validity) and
the sqlite-vec virtual table `vec_traces` holds the embedding x under the same
id. The trace contract from the spec Architecture is
`{id, session, ts, project, text, x, salience, valid_at, invalid_at}`, extended
here with `content_sha` (byte-identical dedup) and `cluster_id` (corpus labels).

Two schema choices are load-bearing and deliberate:

1. Scope and validity are MIRRORED into the vector table. sqlite-vec applies
   metadata predicates before the distance computation, so a filtered k-NN
   still returns k matching rows. Filtering after the search would return
   fewer than k, or none at all, whenever the nearest neighbours happen to
   belong to another project.
2. `content_sha` carries a UNIQUE index. Dedup is byte-identical only, so the
   invariant is a database constraint rather than a convention, and the lookup
   on the write path is an index probe instead of a scan. Similar-but-distinct
   traces are never merged: that is the failure this project exists to fix.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "TRACES_TABLE",
    "VEC_TABLE",
    "META_TABLE",
    "NEVER_INVALID",
    "TRACE_COLUMNS",
    "VEC_COLUMNS",
    "Trace",
    "sha256_text",
    "trace_to_row",
    "trace_from_row",
    "TRACES_DDL",
    "META_DDL",
    "INDEX_DDL",
    "vec_table_ddl",
    "all_ddl",
]

SCHEMA_VERSION = 1

TRACES_TABLE = "traces"
VEC_TABLE = "vec_traces"
META_TABLE = "store_meta"

#: Sentinel for "never invalidated" inside the vector table. vec0 metadata
#: columns reject NULL, and +inf keeps the as-of predicate `invalid_at > t`
#: true for every t, which is exactly the semantics of an open interval.
NEVER_INVALID = float("inf")

TRACE_COLUMNS: tuple[str, ...] = (
    "id",
    "session",
    "ts",
    "project",
    "text",
    "content_sha",
    "salience",
    "valid_at",
    "invalid_at",
    "cluster_id",
)

VEC_COLUMNS: tuple[str, ...] = (
    "trace_id",
    "embedding",
    "project",
    "is_valid",
    "valid_at",
    "invalid_at",
)


def sha256_text(text: str) -> str:
    """Content hash over the UTF-8 bytes of `text`.

    This is the only identity used for deduplication. Two traces dedup when
    their bytes match, never when their meanings are close.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Trace:
    """One episodic trace. The embedding x is held in the vector table.

    `id` and `content_sha` are derived from the text when omitted, so a trace
    built twice from the same capture gets the same id in any process. That
    keeps a corpus reproducible without a random uuid.

    Empty string, not NULL, is the "unset" value for `session` and `project`:
    vec0 metadata columns reject NULL, and the mirrored scope column has to
    match the symbolic one exactly for a filtered k-NN to be correct.
    """

    text: str
    id: str = ""
    session: str = ""
    ts: float = 0.0
    project: str = ""
    salience: float = 1.0
    valid_at: float | None = None
    invalid_at: float | None = None
    cluster_id: str | None = None
    content_sha: str = ""

    def __post_init__(self) -> None:
        sha = self.content_sha or sha256_text(self.text)
        object.__setattr__(self, "content_sha", sha)
        if not self.id:
            object.__setattr__(self, "id", f"t_{sha[:16]}")
        if self.valid_at is None:
            object.__setattr__(self, "valid_at", float(self.ts))

    @property
    def is_valid(self) -> bool:
        """True while the trace has not been invalidated. Nothing is deleted."""
        return self.invalid_at is None


def trace_to_row(trace: Trace) -> tuple[Any, ...]:
    """Flatten a trace into `TRACE_COLUMNS` order."""
    return (
        trace.id,
        trace.session,
        float(trace.ts),
        trace.project,
        trace.text,
        trace.content_sha,
        float(trace.salience),
        float(trace.valid_at if trace.valid_at is not None else trace.ts),
        None if trace.invalid_at is None else float(trace.invalid_at),
        trace.cluster_id,
    )


def trace_from_row(row: Sequence[Any]) -> Trace:
    """Rebuild a trace from a row selected in `TRACE_COLUMNS` order."""
    return Trace(
        id=row[0],
        session=row[1],
        ts=row[2],
        project=row[3],
        text=row[4],
        content_sha=row[5],
        salience=row[6],
        valid_at=row[7],
        invalid_at=row[8],
        cluster_id=row[9],
    )


TRACES_DDL = f"""
CREATE TABLE IF NOT EXISTS {TRACES_TABLE} (
    id          TEXT PRIMARY KEY,
    session     TEXT NOT NULL DEFAULT '',
    ts          REAL NOT NULL DEFAULT 0.0,
    project     TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    content_sha TEXT NOT NULL,
    salience    REAL NOT NULL DEFAULT 1.0,
    valid_at    REAL NOT NULL DEFAULT 0.0,
    invalid_at  REAL,
    cluster_id  TEXT
)
""".strip()

META_DDL = (
    f"CREATE TABLE IF NOT EXISTS {META_TABLE} "
    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
)

INDEX_DDL: tuple[str, ...] = (
    # Unique: byte-identical dedup is an invariant, and the write path probes
    # this index instead of scanning.
    f"CREATE UNIQUE INDEX IF NOT EXISTS traces_content_sha "
    f"ON {TRACES_TABLE}(content_sha)",
    f"CREATE INDEX IF NOT EXISTS traces_project ON {TRACES_TABLE}(project)",
    f"CREATE INDEX IF NOT EXISTS traces_cluster ON {TRACES_TABLE}(cluster_id)",
)


def vec_table_ddl(dim: int) -> str:
    """DDL for the sqlite-vec virtual table holding the d-dimensional x.

    Cosine distance is declared explicitly rather than relying on unit-norm
    inputs plus L2: the separation transform (Task 005) rewrites stored
    vectors, and a metric that is only correct while the norms stay at 1 would
    be a silent trap.
    """
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ValueError(f"embedding dim must be a positive int, got {dim!r}")
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0(\n"
        f"    trace_id TEXT PRIMARY KEY,\n"
        f"    embedding float[{dim}] distance_metric=cosine,\n"
        f"    project TEXT,\n"
        f"    is_valid INTEGER,\n"
        f"    valid_at FLOAT,\n"
        f"    invalid_at FLOAT\n"
        f")"
    )


def all_ddl(dim: int) -> tuple[str, ...]:
    """Every statement needed to bring an empty file up to `SCHEMA_VERSION`."""
    return (META_DDL, TRACES_DDL, vec_table_ddl(dim), *INDEX_DDL)
