"""EpisodicStore: one portable sqlite-vec file with k-NN over stored traces.

Spec decision T2. Nothing is ever hard-deleted: `invalidate` closes the
validity interval and the row stays, which keeps a past retrieval reproducible
and leaves the evidence trail a later provenance layer (C-C) depends on.

Determinism (rules.md): k-NN results are re-sorted by (distance, id) and
`all_traces` orders by id, which is total because ids are unique. The one
residual order dependence is which of several exactly equidistant rows takes
the k-th slot inside sqlite-vec; that scan is in rowid order, so it repeats
for a store built by the same sequence of writes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import sqlite_vec

from brain.config import ensure_dir
from brain.store import queries
from brain.store.schema import (
    META_DDL, META_TABLE, NEVER_INVALID, SCHEMA_VERSION, TRACES_TABLE,
    VEC_TABLE, Trace, all_ddl, trace_from_row, trace_to_row,
)
from brain.store.vectors import (
    DimensionMismatchError, InvalidVectorError, from_blob, prepare,
)

__all__ = ["EpisodicStore", "StoreError", "DuplicateTraceIdError",
           "InvalidVectorError", "DimensionMismatchError"]


class StoreError(RuntimeError):
    """Store could not satisfy a request (bad state, not bad arithmetic)."""


class DuplicateTraceIdError(StoreError):
    """A distinct trace already occupies this id."""


class EpisodicStore:
    """Append-only episodic trace store with vector k-NN.

    `dim` is injected rather than read from an embedder, so the store stays
    importable without torch and a change of embedder cannot silently reshape
    an existing file. Opening creates the schema; reopening at another `dim`
    is refused.
    """

    def __init__(self, db_path: str | Path, dim: int) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"embedding dim must be a positive int, got {dim!r}")
        self._dim, self._closed = int(dim), False
        if str(db_path) == ":memory:":
            self._db_path: str | Path = ":memory:"
        else:
            path = Path(db_path).expanduser()
            ensure_dir(path.parent)
            self._db_path = path
        self._conn = self._connect()
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error) as exc:
            conn.close()
            raise StoreError(
                f"sqlite-vec extension failed to load for {self._db_path}: {exc}"
            ) from exc
        return conn

    def init_schema(self) -> None:
        """Create the tables and record the store metadata. Idempotent."""
        with self._conn:
            self._conn.execute(META_DDL)
            row = self._conn.execute(
                f"SELECT value FROM {META_TABLE} WHERE key = 'dim'"
            ).fetchone()
            if row is not None and int(row[0]) != self._dim:
                raise StoreError(
                    f"{self._db_path} was created with dim {int(row[0])}, "
                    f"reopened with dim {self._dim}"
                )
            for statement in all_ddl(self._dim):
                self._conn.execute(statement)
            vec_version = self._conn.execute("SELECT vec_version()").fetchone()[0]
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES (?, ?)",
                sorted({"schema_version": str(SCHEMA_VERSION), "dim": str(self._dim),
                        "vec_version": str(vec_version)}.items()),
            )

    def close(self) -> None:
        """Commit and close. Idempotent."""
        if not self._closed:
            self._conn.commit()
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "EpisodicStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def db_path(self) -> str | Path:
        return self._db_path

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"EpisodicStore({self._db_path!r}, dim={self._dim})"

    # -- writes ------------------------------------------------------------

    def add(self, trace: Trace, vector: Any) -> str:
        """Store a trace and return the authoritative id.

        Dedup is byte-identical only: an existing trace with the same
        `content_sha` wins and nothing is inserted, so the returned id differs
        from `trace.id` exactly when the capture was a repeat. Similar-but-
        distinct traces are never merged.
        """
        vec = prepare(vector, self._dim)
        with self._conn:
            return self._add_in_txn(trace, vec)

    def add_many(self, traces: Sequence[Trace], vectors: Any) -> list[str]:
        """Add a batch in one transaction, returning ids in input order.

        A repeat inside the batch dedups against its earlier sibling.
        """
        traces = list(traces)
        if not traces:
            return []
        block = prepare(vectors, self._dim, rows=len(traces))
        with self._conn:
            return [self._add_in_txn(t, block[i]) for i, t in enumerate(traces)]

    def _add_in_txn(self, trace: Trace, vec: np.ndarray) -> str:
        """Dedup probe (unique index) plus insert. Caller owns the transaction."""
        existing = self._conn.execute(
            f"SELECT id FROM {TRACES_TABLE} WHERE content_sha = ?",
            (trace.content_sha,),
        ).fetchone()
        if existing is not None:
            return str(existing[0])
        try:
            self._conn.execute(queries.INSERT_TRACE_SQL, trace_to_row(trace))
        except sqlite3.IntegrityError as exc:
            raise DuplicateTraceIdError(
                f"cannot insert trace {trace.id!r}: id or content_sha already present"
            ) from exc
        self._conn.execute(
            queries.INSERT_VEC_SQL,
            (trace.id, vec.tobytes(), trace.project, int(trace.is_valid),
             float(trace.valid_at if trace.valid_at is not None else trace.ts),
             NEVER_INVALID if trace.invalid_at is None else float(trace.invalid_at)),
        )
        return trace.id

    def update_vector(self, trace_id: str, vector: Any) -> None:
        """Overwrite the stored x. Task 005 writes separated vectors back here."""
        vec = prepare(vector, self._dim)
        with self._conn:
            self._set_vector(str(trace_id), vec)

    def update_vectors(self, items: Iterable[tuple[str, Any]]) -> None:
        """Batch form of `update_vector`, applied in id order for determinism."""
        prepared = [
            (str(tid), prepare(vec, self._dim))
            for tid, vec in sorted(items, key=lambda item: str(item[0]))
        ]
        with self._conn:
            for trace_id, vec in prepared:
                self._set_vector(trace_id, vec)

    def _set_vector(self, trace_id: str, vec: np.ndarray) -> None:
        cur = self._conn.execute(
            f"UPDATE {VEC_TABLE} SET embedding = ? WHERE trace_id = ?",
            (vec.tobytes(), trace_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"unknown trace id {trace_id!r}")

    def invalidate(self, trace_id: str, at: float) -> None:
        """Close the validity interval at `at`. The row is never deleted."""
        stamp, tid = float(at), str(trace_id)
        with self._conn:
            cur = self._conn.execute(
                f"UPDATE {TRACES_TABLE} SET invalid_at = ? WHERE id = ?", (stamp, tid)
            )
            if cur.rowcount == 0:
                raise KeyError(f"unknown trace id {trace_id!r}")
            self._conn.execute(
                f"UPDATE {VEC_TABLE} SET is_valid = 0, invalid_at = ? WHERE trace_id = ?",
                (stamp, tid),
            )

    # -- reads -------------------------------------------------------------

    def knn(
        self, vector: Any, k: int, *, project: str | None = None,
        valid_only: bool = True, as_of: float | None = None,
    ) -> list[tuple[str, float]]:
        """k nearest traces as (trace_id, cosine distance), nearest first.

        Similarity is `1 - distance`: the sim(x_i, x_j) compared against tau in
        M-A. Filters run before the distance computation (see `queries`), so
        the result holds k matching rows, not the survivors of a post-filtered
        top-k. `as_of` supersedes `valid_only` when both are given.
        """
        vec = prepare(vector, self._dim)
        if int(k) <= 0:
            return []
        sql, params = queries.knn_query(
            vec.tobytes(), int(k), project, valid_only, as_of
        )
        rows = self._conn.execute(sql, params).fetchall()
        # Clamp only absorbs float32 rounding at the endpoints: an exact match
        # can come back as -2e-16, which would push sim = 1 - distance above 1.
        return sorted(
            ((str(row[0]), min(2.0, max(0.0, float(row[1])))) for row in rows),
            key=lambda hit: (hit[1], hit[0]),
        )

    def get(self, trace_id: str) -> Trace | None:
        """The trace, invalidated or not: invalidation hides, it never deletes."""
        row = self._conn.execute(
            f"SELECT {queries.TRACE_COLS} FROM {TRACES_TABLE} WHERE id = ?",
            (str(trace_id),),
        ).fetchone()
        return None if row is None else trace_from_row(row)

    def get_vector(self, trace_id: str) -> np.ndarray | None:
        """The stored x, or None if the id is unknown."""
        row = self._conn.execute(
            f"SELECT embedding FROM {VEC_TABLE} WHERE trace_id = ?", (str(trace_id),)
        ).fetchone()
        return None if row is None else from_blob(row[0])

    def get_vectors(self, trace_ids: Sequence[str]) -> np.ndarray:
        """Stored vectors as (n, d) float32, in the order of `trace_ids`."""
        ids = [str(t) for t in trace_ids]
        out = np.empty((len(ids), self._dim), dtype=np.float32)
        if not ids:
            return out
        marks = ", ".join("?" * len(ids))
        found = {
            str(row[0]): row[1]
            for row in self._conn.execute(
                f"SELECT trace_id, embedding FROM {VEC_TABLE} "
                f"WHERE trace_id IN ({marks})",
                ids,
            )
        }
        missing = sorted({tid for tid in ids if tid not in found})
        if missing:
            raise KeyError(f"unknown trace ids: {missing}")
        for i, tid in enumerate(ids):
            out[i] = from_blob(found[tid])
        return out

    def all_traces(
        self, *, project: str | None = None, valid_only: bool = True,
        as_of: float | None = None, limit: int | None = None,
    ) -> list[Trace]:
        """Traces ordered by id, valid ones by default."""
        where, params = queries.row_filter(project, valid_only, as_of)
        sql = queries.select_traces_sql(where, limit)
        if limit is not None:
            params = [*params, int(limit)]
        return [trace_from_row(row) for row in self._conn.execute(sql, params)]

    def count(
        self, *, project: str | None = None, valid_only: bool = False,
        as_of: float | None = None,
    ) -> int:
        """Number of stored traces. Counts invalidated ones unless asked not to."""
        where, params = queries.row_filter(project, valid_only, as_of)
        row = self._conn.execute(
            f"SELECT count(*) FROM {TRACES_TABLE}{where}", params
        ).fetchone()
        return int(row[0])
