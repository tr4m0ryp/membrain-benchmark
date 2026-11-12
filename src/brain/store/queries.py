"""Statement construction for the two tables that hold a trace.

The symbolic table and the vector table are filtered by the same three
predicates (scope, validity, as-of instant), expressed twice because one is
plain SQL and the other is a vec0 metadata constraint. Building both here
keeps them provably parallel: a k-NN and an `all_traces` with the same
arguments select the same set of traces.

The vec0 predicates are pushed into the k-NN query on purpose. sqlite-vec
applies metadata constraints before the distance computation, so a filtered
search returns k matching rows; filtering afterwards would return fewer than
k, or none at all, whenever the nearest neighbours sit in another project.
"""

from __future__ import annotations

from typing import Any

from brain.store.schema import TRACE_COLUMNS, TRACES_TABLE, VEC_COLUMNS, VEC_TABLE

__all__ = [
    "TRACE_COLS",
    "INSERT_TRACE_SQL",
    "INSERT_VEC_SQL",
    "row_filter",
    "knn_query",
    "select_traces_sql",
]

TRACE_COLS = ", ".join(TRACE_COLUMNS)
_TRACE_MARKS = ", ".join("?" * len(TRACE_COLUMNS))
_VEC_COLS = ", ".join(VEC_COLUMNS)
_VEC_MARKS = ", ".join("?" * len(VEC_COLUMNS))

INSERT_TRACE_SQL = (
    f"INSERT INTO {TRACES_TABLE} ({TRACE_COLS}) VALUES ({_TRACE_MARKS})"
)
INSERT_VEC_SQL = f"INSERT INTO {VEC_TABLE} ({_VEC_COLS}) VALUES ({_VEC_MARKS})"


def row_filter(
    project: str | None, valid_only: bool, as_of: float | None
) -> tuple[str, list[Any]]:
    """WHERE clause over the symbolic table. `as_of` supersedes `valid_only`."""
    clauses: list[str] = []
    params: list[Any] = []
    if project is not None:
        clauses.append("project = ?")
        params.append(str(project))
    if as_of is not None:
        clauses.append("valid_at <= ? AND (invalid_at IS NULL OR invalid_at > ?)")
        params += [float(as_of), float(as_of)]
    elif valid_only:
        clauses.append("invalid_at IS NULL")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def knn_query(
    blob: bytes,
    k: int,
    project: str | None,
    valid_only: bool,
    as_of: float | None,
) -> tuple[str, list[Any]]:
    """k-NN statement over the vector table, with every filter pre-applied."""
    clauses = ["embedding MATCH ?", "k = ?"]
    params: list[Any] = [blob, int(k)]
    if project is not None:
        clauses.append("project = ?")
        params.append(str(project))
    if as_of is not None:
        clauses += ["valid_at <= ?", "invalid_at > ?"]
        params += [float(as_of), float(as_of)]
    elif valid_only:
        clauses.append("is_valid = 1")
    sql = (
        f"SELECT trace_id, distance FROM {VEC_TABLE} WHERE " + " AND ".join(clauses)
    )
    return sql, params


def select_traces_sql(where: str, limit: int | None) -> str:
    """Ordered by id: unique ids make the order total, hence reproducible."""
    sql = f"SELECT {TRACE_COLS} FROM {TRACES_TABLE}{where} ORDER BY id"
    return sql if limit is None else sql + " LIMIT ?"
