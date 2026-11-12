"""Dump everything Mem0 persisted, so the scoring can be done outside Mem0.

The probe's questions are about what survives a write, so a partial dump would
beg them. Three stores are read in full: the memory vector store, the entity
vector store where the version has one, and every table of the history
database. Embeddings are dropped (they are derived, and enormous); nothing else
is filtered, because a field this module chose to skip could be exactly the one
that records a relation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

#: Payload keys that are derived from the text and would swamp the dump.
_DROPPED = ("vector", "embedding", "fact_embedding", "text_lemmatized")


def _rows(listed: Any) -> list[Any]:
    """Mem0's `list()` returns a list of result pages in some versions."""
    if listed is None:
        return []
    out: list[Any] = []
    for item in listed:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out


def _clean(payload: Any) -> dict[str, Any]:
    data = dict(payload or {})
    for key in _DROPPED:
        data.pop(key, None)
    return data


def dump_store(store: Any, filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Every record of one vector store, id-sorted, embeddings removed."""
    if store is None:
        return []
    try:
        listed = store.list(filters=filters, top_k=10000)
    except TypeError:
        listed = store.list(filters=filters, limit=10000)
    except Exception:  # a store that cannot enumerate is reported as such
        return [{"id": None, "error": "vector store does not support list()"}]
    records = [
        {"id": str(getattr(row, "id", "")), "payload": _clean(getattr(row, "payload", None))}
        for row in _rows(listed)
    ]
    records.sort(key=lambda record: record["id"])
    return records


def dump_history(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Every row of every table in the history database."""
    db = Path(path)
    if not db.is_file():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    connection = sqlite3.connect(str(db))
    try:
        connection.row_factory = sqlite3.Row
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        for table in tables:
            if table.startswith("sqlite_"):
                continue
            rows = [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
            for row in rows:
                for key in _DROPPED:
                    row.pop(key, None)
            out[table] = rows
    finally:
        connection.close()
    return out


def dump_memory(memory: Any, *, user_id: str, history_db_path: str | Path) -> dict[str, Any]:
    """The full persisted state of one Mem0 instance."""
    filters = {"user_id": user_id}
    state: dict[str, Any] = {
        "memories": dump_store(getattr(memory, "vector_store", None), filters),
        "history": dump_history(history_db_path),
    }
    entity_store = getattr(memory, "entity_store", None)
    state["entities"] = dump_store(entity_store, filters) if entity_store is not None else []
    state["has_entity_store"] = entity_store is not None
    return state
