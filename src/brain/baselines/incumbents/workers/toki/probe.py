"""Execute TOKI's write path over its own operator set, exhaustively.

Runs INSIDE the TOKI virtual environment. Imports nothing from `brain`.

    python probe.py <input.json> <output.json>

TOKI is the one system in the set whose contradiction test is a closed-form
predicate rather than a model call, and it is stated in its own source
(`bitemporal/operators.py:is_contradiction`):

    f1 # f2  iff  subject equal AND predicate equal AND object different
                  AND the closed-open valid-time periods overlap

So the write path can be characterised completely by executing the predicate and
the four resolution operators over the three arrangements a pair of episodes can
take, with no key extraction of ours in the way. Which arrangement an
interference pair falls into is the whole question:

* **same key, overlapping validity** -- the pair is a contradiction and one row
  is closed, which is destruction of a distinct event;
* **same key, disjoint validity** -- not a contradiction, both rows stay open,
  and nothing distinguishes them from two unrelated facts;
* **different key** -- invisible to the predicate entirely.

TOKI's own paper names embedding-similarity matching as an open problem, so this
is not a deficiency found by us; it is the boundary the paper draws, measured.

Every one of the four operators (`lww`, `evi`, `await`, `rule`) is run, so the
result quantifies over TOKI's whole resolution algebra rather than its default.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any

SCHEMA = "toki-probe-v1"
OPERATORS = ("lww", "evi", "await", "rule")


def _fact(schema: Any, **overrides: Any) -> dict[str, Any]:
    base = {
        "fact_id": "f",
        "subject": "services/auth/login_flow.py",
        "predicate": "incident",
        "object": "flag cache not initialised",
        "valid_from": dt.datetime(2026, 1, 1),
        "valid_to": dt.datetime(2026, 12, 31),
        "system_time_start": dt.datetime(2026, 1, 1),
        "system_time_end": schema.SENTINEL_END,
        "provenance_id": "p1",
        "confidence": 0.8,
        "resolution_strategy_id": "lww",
    }
    base.update(overrides)
    return base


#: The three arrangements a pair of distinct episodes can take under TOKI's key.
ARRANGEMENTS = {
    "same_key_overlapping": {"subject": None, "valid_from": dt.datetime(2026, 6, 1), "valid_to": dt.datetime(2026, 12, 31)},
    "same_key_disjoint": {"subject": None, "valid_from": dt.datetime(2027, 1, 1), "valid_to": dt.datetime(2027, 12, 31)},
    "different_key": {"subject": "platform/featureflags/eval.go", "valid_from": dt.datetime(2026, 6, 1), "valid_to": dt.datetime(2026, 12, 31)},
}


def _rows(schema: Any, conn: Any, when: dt.datetime) -> list[dict[str, Any]]:
    rows = schema.as_of(conn, system_time=when)
    return [{key: str(value) for key, value in dict(row).items()} for row in rows]


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    import duckdb
    from bitemporal import Schema, ingest
    from bitemporal.operators import is_contradiction

    now = dt.datetime(2026, 12, 1)
    results: list[dict[str, Any]] = []
    for operator in OPERATORS:
        for arrangement, changes in ARRANGEMENTS.items():
            first = _fact(Schema, fact_id="a", provenance_id="pa", resolution_strategy_id=operator)
            second_changes = {key: value for key, value in changes.items() if value is not None}
            second = _fact(
                Schema,
                fact_id="b",
                provenance_id="pb",
                resolution_strategy_id=operator,
                object="rollout map not initialised",
                **second_changes,
            )
            conn = duckdb.connect(":memory:")
            try:
                Schema.create(conn)
                callback = (lambda *a, **k: True) if operator == "await" else None
                policy = (lambda *a, **k: 0) if operator == "rule" else None
                kwargs: dict[str, Any] = {}
                if callback is not None:
                    kwargs["callback"] = callback
                if policy is not None:
                    kwargs["policy"] = policy
                ingest(conn, first, now=now, **kwargs)
                outcome = ingest(conn, second, now=now, **kwargs)
                rows = _rows(Schema, conn, now)
                current = [row for row in rows if row.get("row_kind") == "current"]
                # The audit rows are TOKI's own defence against silent erasure
                # and live in a separate projection, so they are read
                # separately. Whether they fire is the whole difference between
                # a resolution that leaves a trace and one that does not.
                audit = [
                    {key: str(value) for key, value in dict(row).items()}
                    for row in Schema.audit_log(conn)
                ]
                results.append(
                    {
                        "operator": operator,
                        "arrangement": arrangement,
                        "is_contradiction": bool(is_contradiction(first, second)),
                        "resolution_fired": outcome is not None,
                        "n_current_rows": len(current),
                        "both_facts_open": {row.get("fact_id") for row in current} == {"a", "b"},
                        "n_audit_rows": len(audit),
                        "current_fact_ids": sorted({row.get("fact_id") for row in current}),
                        "rows": rows,
                        "audit_rows": audit,
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "operator": operator,
                        "arrangement": arrangement,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                conn.close()

    return {
        "arrangements": sorted(ARRANGEMENTS),
        "operators": list(OPERATORS),
        "results": results,
        "schema_columns": list(Schema.COLUMNS) + [Schema.ROW_KIND],
        "contradiction_predicate": (
            "subject equal AND predicate equal AND object different AND valid-time overlap "
            "(bitemporal/operators.py:is_contradiction)"
        ),
    }


def main() -> int:
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(source.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {"schema": SCHEMA, "spec": spec, "python": sys.version.split()[0]}
    try:
        payload["components"] = {
            "toki": "github.com/ZhoeMrety/toki-bitemporal-memory (distribution 'bitemporal')",
            "store": "DuckDB, in-memory, no server",
            "substitutions": "none; the contradiction predicate is closed form and needs no model",
        }
        payload.update(_run(spec))
        payload["ok"] = True
    except Exception as error:
        payload["ok"] = False
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
