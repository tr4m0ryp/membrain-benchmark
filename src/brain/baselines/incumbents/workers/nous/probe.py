"""Drive Nous's write path and dump what it persisted.

Runs INSIDE the nous-state virtual environment. Imports nothing from `brain`.

    python probe.py <input.json> <output.json>

Nous is the one system in the comparison set that needs no credential at all:
`nous-state` declares zero runtime dependencies and stores into stdlib sqlite3,
and its default extractor is rule-based, so `observe()` runs offline exactly as
shipped. Nothing is substituted here, which makes it the only arm in this report
measured with its own components end to end.

Its write is `observe(text)`: extract (entity, attribute, value) claims, append
the raw turn to `evidence_events`, and Bayesian-update one belief dimension per
`entity.attribute`. Two things are therefore worth dumping and are dumped in
full: the evidence layer, where nothing is ever replaced, and the belief layer,
where two claims on the same dimension become competing values of one variable.

Nous carries no per-record identifier of ours, so attribution is by exact text.
That is the strictest reading available and it is the right one: if the stored
text is no longer byte-identical to the episode, the episode is no longer
separately addressable by it.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any

SCHEMA = "nous-probe-v1"


def dump_db(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Every row of every table, embeddings excluded."""
    if not path.is_file():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    connection = sqlite3.connect(str(path))
    try:
        connection.row_factory = sqlite3.Row
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        for table in names:
            if table.startswith("sqlite_") or table == "embeddings":
                continue
            out[table] = [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
    finally:
        connection.close()
    return out


def _memories(state: dict[str, list[dict[str, Any]]], writes: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Recast the evidence layer into the shared record shape the scoring uses.

    A record is attributable to an episode when Nous still holds that episode's
    text verbatim. The belief dimensions are carried alongside as fields on the
    record, so a dimension shared between the two episodes shows up as a field
    they both hold, which is what the structural signature compares.
    """
    by_text = {str(write["text"]): str(write["trace_id"]) for write in writes}
    records = []
    for event in state.get("evidence_events", []):
        text = str(event.get("text", ""))
        payload = dict(event)
        payload.pop("text", None)
        payload["data"] = text
        trace_id = by_text.get(text)
        if trace_id:
            payload["trace_id"] = trace_id
        records.append({"id": str(event.get("id")), "payload": payload})
    records.sort(key=lambda record: record["id"])
    return records


def run_case(case: dict[str, Any], spec: dict[str, Any], workdir: Path) -> dict[str, Any]:
    from nous import Nous

    db = workdir / f"{case['case_id']}.db"
    memory = Nous(str(db))
    deltas: list[Any] = []
    for index, record in enumerate(case["writes"]):
        produced = memory.observe(
            record["text"],
            source=str(record.get("trace_id", "user")),
            timestamp=float(spec.get("base_time", 1_760_000_000.0)) + 86400.0 * index,
        )
        deltas.append([str(delta) for delta in (produced or [])])
    state = dump_db(db)
    return {
        "case_id": case["case_id"],
        "writes": case["writes"],
        "add_results": deltas,
        "state": {
            "memories": _memories(state, case["writes"]),
            "entities": [],
            "tables": {name: len(rows) for name, rows in state.items()},
            "dimensions": state.get("dimensions", []),
            "deltas": state.get("deltas", []),
            "relationship_edges": state.get("relationship_edges", []),
            "has_entity_store": False,
        },
        "judge": {"write": {"judge": "none (rule-based extractor, as shipped)", "calls": []}},
        "extracted_claims": sum(len(item) for item in deltas),
    }


#: Nous's shipped extractor targets conversational personal facts, and the
#: frozen corpus is engineering session notes, so a zero-claim result there says
#: "out of domain" rather than "keeps everything". These four cases are written
#: in Nous's own domain, in its README's idiom, so its belief layer actually
#: engages and the third-branch question can be put to it properly.
DOMAIN_CASES = (
    {
        "case_id": "domain-interference",
        "note": "two facts that are BOTH true and land on the same belief dimension",
        "writes": [
            {"trace_id": "d-a", "text": "Pranav works at Google DeepMind."},
            {"trace_id": "d-b", "text": "Pranav works at IIT Ropar."},
        ],
    },
    {
        "case_id": "domain-supersession",
        "note": "a later fact that replaces an earlier one on the same dimension",
        "writes": [
            {"trace_id": "d-a", "text": "Pranav works at Google DeepMind."},
            {"trace_id": "d-c", "text": "Pranav now works at OpenAI."},
        ],
    },
    {
        "case_id": "domain-control",
        "note": "two unrelated facts about different subjects",
        "writes": [
            {"trace_id": "d-a", "text": "Pranav works at Google DeepMind."},
            {"trace_id": "d-d", "text": "Rahul works at Infosys."},
        ],
    },
)


def run_domain_cases(spec: dict[str, Any], workdir: Path) -> list[dict[str, Any]]:
    """The same probe, in Nous's own domain, with its belief layer engaged."""
    out = []
    for case in DOMAIN_CASES:
        record = run_case(dict(case), spec, workdir)
        record["note"] = case["note"]
        record["belief_dimensions"] = [
            {
                "id": row.get("id"),
                "distribution": row.get("distribution"),
            }
            for row in record["state"]["dimensions"]
        ]
        out.append(record)
    return out


def main() -> int:
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(source.read_text(encoding="utf-8"))
    workdir = Path(spec.get("workdir") or (destination.parent / "work"))
    workdir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema": SCHEMA, "python": sys.version.split()[0]}
    try:
        import nous

        payload["components"] = {
            "nous-state": str(getattr(nous, "__version__", "unknown")),
            "extractor": "rule-based, shipped default (no LLM, no key)",
            "store": "stdlib sqlite3",
            "substitutions": "none",
        }
        payload["cases"] = [run_case(case, spec, workdir) for case in spec.get("cases", [])]
        payload["domain_cases"] = run_domain_cases(spec, workdir)
        payload["ok"] = True
    except Exception as error:
        payload["ok"] = False
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
