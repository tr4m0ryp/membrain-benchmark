"""Execute Graphiti's edge-resolution branch set exhaustively.

Runs INSIDE the Graphiti virtual environment. Imports nothing from `brain`.

    python probe.py <input.json> <output.json>

Why exhaustive rather than sampled. `resolve_extracted_edge` decides what
happens to a new fact that collides with an existing one, and it decides it from
a single structured response whose schema is `EdgeDuplicate`: two integer lists,
`duplicate_facts` and `contradicted_facts`, and nothing else. With one duplicate
candidate and one invalidation candidate in context, the entire space of
meaningful responses has eight elements. All eight are executed here, so the
result is not "an LLM did not record a relation" but "no response the schema
admits records one". That removes the model from the argument, which is what a
claim about architecture requires.

The function takes its candidate edges as plain arguments and never touches a
graph driver, so this runs with no database, no API key and no network.

Three things are recorded per response:

* the returned `(resolved, duplicates, invalidated)` triple, serialised;
* whether that triple differs from the one returned when NO candidate edge is
  offered at all, which is the control for the third-branch claim; and
* whether the extracted edge itself came back mutated.

Schema probes are run alongside: responses carrying invented relation fields, to
see whether the response model can express a keep-both instruction, and a full
field listing of `EntityEdge`, to see whether the stored form could hold one.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from itertools import chain, combinations
from pathlib import Path
from typing import Any

SCHEMA = "graphiti-probe-v1"
GROUP = "brain-probe"


def _powerset(items: list[int]) -> list[list[int]]:
    return [list(subset) for subset in chain.from_iterable(combinations(items, size) for size in range(len(items) + 1))]


class StubLLMClient:
    """Returns a fixed structured response. No model, no network, no key.

    Duck-typed rather than subclassed: `resolve_extracted_edge` needs exactly
    `generate_response`, and going through `LLMClient.__init__` would demand a
    provider config that only exists to hold an API key.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = dict(response)
        self.prompts: list[str] = []

    async def generate_response(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append("\n\n".join(str(getattr(message, "content", message)) for message in messages))
        return dict(self.response)


def _edge(
    edge_cls: Any, *, fact: str, source: str, target: str, valid_at: datetime, name: str = "RELATES_TO"
) -> Any:
    return edge_cls(
        uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, f"brain/{fact}")),
        group_id=GROUP,
        source_node_uuid=source,
        target_node_uuid=target,
        name=name,
        fact=fact,
        episodes=[],
        created_at=valid_at,
        valid_at=valid_at,
    )


def _serialise(edge: Any) -> dict[str, Any]:
    data = edge.model_dump(mode="json")
    data.pop("fact_embedding", None)
    return data


def _triple(result: tuple[Any, list[Any], list[Any]]) -> dict[str, Any]:
    # The return order is (resolved, invalidated, duplicates). The docstring in
    # graphiti 0.29.2 states the last two the other way round; the code is
    # authoritative and was checked against `resolve_edge_contradictions`.
    resolved, invalidated, duplicates = result
    return {
        "resolved": _serialise(resolved),
        "duplicates": [_serialise(item) for item in duplicates],
        "invalidated": [_serialise(item) for item in invalidated],
    }


#: Two temporal shapes. Graphiti's invalidation is time-ordered, so the shape
#: decides whether the invalidate branch can fire at all, and both cases occur
#: in the relational corpus: a later fact that supersedes an earlier one, and
#: two distinct events on the same day, which is the interference case.
SHAPES = {
    "supersession": {"candidates": datetime(2026, 1, 1, tzinfo=timezone.utc), "new": datetime(2026, 2, 1, tzinfo=timezone.utc)},
    "interference": {"candidates": datetime(2026, 1, 1, tzinfo=timezone.utc), "new": datetime(2026, 1, 1, tzinfo=timezone.utc)},
}


async def _shape(spec: dict[str, Any], name: str, times: dict[str, datetime]) -> dict[str, Any]:
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EpisodicNode, EpisodeType
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge

    source_uuid, target_uuid = str(uuid.uuid4()), str(uuid.uuid4())
    episode = EpisodicNode(
        uuid=str(uuid.uuid4()),
        name="probe-episode",
        group_id=GROUP,
        source=EpisodeType.text,
        source_description="brain incumbent probe",
        content=str(spec["new_fact"]),
        valid_at=times["new"],
        created_at=times["new"],
    )

    def fresh() -> tuple[Any, list[Any], list[Any]]:
        return (
            _edge(EntityEdge, fact=str(spec["new_fact"]), source=source_uuid, target=target_uuid, valid_at=times["new"]),
            [
                _edge(
                    EntityEdge,
                    fact=str(spec["neighbour_fact"]),
                    source=source_uuid,
                    target=target_uuid,
                    valid_at=times["candidates"],
                )
            ],
            [
                _edge(
                    EntityEdge,
                    fact=str(spec["graph_fact"]),
                    source=source_uuid,
                    target=str(uuid.uuid4()),
                    valid_at=times["candidates"],
                )
            ],
        )

    # The control: the same new edge, offered no candidate at all.
    extracted, _, _ = fresh()
    control_client = StubLLMClient({"duplicate_facts": [], "contradicted_facts": []})
    control = _triple(await resolve_extracted_edge(control_client, extracted, [], [], episode))

    results: list[dict[str, Any]] = []
    for duplicates in _powerset([0]):
        for contradictions in _powerset([0, 1]):
            extracted, related, existing = fresh()
            client = StubLLMClient({"duplicate_facts": duplicates, "contradicted_facts": contradictions})
            triple = _triple(await resolve_extracted_edge(client, extracted, related, existing, episode))
            results.append(
                {
                    "shape": name,
                    "response": {"duplicate_facts": duplicates, "contradicted_facts": contradictions},
                    "triple": triple,
                    "identical_to_control": triple == control,
                    "resolved_is_new_edge": triple["resolved"]["uuid"] == control["resolved"]["uuid"],
                    "n_duplicates": len(triple["duplicates"]),
                    "n_invalidated": len(triple["invalidated"]),
                    "resolved_expired": triple["resolved"].get("expired_at") is not None,
                    "neighbour_in_prompt": str(spec["neighbour_fact"]) in "\n".join(client.prompts),
                }
            )
    return {"control": control, "responses": results}


async def _run(spec: dict[str, Any]) -> dict[str, Any]:
    from graphiti_core.edges import EntityEdge
    from graphiti_core.prompts.dedupe_edges import EdgeDuplicate

    shapes: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    control: dict[str, Any] = {}
    for name, times in SHAPES.items():
        outcome = await _shape(spec, name, times)
        shapes[name] = outcome
        results.extend(outcome["responses"])
        control = outcome["control"]

    schema_probe = []
    for extra in spec.get("invented_fields", []):
        payload = {"duplicate_facts": [], "contradicted_facts": [], **extra}
        try:
            parsed = EdgeDuplicate(**payload).model_dump()
            schema_probe.append({"sent": payload, "accepted": True, "parsed": parsed, "survived": sorted(set(extra) & set(parsed))})
        except Exception as error:
            schema_probe.append({"sent": payload, "accepted": False, "error": f"{type(error).__name__}: {error}"})

    return {
        "control": control,
        "shapes": shapes,
        "responses": results,
        "response_model_fields": sorted(EdgeDuplicate.model_fields),
        "entity_edge_fields": sorted(EntityEdge.model_fields),
        "schema_probe": schema_probe,
    }


def main() -> int:
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(source.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {"schema": SCHEMA, "spec": spec, "python": sys.version.split()[0]}
    try:
        from importlib.metadata import version

        payload["components"] = {"graphiti-core": version("graphiti-core")}
        payload.update(asyncio.run(_run(spec)))
        payload["ok"] = True
    except Exception as error:
        payload["ok"] = False
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
