"""Run every incumbent probe that can run, and record why the rest cannot.

    python -m brain.baselines.incumbents

One configuration is one (system, judge) pair. Each writes its own JSON blob
into the combined result file, carrying the resolved package versions, the exact
command, the substitutions made, and the scored outcomes. A configuration whose
interpreter is absent is recorded as BLOCKED with the missing dependency named,
never as a zero and never as an estimate.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from brain.config import results_dir
from brain.corpus.relations.loader import assert_synthetic_corpus_unchanged, load_relations
from brain.eval.results import environment, jsonable

from brain.baselines.incumbents import aggregate, bridge, cases, outcomes, published, specs

__all__ = ["CONFIGURATIONS", "run_configuration", "run_all", "main"]

#: (name, worker, venv key, worker spec). Frozen in the pre-registration.
CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {"name": "mem0-2.0.15-identity", "worker": "mem0", "venv": "mem0", "spec": {"family": "additive", "judge": "identity"}},
    {"name": "mem0-2.0.15-oracle", "worker": "mem0", "venv": "mem0", "spec": {"family": "additive", "judge": "oracle"}},
    {"name": "mem0-0.1.118-identity", "worker": "mem0", "venv": "mem0_legacy", "spec": {"family": "legacy", "judge": "identity"}},
    {"name": "mem0-0.1.118-oracle", "worker": "mem0", "venv": "mem0_legacy", "spec": {"family": "legacy", "judge": "oracle"}},
    {"name": "mem0-0.1.118-tau085", "worker": "mem0", "venv": "mem0_legacy", "spec": {"family": "legacy", "judge": "tau085"}},
    {"name": "nous-0.2.0-asshipped", "worker": "nous", "venv": "nous", "spec": {}},
)

#: The Graphiti probe is not a per-pair run: its response space is finite, so it
#: is executed exhaustively instead of sampled. Kept separate for that reason.
GRAPHITI_SPEC = {
    "new_fact": "platform/featureflags/eval.go returned a nil pointer on PR #9166",
    "neighbour_fact": "platform/featureflags/eval.go returned a nil pointer on PR #9204",
    "graph_fact": "platform/featureflags/eval.go was refactored to use a pointer receiver",
    "invented_fields": [
        {"sibling_facts": [0]},
        {"keep_both": True},
        {"distinct_facts": [0], "relation": "INTERFERENCE"},
    ],
}


def run_configuration(
    configuration: dict[str, Any], plan: cases.CasePlan, workroot: Path
) -> dict[str, Any]:
    """One system under one judge, scored, or BLOCKED with the reason."""
    name = str(configuration["name"])
    record: dict[str, Any] = {"configuration": name, "spec": dict(configuration["spec"])}
    spec = dict(configuration["spec"])
    spec["cases"] = [case.to_dict() for case in plan.cases]
    spec["workdir"] = str(workroot / name / "stores")
    try:
        payload, venv, invocation = bridge.run_worker(
            configuration["worker"], configuration["venv"], spec, workdir=workroot / name
        )
    except (bridge.VenvMissing, bridge.WorkerFailed, Exception) as error:
        record["status"] = "BLOCKED"
        record["blocked_reason"] = f"{type(error).__name__}: {error}"
        return record

    by_case = {str(item["case_id"]): item for item in payload.get("cases", [])}
    scored = []
    for case in plan.pairs():
        control_id = plan.control_of[case.case_id]
        if case.case_id not in by_case or control_id not in by_case:
            continue
        scored.append(outcomes.score_case(by_case[case.case_id], by_case[control_id], case.to_dict()))

    record.update(
        {
            "status": "RAN",
            "venv": venv.to_dict(),
            "invocation": invocation,
            "components": payload.get("components", {}),
            "summary": aggregate.summarise(scored, label=name),
            "per_case": [item.to_dict() for item in scored],
        }
    )
    return record


def run_graphiti(workroot: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"configuration": "graphiti-0.29.2-exhaustive"}
    try:
        payload, venv, invocation = bridge.run_worker(
            "graphiti", "graphiti", GRAPHITI_SPEC, workdir=workroot / "graphiti"
        )
    except Exception as error:
        record["status"] = "BLOCKED"
        record["blocked_reason"] = f"{type(error).__name__}: {error}"
        return record
    responses = payload.get("responses", [])
    keep_both = [item for item in responses if not item["response"]["duplicate_facts"] and not item["response"]["contradicted_facts"]]
    record.update(
        {
            "status": "RAN",
            "venv": venv.to_dict(),
            "invocation": invocation,
            "components": payload.get("components", {}),
            "n_responses_executed": len(responses),
            "keep_both_identical_to_control": all(item["identical_to_control"] for item in keep_both),
            "n_identical_to_control": sum(1 for item in responses if item["identical_to_control"]),
            "response_model_fields": payload.get("response_model_fields", []),
            "entity_edge_fields": payload.get("entity_edge_fields", []),
            "invented_fields_surviving": [
                probe.get("survived") for probe in payload.get("schema_probe", [])
            ],
            "responses": responses,
            "schema_probe": payload.get("schema_probe", []),
        }
    )
    return record


#: Mem0 0.1.x dispatches its write on a string `event`, so the branch set is
#: finite and is executed exhaustively. The last two are invented: what the
#: dispatcher does with an operation it does not know is the claim under test.
MEM0_BRANCH_EVENTS = ("ADD", "UPDATE", "DELETE", "NONE", "LINK", "SIBLING")


def run_mem0_branches(plan: cases.CasePlan, workroot: Path) -> dict[str, Any]:
    """Every value of Mem0 0.1.x's write `event`, on one INTERFERENCE pair."""
    record: dict[str, Any] = {"configuration": "mem0-0.1.118-branch-enumeration"}
    pairs = plan.pairs()
    if not pairs:
        record["status"] = "BLOCKED"
        record["blocked_reason"] = "no INTERFERENCE pair available"
        return record
    spec = {
        "family": "legacy",
        "judge": "identity",
        "branch_case": pairs[0].to_dict(),
        "branch_events": list(MEM0_BRANCH_EVENTS),
        "workdir": str(workroot / "mem0-branches" / "stores"),
    }
    try:
        payload, venv, invocation = bridge.run_worker(
            "mem0", "mem0_legacy", spec, workdir=workroot / "mem0-branches"
        )
    except Exception as error:
        record["status"] = "BLOCKED"
        record["blocked_reason"] = f"{type(error).__name__}: {error}"
        return record

    traces = [write["trace_id"] for write in pairs[0].to_dict()["writes"]]
    branches = []
    for item in payload.get("branches", []):
        state = item.get("state", {})
        resident = {trace: outcomes.attributable(state, trace) for trace in traces}
        branches.append(
            {
                "event": item["event"],
                "n_memories": len(state.get("memories", [])),
                "both_attributable": all(resident[trace] for trace in traces),
                "resident": resident,
                "link_edges": sorted(outcomes.link_edges(state)),
                "signature": outcomes.signature(state),
            }
        )
    known = {item["event"] for item in branches if item["event"] in ("ADD", "UPDATE", "DELETE", "NONE")}
    invented = [item for item in branches if item["event"] not in known]
    # "No-op" means NONE, the branch that changes nothing. Comparing against ADD
    # would ask a different question, whether the write still landed.
    none_branch = next((item for item in branches if item["event"] == "NONE"), None)
    record.update(
        {
            "status": "RAN",
            "venv": venv.to_dict(),
            "invocation": invocation,
            "components": payload.get("components", {}),
            "pair_id": pairs[0].pair_id,
            "branches": branches,
            "events_executed": list(MEM0_BRANCH_EVENTS),
            "invented_events_are_a_silent_no_op": bool(none_branch)
            and all(item["signature"] == none_branch["signature"] for item in invented),
            "invented_events": [item["event"] for item in invented],
            "no_event_records_a_link": all(not item["link_edges"] for item in branches),
        }
    )
    return record


def run_toki(workroot: Path) -> dict[str, Any]:
    """TOKI's resolution algebra, executed over every operator and arrangement.

    Not a per-pair run either. TOKI's contradiction test is a closed-form
    predicate over an exact key, so what a pair of episodes does to the store is
    decided by which of three arrangements it falls into, and all three are
    executed under all four operators.
    """
    record: dict[str, Any] = {"configuration": "toki-exhaustive"}
    try:
        payload, venv, invocation = bridge.run_worker("toki", "toki", {}, workdir=workroot / "toki")
    except Exception as error:
        record["status"] = "BLOCKED"
        record["blocked_reason"] = f"{type(error).__name__}: {error}"
        return record
    results = payload.get("results", [])
    interference = [item for item in results if item.get("arrangement") == "same_key_overlapping"]
    kept = [item for item in results if item.get("arrangement") == "same_key_disjoint"]
    unrelated = {item["operator"]: item for item in results if item.get("arrangement") == "different_key"}

    def signature(item: dict[str, Any]) -> tuple:
        return (
            item.get("n_current_rows"),
            item.get("n_audit_rows"),
            tuple(sorted(tuple(sorted(row)) for row in item.get("rows", []))),
        )

    record.update(
        {
            "status": "RAN",
            "venv": venv.to_dict(),
            "invocation": invocation,
            "components": payload.get("components", {}),
            "contradiction_predicate": payload.get("contradiction_predicate"),
            "n_executions": len(results),
            "operators": payload.get("operators", []),
            "destroyed_under_every_operator": all(
                not item.get("both_facts_open") for item in interference
            ),
            "audit_row_written_on_every_resolution": all(
                int(item.get("n_audit_rows", 0)) >= 1 for item in interference
            ),
            "keep_both_indistinguishable_from_unrelated": all(
                signature(item) == signature(unrelated[item["operator"]]) for item in kept
            ),
            "results": results,
        }
    )
    return record


def run_all(*, workroot: Path, limit: int | None = None) -> dict[str, Any]:
    """Every configuration, plus the requirement survey and published numbers."""
    synthetic_sha = assert_synthetic_corpus_unchanged()
    corpus = load_relations()
    plan = cases.build_case_plan(corpus, limit=limit)

    results = [run_configuration(configuration, plan, workroot) for configuration in CONFIGURATIONS]
    results.append(run_graphiti(workroot))
    results.append(run_toki(workroot))
    results.append(run_mem0_branches(plan, workroot))

    return {
        "schema": "incumbent-execution-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "corpus": {
            "synthetic_content_sha256": synthetic_sha,
            "relations_content_sha256": corpus.manifest.get("content_sha256"),
            "n_interference_pairs": len(plan.pairs()),
            "case_plan": plan.to_dict(),
        },
        "requirements": specs.survey(),
        "configurations": results,
        "published_numbers": published.table(),
        "environment": environment(),
        "notes": {
            "substitution": (
                "Mem0 has no credential-free LLM in process, so its write path is driven by a "
                "declared judge. Nous runs with its own shipped components and nothing "
                "substituted. Graphiti's branch set is enumerated exhaustively, which removes "
                "the model from the argument entirely."
            ),
            "published_numbers_are_not_ours": (
                "Every entry under published_numbers was read from a paper. None of it was "
                "produced on this machine and none of it is comparable to the measured rows."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the incumbent write-path probes")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of pairs (smoke)")
    parser.add_argument("--workroot", default=os.environ.get("BRAIN_INCUMBENT_WORK", "/tmp/brain-incumbents"))
    parser.add_argument("--results-dir", default=str(results_dir("incumbents")))
    arguments = parser.parse_args(argv)

    payload = run_all(workroot=Path(arguments.workroot), limit=arguments.limit)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(arguments.results_dir) / f"incumbents-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    for record in payload["configurations"]:
        print(f"  {record['configuration']:28s} {record['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
