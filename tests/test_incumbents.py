"""Tests for the incumbent execution package.

These run without any incumbent installed. The probes themselves live in other
interpreters and cannot be imported here, so what is tested is everything this
project owns: how cases are built, how a dump is scored, how the rates
aggregate, and that a missing incumbent reports BLOCKED with the dependency
named rather than silently scoring zero.

The scoring tests use hand-built dumps rather than recorded ones on purpose. A
recorded dump would test that the code reproduces a past run; a hand-built one
tests that the definitions in the pre-registration are the definitions the code
implements.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from brain.baselines.incumbents import aggregate, arm, bridge, cases, outcomes, published, specs
from brain.corpus.relations.loader import load_relations

WORKERS = Path(bridge.WORKER_ROOT)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record(record_id: str, trace_id: str, **extra):
    payload = {"data": f"text of {trace_id}", "trace_id": trace_id, "created_at": "2026-01-01"}
    payload.update(extra)
    return {"id": record_id, "payload": payload}


def _state(*records, entities=()):
    return {"memories": list(records), "entities": list(entities), "history": {}}


def _case(case_id="pair-p0", traces=("t-a", "t-b")):
    return {
        "case_id": case_id,
        "pair_id": "p0",
        "regime": "long",
        "cluster_id": "t01",
        "writes": [{"trace_id": trace, "text": f"text of {trace}"} for trace in traces],
    }


# -- case construction -----------------------------------------------------


def test_case_plan_pairs_every_interference_pair_with_a_cross_cluster_control():
    corpus = load_relations()
    plan = cases.build_case_plan(corpus, limit=6)
    pairs = plan.pairs()
    assert pairs and len(plan.cases) == 2 * len(pairs)
    by_id = {episode.trace_id: episode for episode in corpus.episodes}
    for pair in pairs:
        control = by_id_lookup(plan, pair)
        first, second = (by_id[write["trace_id"]] for write in pair.writes)
        control_second = by_id[control.writes[1]["trace_id"]]
        assert control.writes[0] == pair.writes[0], "the control must share its first write"
        assert control_second.cluster_id not in {first.cluster_id, second.cluster_id}
        assert control_second.regime == pair.regime, "control must match the regime"
        assert pair.cls == cases.INTERFERENCE


def by_id_lookup(plan, pair):
    return plan.by_id()[plan.control_of[pair.case_id]]


def test_case_plan_is_deterministic():
    corpus = load_relations()
    first = cases.build_case_plan(corpus, limit=4)
    second = cases.build_case_plan(corpus, limit=4)
    assert first.to_dict() == second.to_dict()


# -- scoring ---------------------------------------------------------------


def test_both_kept_and_no_link_is_not_destroyed_and_not_recorded():
    state = _state(_record("m1", "t-a"), _record("m2", "t-b"))
    result = outcomes.score_case({"state": state}, {"state": state}, _case())
    assert not result.destroyed
    assert result.both_attributable
    assert not result.relation_recorded
    assert result.indistinguishable


def test_a_merge_that_drops_a_trace_id_counts_as_destruction():
    merged = _state(_record("m1", "t-b"))  # the older episode's attribution is gone
    control = _state(_record("m1", "t-a"), _record("m2", "t-c"))
    result = outcomes.score_case({"state": merged}, {"state": control}, _case())
    assert result.destroyed
    assert not result.both_attributable


def test_a_link_present_in_the_pair_and_absent_from_the_control_is_recorded():
    pair = _state(_record("m1", "t-a", linked_memory_ids=["m2"]), _record("m2", "t-b"))
    control = _state(_record("m1", "t-a"), _record("m2", "t-c"))
    result = outcomes.score_case({"state": pair}, {"state": control}, _case())
    assert result.relation_recorded


def test_a_link_the_control_also_gets_is_not_a_record_of_confusability():
    linked = lambda second: _state(  # noqa: E731
        _record("m1", "t-a"),
        _record("m2", second),
        entities=[{"id": "e1", "payload": {"data": "eval.go", "linked_memory_ids": ["m1", "m2"]}}],
    )
    result = outcomes.score_case({"state": linked("t-b")}, {"state": linked("t-c")}, _case())
    assert not result.relation_recorded, "a link both cases get records co-occurrence, not confusability"


def test_entity_store_links_are_read_as_memory_to_memory_edges():
    state = _state(
        _record("m1", "t-a"),
        _record("m2", "t-b"),
        entities=[{"id": "e1", "payload": {"linked_memory_ids": ["m1", "m2"]}}],
    )
    assert outcomes.link_edges(state) == {("m1", "m2")}


def test_signature_is_blind_to_content_but_not_to_structure():
    left = _state(_record("m1", "t-a"), _record("m2", "t-b"))
    right = _state(_record("x9", "t-q"), _record("z1", "t-r"))
    assert outcomes.signature(left) == outcomes.signature(right)
    extra = _state(_record("m1", "t-a", invalid_at="2026-02-01"), _record("m2", "t-b"))
    assert outcomes.signature(left) != outcomes.signature(extra)


def test_neighbour_offered_requires_the_second_write_to_have_seen_something():
    blind = {"judge": {"write": {"calls": [{"existing_ids_offered": []}, {"existing_ids_offered": []}]}}}
    seeing = {"judge": {"write": {"calls": [{"existing_ids_offered": []}, {"existing_ids_offered": ["0"]}]}}}
    state = _state(_record("m1", "t-a"), _record("m2", "t-b"))
    assert not outcomes.score_case({**blind, "state": state}, {"state": state}, _case()).neighbour_offered
    assert outcomes.score_case({**seeing, "state": state}, {"state": state}, _case()).neighbour_offered


# -- aggregation -----------------------------------------------------------


def test_summary_reports_every_rate_with_a_cluster_interval():
    scored = [
        outcomes.score_case(
            {"state": _state(_record("m1", "t-a"), _record("m2", "t-b"))},
            {"state": _state(_record("m1", "t-a"), _record("m2", "t-c"))},
            {**_case(f"pair-p{index}"), "cluster_id": f"t0{index % 3}"},
        )
        for index in range(9)
    ]
    summary = aggregate.summarise(scored, label="unit")
    assert summary["n_pairs"] == 9
    for name in aggregate.RATES:
        assert 0.0 <= summary[name]["pooled"]["rate"] <= 1.0
        assert summary[name]["pooled"]["interval"]["method"].startswith("cluster-bootstrap")
    assert summary["destroyed"]["pooled"]["rate"] == 0.0


# -- blocked reporting -----------------------------------------------------


def test_a_missing_incumbent_names_the_environment_variable(monkeypatch):
    monkeypatch.delenv("BRAIN_INCUMBENT_VENV_MEM0", raising=False)
    with pytest.raises(bridge.VenvMissing) as caught:
        bridge.venv_for("mem0")
    assert "BRAIN_INCUMBENT_VENV_MEM0" in str(caught.value)


def test_the_arm_reports_blocked_rather_than_zero(monkeypatch):
    monkeypatch.delenv("BRAIN_INCUMBENT_VENV_MEM0", raising=False)
    assert arm.Mem0Retriever().blocked_reason is not None
    assert not arm.Mem0Retriever().is_available()


def test_every_worker_script_exists():
    for name, path in bridge.WORKERS.items():
        assert path.is_file(), f"{name} worker missing at {path}"


# -- the record itself -----------------------------------------------------


def test_requirements_name_a_blocker_or_state_there_is_none():
    survey = specs.survey()
    systems = {item["system"] for item in survey["requirements"]}
    assert {"mem0", "graphiti", "toki", "nous"} <= systems
    for item in survey["requirements"]:
        assert item["version_inspected"], item["system"]
        if item["network_required"]:
            assert item["blocker"], f"{item['system']} needs network but names no blocker"
    assert survey["credentials_present"]["OPENAI_API_KEY"] is False


def test_published_numbers_are_all_labelled_as_not_ours():
    rows = published.table()["rows"]
    assert rows
    for row in rows:
        assert row["measured_by_us"] is False
        assert row["source"], row
        assert any(token in row["source"] for token in ("arXiv", "Table", "section"))


# -- the judges, which decide what a probe measures -------------------------


#: The shape Mem0 2.x actually builds. The worked examples inside the real
#: system prompt also contain `"id": "0"`, which is why the heading is the
#: anchor and a bare regex would be wrong.
ADDITIVE_PROMPT = (
    '## Existing Memories\n[{"id": "0", "text": "the neighbour"}]\n'
    "## New Messages\n[]\n## Observation Date\n2026-01-01\n"
)
#: The shape Mem0 0.1.x actually builds: a PYTHON repr in a fenced block, with
#: the retrieved facts fenced after it.
LEGACY_PROMPT = (
    "Below is the current content of my memory\n\n```\n"
    "[{'id': '0', 'text': 'an old memory'}]\n```\n\n"
    "The new retrieved facts are mentioned in the triple backticks.\n\n```\n['a new memory']\n```\n"
)


def test_the_oracle_judge_asserts_a_relation_and_never_merges():
    judges = _load("incumbent_judges", WORKERS / "mem0" / "judges.py")
    judge = judges.AdditiveJudge(oracle=True)
    judge.current = {"text": "an episode"}
    judge.store_uuids = ["uuid-a"]
    import json

    payload = json.loads(judge.respond([{"role": "user", "content": ADDITIVE_PROMPT}]))
    record = payload["memory"][0]
    assert record["text"] == "an episode"
    for name in judges.ORACLE_RELATION_FIELDS:
        assert name in record, f"the oracle must assert {name}"
    assert "uuid-a" in record["linked_memory_ids"]
    assert judge.calls[-1].existing_ids == ["0"], "the neighbour must be seen as offered"


def test_the_identity_judge_claims_no_relation():
    judges = _load("incumbent_judges", WORKERS / "mem0" / "judges.py")
    judge = judges.AdditiveJudge(oracle=False)
    judge.current = {"text": "an episode"}
    import json

    record = json.loads(judge.respond([{"role": "user", "content": ADDITIVE_PROMPT}]))["memory"][0]
    assert not set(judges.ORACLE_RELATION_FIELDS) & set(record)


def test_an_empty_existing_section_offers_nothing():
    judges = _load("incumbent_judges", WORKERS / "mem0" / "judges.py")
    judge = judges.AdditiveJudge(oracle=False)
    judge.current = {"text": "an episode"}
    judge.respond([{"role": "user", "content": "## Existing Memories\n[]\n## New Messages\n[]\n"}])
    assert judge.calls[-1].existing_ids == []


def test_the_legacy_prompt_is_parsed_as_a_python_repr_not_as_json():
    judges = _load("incumbent_judges", WORKERS / "mem0" / "judges.py")
    offered = judges._parse_old_memory(LEGACY_PROMPT)
    assert offered == [{"id": "0", "text": "an old memory"}]


def test_the_tau_judge_merges_only_above_the_threshold():
    judges = _load("incumbent_judges", WORKERS / "mem0" / "judges.py")
    import json

    high = judges.LegacyUpdateJudge("tau085", similarity=lambda a, b: 0.9)
    low = judges.LegacyUpdateJudge("tau085", similarity=lambda a, b: 0.2)
    for judge in (high, low):
        judge.current = {"text": "a new memory"}
    message = [{"role": "user", "content": LEGACY_PROMPT}]
    assert json.loads(high.respond(message))["memory"][0]["event"] == "UPDATE"
    assert json.loads(low.respond(message))["memory"][0]["event"] == "ADD"
