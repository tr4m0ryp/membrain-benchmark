"""Drive Mem0's real write path and dump what it persisted.

Runs INSIDE the Mem0 virtual environment, never inside this project's, so the
project's pinned stack is untouched. It imports nothing from `brain`: the
contract is one JSON file in, one JSON file out.

    python probe.py <input.json> <output.json>

Two tasks:

* **cases** -- write a sequence of episodes into a fresh store and dump the
  result. The scoring (destruction, relation recording, structural
  indistinguishability) happens outside, on the dump, so that no judgement is
  made by the code that produced the state.
* **branches** -- enumerate the write-decision space itself. Mem0's v0.1 write
  dispatches on a string `event`, so the branch set is finite and can be
  executed exhaustively rather than sampled. Invented events are included: what
  the dispatcher does with an operation it does not know is the whole question.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build  # noqa: E402
import dump as dump_mod  # noqa: E402
import judges as judges_mod  # noqa: E402

SCHEMA = "mem0-probe-v1"
USER_ID = "brain-probe"


#: One model for the whole process. A judge is built per case, so without this
#: the threshold judge would reload the encoder 120 times and the run would
#: measure model loading.
_SIMILARITY: dict[str, Any] = {}


def _similarity_fn(model_name: str):
    if model_name in _SIMILARITY:
        return _SIMILARITY[model_name]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def similarity(left: str, right: str) -> float:
        import numpy as np

        vectors = model.encode([left, right], normalize_embeddings=True)
        return float(np.dot(vectors[0], vectors[1]))

    _SIMILARITY[model_name] = similarity
    return similarity


def make_judges(spec: dict[str, Any]) -> dict[str, Any]:
    """The judge (or judge pair) for one family, per the pre-registration."""
    family = str(spec.get("family", "additive"))
    mode = str(spec.get("judge", "identity"))
    if family == "additive":
        if mode == "tau085":
            raise SystemExit(
                "tau085 is inapplicable to the additive family: its response schema "
                "has no merge or invalidate operation to emit"
            )
        return {"write": judges_mod.AdditiveJudge(oracle=mode == "oracle")}
    similarity = _similarity_fn(spec.get("embed_model", build.EMBED_MODEL)) if mode == "tau085" else None
    return {
        "extract": judges_mod.LegacyExtractionJudge(),
        "write": judges_mod.LegacyUpdateJudge(mode, similarity=similarity, tau=float(spec.get("tau", 0.85))),
    }


class Router:
    """Sends Mem0's first call to the extractor and the rest to the writer.

    Mem0 v0.1 makes exactly two LLM calls per `add`: extract facts, then decide
    what to do with them. v2 makes one. Routing by call order rather than by
    sniffing the prompt keeps this independent of prompt wording.
    """

    def __init__(self, extract: Any, write: Any) -> None:
        self._extract = extract
        self._write = write
        self._calls = 0
        self.current: dict[str, Any] = {}
        self.store_uuids: list[str] = []

    def begin(self, record: dict[str, Any], store_uuids: list[str]) -> None:
        self._calls = 0
        for judge in (self._extract, self._write):
            if judge is not None:
                judge.current = dict(record)
                judge.store_uuids = list(store_uuids)

    def respond(self, messages: list[dict[str, Any]], response_format: Any = None) -> str:
        self._calls += 1
        judge = self._extract if (self._extract is not None and self._calls == 1) else self._write
        return judge.respond(messages, response_format)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"write": self._write.to_dict()}
        if self._extract is not None:
            out["extract"] = self._extract.to_dict()
        return out


def _resident_uuids(memory: Any) -> list[str]:
    try:
        return [row["id"] for row in dump_mod.dump_store(memory.vector_store, {"user_id": USER_ID})]
    except Exception:
        return []


def run_case(case: dict[str, Any], spec: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """One fresh store, one ordered sequence of writes, one dump."""
    parts = make_judges(spec)
    router = Router(parts.get("extract"), parts["write"])
    store_dir = workdir / str(case["case_id"])
    memory = build.build_memory(store_dir, router, embed_model=spec.get("embed_model", build.EMBED_MODEL))
    add_results: list[Any] = []
    for record in case["writes"]:
        router.begin(record, _resident_uuids(memory))
        payload = memory.add(
            [{"role": "user", "content": record["text"]}],
            user_id=USER_ID,
            metadata={"trace_id": record["trace_id"]},
            infer=True,
        )
        add_results.append(payload)
    state = dump_mod.dump_memory(
        memory, user_id=USER_ID, history_db_path=build.memory_config(store_dir)["history_db_path"]
    )
    return {
        "case_id": case["case_id"],
        "writes": case["writes"],
        "add_results": add_results,
        "state": state,
        "judge": router.to_dict(),
    }


def run_branches(spec: dict[str, Any], workdir: Path) -> list[dict[str, Any]]:
    """Execute every value of Mem0 v0.1's `event` field, invented ones included."""
    case = spec["branch_case"]
    out: list[dict[str, Any]] = []
    for event in spec.get("branch_events", []):
        judge = judges_mod.LegacyUpdateJudge("identity")
        forced = _ForcedEventJudge(judge, event, bool(spec.get("branch_relation_fields", True)))
        router = Router(judges_mod.LegacyExtractionJudge(), forced)
        store_dir = workdir / f"branch-{event}"
        memory = build.build_memory(store_dir, router, embed_model=spec.get("embed_model", build.EMBED_MODEL))
        results = []
        for record in case["writes"]:
            router.begin(record, _resident_uuids(memory))
            results.append(
                memory.add(
                    [{"role": "user", "content": record["text"]}],
                    user_id=USER_ID,
                    metadata={"trace_id": record["trace_id"]},
                    infer=True,
                )
            )
        out.append(
            {
                "event": event,
                "add_results": results,
                "state": dump_mod.dump_memory(
                    memory,
                    user_id=USER_ID,
                    history_db_path=build.memory_config(store_dir)["history_db_path"],
                ),
                "judge": router.to_dict(),
            }
        )
    return out


def run_arm(spec: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Index the frozen corpus into one Mem0 store and answer every query.

    Ranking is Mem0's own: the order `Memory.search` returns is preserved
    verbatim and turned into a score by position, so nothing about how this
    project ranks leaks into the arm. Only the mapping back from a memory to the
    episode it came from is ours, and it is exact, because the episode's
    trace_id is written into the metadata of every memory it produces.
    """
    parts = make_judges(spec)
    router = Router(parts.get("extract"), parts["write"])
    store_dir = workdir / "arm"
    memory = build.build_memory(store_dir, router, embed_model=spec.get("embed_model", build.EMBED_MODEL))
    for record in spec["episodes"]:
        router.begin(record, [])
        memory.add(
            [{"role": "user", "content": record["text"]}],
            user_id=USER_ID,
            metadata={"trace_id": record["trace_id"]},
            infer=True,
        )
    state = dump_mod.dump_memory(
        memory, user_id=USER_ID, history_db_path=build.memory_config(store_dir)["history_db_path"]
    )
    k = int(spec.get("k", 5))
    answers: dict[str, list[str]] = {}
    for query in spec["queries"]:
        # 2.x wants the scope in `filters` and rejects it as a keyword; 0.1.x
        # only accepts the keyword. Try the newer form first, fall back once.
        try:
            found = memory.search(query["text"], filters={"user_id": USER_ID}, limit=max(k * 4, 20))
        except (TypeError, ValueError):
            found = memory.search(query["text"], user_id=USER_ID, limit=max(k * 4, 20))
        rows = found.get("results", found) if isinstance(found, dict) else found
        ordered: list[str] = []
        for row in rows:
            payload = row if isinstance(row, dict) else {}
            trace = str((payload.get("metadata") or {}).get("trace_id") or payload.get("trace_id") or "")
            if trace and trace not in ordered:
                ordered.append(trace)
        answers[str(query["query_id"])] = ordered[:k]
    return {
        "n_indexed_episodes": len(spec["episodes"]),
        "n_memories": len(state.get("memories", [])),
        "n_queries": len(answers),
        "answers": answers,
        "state_summary": {
            "memories": len(state.get("memories", [])),
            "entities": len(state.get("entities", [])),
            "distinct_traces": len(
                {
                    str((record.get("payload") or {}).get("trace_id"))
                    for record in state.get("memories", [])
                    if (record.get("payload") or {}).get("trace_id")
                }
            ),
        },
    }


class _ForcedEventJudge(judges_mod.Judge):
    """Emit a fixed `event` for the second write, whatever it is."""

    def __init__(self, inner: Any, event: str, relation_fields: bool) -> None:
        super().__init__()
        self.name = f"forced-{event}"
        self._inner = inner
        self._event = str(event)
        self._relation_fields = bool(relation_fields)

    def respond(self, messages: list[dict[str, Any]], response_format: Any = None) -> str:
        prompt = self._user_prompt(messages)
        offered = judges_mod._parse_old_memory(prompt)
        text = str(self.current.get("text", ""))
        if not offered:  # the first write has nothing to decide against
            record: dict[str, Any] = {"id": "0", "text": text, "event": "ADD"}
        else:
            record = {"id": offered[0]["id"], "text": text, "event": self._event}
            record["old_memory"] = offered[0].get("text", "")
            if self._relation_fields:
                record["linked_memory_ids"] = [item["id"] for item in offered]
                record["relation"] = "INTERFERENCE_SIBLING"
                record["keep_both"] = True
        return self._record(prompt, json.dumps({"memory": [record]}), [i["id"] for i in offered])


def main() -> int:
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(source.read_text(encoding="utf-8"))
    workdir = Path(spec.get("workdir") or (destination.parent / "work"))
    workdir.mkdir(parents=True, exist_ok=True)
    build.apply_offline_env()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "spec": {key: value for key, value in spec.items() if key not in ("cases", "branch_case")},
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        payload["components"] = build.component_report()
        payload["cases"] = [run_case(case, spec, workdir) for case in spec.get("cases", [])]
        if spec.get("branch_events"):
            payload["branches"] = run_branches(spec, workdir)
        if spec.get("episodes") and spec.get("queries"):
            payload["arm"] = run_arm(spec, workdir)
        payload["ok"] = True
    except Exception as error:  # a failed probe reports why, it does not vanish
        payload["ok"] = False
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    code = main()
    # Every Mem0 instance leaves faiss and OpenMP thread pools behind and never
    # releases them, so a run over 120 stores reaches several hundred live
    # threads and normal interpreter shutdown spends longer joining them than
    # the probe spent measuring. The output file is written and closed by the
    # time we get here, so exiting without teardown loses nothing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
