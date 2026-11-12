"""Mem0 as a retrieval arm, scored by the same harness as every other arm.

`brain.eval.runner.run_arm` talks to an arm through `index`, `prefetch` and
`retrieve`, so Mem0 can be scored on disambiguation exactly as `plain_vector`
and `bm25` are, with the same cluster bootstrap and the same k. What differs is
where the work happens: Mem0 lives in another interpreter, so the whole corpus
and the whole query set cross the boundary once, in `prefetch`, and every
`retrieve` is served from the answers that came back. That is the same shape the
LLM arm already uses, and for the same reason.

What this arm measures, stated precisely so no reader can mistake it: Mem0's
storage and retrieval layer, on our corpus, with its LLM replaced by the
`identity` judge (one memory per episode, text unchanged). It is NOT Mem0 as
shipped, because Mem0 as shipped needs a paid key to write at all. The
substitution removes Mem0's fact rewriting, which by F23's finding is the part
that costs accuracy, so this arm is the FAVOURABLE reading of Mem0 rather than
a handicapped one.

`blocked_reason` is not None when no Mem0 interpreter is configured, so the
harness reports BLOCKED rather than zero.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from brain.baselines.base import BaseRetriever, rank_hits
from brain.baselines.incumbents import bridge
from brain.store import Trace

__all__ = ["Mem0Retriever"]


class Mem0Retriever(BaseRetriever):
    """Mem0's write and search path, driven in its own interpreter."""

    name = "mem0"

    def __init__(
        self,
        *,
        venv: str = "mem0",
        judge: str = "identity",
        family: str = "additive",
        workroot: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            name=name or f"mem0-{judge}",
            params={
                "venv": venv,
                "judge": judge,
                "family": family,
                "llm": "substituted; see brain.baselines.incumbents.workers.mem0.judges",
                "embedder": "sentence-transformers/all-MiniLM-L6-v2, as in every other arm",
                "ranking": "Mem0's own search order, preserved by position",
            },
        )
        self._venv = str(venv)
        self._judge = str(judge)
        self._family = str(family)
        self._workroot = Path(workroot or os.environ.get("BRAIN_INCUMBENT_WORK", "/tmp/brain-incumbents"))
        self._answers: dict[str, list[str]] = {}
        self._payload: dict[str, Any] = {}

    @property
    def blocked_reason(self) -> str | None:
        try:
            bridge.venv_for(self._venv)
        except Exception as error:
            return str(error)
        return None

    def _index(self, traces: Sequence[Trace]) -> None:
        # Nothing crosses the boundary yet: the queries are needed too, and they
        # only arrive at prefetch.
        self.stats["n_episodes"] = len(traces)

    def prefetch(self, queries: Sequence[str], k: int, **kwargs: Any) -> int:
        """Index the corpus and answer every query in one subprocess call."""
        spec: dict[str, Any] = {
            "family": self._family,
            "judge": self._judge,
            "k": int(k),
            "workdir": str(self._workroot / self.name / "stores"),
            "episodes": [{"trace_id": trace.id, "text": trace.text} for trace in self.traces],
            "queries": [
                {"query_id": str(index), "text": str(text)} for index, text in enumerate(queries)
            ],
        }
        payload, venv, invocation = bridge.run_worker(
            "mem0", self._venv, spec, workdir=self._workroot / self.name
        )
        arm = payload.get("arm", {})
        self._answers = {
            spec["queries"][int(query_id)]["text"]: list(found)
            for query_id, found in arm.get("answers", {}).items()
        }
        self._payload = {
            "venv": venv.to_dict(),
            "invocation": invocation,
            "components": payload.get("components", {}),
            "state_summary": arm.get("state_summary", {}),
        }
        self.stats.update(
            {
                "n_memories": arm.get("n_memories"),
                "distinct_traces_stored": (arm.get("state_summary") or {}).get("distinct_traces"),
                "n_answered": len(self._answers),
            }
        )
        return len(self._answers)

    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        found = self._answers.get(query)
        if found is None:
            # Prefetch is how this arm answers. Failing loudly is correct: a
            # silent empty result would read as a wrong answer rather than as a
            # missing run.
            raise bridge.WorkerFailed(
                f"{self.name}: query was not prefetched, so Mem0 never saw it: {query[:60]!r}"
            )
        # Mem0's own order, turned into a score by position so the shared
        # tie-break never reorders it.
        return rank_hits(
            ((trace_id, float(-position)) for position, trace_id in enumerate(found)), k
        )

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record["incumbent"] = dict(self._payload)
        return record
