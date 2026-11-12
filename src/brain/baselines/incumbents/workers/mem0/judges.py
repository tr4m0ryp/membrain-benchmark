"""The judges that stand in for the LLM inside Mem0's write path.

No API key exists on this machine, so the model Mem0 would call is substituted.
The substitution is the point rather than a workaround: a judge we control lets
the architecture be measured independently of model quality, and the `oracle`
judge below is deliberately the most preserving legal judge Mem0's own schema
admits. A failure under it is a property of the write path, not of a model.

Three judges, fixed before any run (see the pre-registration):

* ``identity``  -- one memory per episode, text unchanged, no relation claimed.
* ``oracle``    -- the same, plus every relation-bearing field Mem0's own prompt
  or response schema defines, plus invented ones. It never emits a merge and
  never emits an invalidate. It is trying as hard as the schema allows to say
  "these two are distinct, keep both, and record that they are siblings".
* ``tau085``    -- no ground truth: cosine similarity against the memories the
  prompt actually offered, merging above 0.85. F17's threshold, unchanged.

Every call records the prompt it saw, because "the neighbour was in context" is
a validity condition for the whole probe: a keep-both decision that was never
offered the neighbour would say nothing about the write path.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

#: Fields the oracle asserts. The first is Mem0's own (its additive prompt
#: documents `linked_memory_ids`); the rest are invented, to test whether ANY
#: relation-bearing field survives the parser rather than only that one.
ORACLE_RELATION_FIELDS = (
    "linked_memory_ids",
    "sibling_memory_ids",
    "relation",
    "keep_both",
    "distinct_event",
)

#: The v2 additive prompt puts the resident memories under this heading, as a
#: JSON list. Anchoring on the heading matters: the prompt's worked examples
#: also contain `"id": "0"`, so a bare regex would report a neighbour that was
#: never offered and would quietly invalidate the whole probe.
_EXISTING_SECTION = re.compile(r"##\s*Existing Memories\s*\n(.*?)(?:\n##\s|\Z)", re.DOTALL)
#: The v0.1 update prompt fences the resident memories in a code block, as a
#: PYTHON repr rather than JSON, so it is read with `ast.literal_eval`.
_FENCED = re.compile(r"```\s*(\[.*?\])\s*```", re.DOTALL)


class JudgeCall:
    """One recorded LLM call: what was asked, what was answered."""

    __slots__ = ("prompt", "response", "existing_ids")

    def __init__(self, prompt: str, response: str, existing_ids: list[str]) -> None:
        self.prompt = prompt
        self.response = response
        self.existing_ids = existing_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "existing_ids_offered": list(self.existing_ids),
        }


class Judge:
    """Base judge. The driver sets `current` before each `Memory.add` call."""

    name = "base"

    def __init__(self) -> None:
        self.current: dict[str, Any] = {}
        self.store_uuids: list[str] = []
        self.calls: list[JudgeCall] = []

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _user_prompt(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return "" if not messages else str(messages[-1].get("content", ""))

    @staticmethod
    def _offered(prompt: str) -> list[dict[str, str]]:
        """The existing memories Mem0 actually showed the model, id and text."""
        match = _EXISTING_SECTION.search(prompt)
        if match is None:
            return []
        try:
            rows = json.loads(match.group(1).strip())
        except (ValueError, TypeError):
            return []
        return [
            {"id": str(row["id"]), "text": str(row.get("text", ""))}
            for row in rows
            if isinstance(row, dict) and "id" in row
        ]

    @classmethod
    def _offered_ids(cls, prompt: str) -> list[str]:
        """The ids Mem0 showed the model for the existing memories."""
        return [item["id"] for item in cls._offered(prompt)]

    def _record(self, prompt: str, response: str, offered: list[str]) -> str:
        self.calls.append(JudgeCall(prompt, response, offered))
        return response

    # -- the interface the stub LLM calls ---------------------------------

    def respond(self, messages: list[dict[str, Any]], response_format: Any = None) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"judge": self.name, "calls": [call.to_dict() for call in self.calls]}


class AdditiveJudge(Judge):
    """For the v2 additive schema: `{"memory": [{"id", "text", ...}]}`.

    The additive schema has exactly one operation, so `identity` and `oracle`
    differ only in the relation fields the oracle attaches.
    """

    def __init__(self, *, oracle: bool) -> None:
        super().__init__()
        self.name = "oracle" if oracle else "identity"
        self._oracle = bool(oracle)

    def respond(self, messages: list[dict[str, Any]], response_format: Any = None) -> str:
        prompt = self._user_prompt(messages)
        offered = self._offered_ids(prompt)
        record: dict[str, Any] = {
            "id": "0",
            "text": str(self.current.get("text", "")),
            "attributed_to": "user",
        }
        if self._oracle:
            # Every id the prompt offered, plus every uuid actually resident, so
            # the claim cannot fail for want of naming the neighbour correctly.
            linked = offered + list(self.store_uuids)
            record["linked_memory_ids"] = linked
            record["sibling_memory_ids"] = linked
            record["relation"] = "INTERFERENCE_SIBLING"
            record["keep_both"] = True
            record["distinct_event"] = True
        return self._record(prompt, json.dumps({"memory": [record]}), offered)


class LegacyExtractionJudge(Judge):
    """Phase 1 of the v0.1 two-call write: fact extraction, `{"facts": [...]}`."""

    name = "legacy-extract"

    def respond(self, messages: list[dict[str, Any]], response_format: Any = None) -> str:
        prompt = self._user_prompt(messages)
        payload = json.dumps({"facts": [str(self.current.get("text", ""))]})
        return self._record(prompt, payload, [])


class LegacyUpdateJudge(Judge):
    """Phase 2 of the v0.1 write: the ADD / UPDATE / DELETE / NONE decision.

    This is the branch set the paper's claim is about, and it is enumerable, so
    the judge here is a policy rather than a model:

    * ``oracle``   -- always ADD, and additionally assert the sibling relation in
      invented fields, to test whether the parser can carry one at all.
    * ``identity`` -- always ADD, nothing else.
    * ``tau085``   -- cosine similarity against each offered memory; the first
      one at or above 0.85 is merged into (UPDATE), otherwise ADD. This is a
      similarity policy, ours and not Mem0's, and every number it produces is
      labelled with it.
    """

    def __init__(self, mode: str, *, similarity: Any = None, tau: float = 0.85) -> None:
        super().__init__()
        self.name = str(mode)
        self._mode = str(mode)
        self._similarity = similarity
        self._tau = float(tau)

    def _decide(self, prompt: str, offered: list[dict[str, str]]) -> list[dict[str, Any]]:
        text = str(self.current.get("text", ""))
        if self._mode == "tau085" and offered and self._similarity is not None:
            scores = [self._similarity(text, item.get("text", "")) for item in offered]
            best = max(range(len(scores)), key=lambda i: (scores[i], -i))
            if scores[best] >= self._tau:
                return [
                    {
                        "id": offered[best]["id"],
                        "text": text,
                        "event": "UPDATE",
                        "old_memory": offered[best].get("text", ""),
                        "similarity": round(float(scores[best]), 6),
                    }
                ]
        record: dict[str, Any] = {"id": str(len(offered)), "text": text, "event": "ADD"}
        if self._mode == "oracle":
            ids = [item["id"] for item in offered] + list(self.store_uuids)
            record["linked_memory_ids"] = ids
            record["sibling_memory_ids"] = ids
            record["relation"] = "INTERFERENCE_SIBLING"
            record["keep_both"] = True
            record["distinct_event"] = True
        return [record]

    def respond(self, messages: list[dict[str, Any]], response_format: Any = None) -> str:
        prompt = self._user_prompt(messages)
        offered = _parse_old_memory(prompt)
        payload = json.dumps({"memory": self._decide(prompt, offered)})
        return self._record(prompt, payload, [item["id"] for item in offered])


def _parse_old_memory(prompt: str) -> list[dict[str, str]]:
    """Pull the retrieved-memory list out of the v0.1 update prompt.

    Mem0 interpolates the list with an f-string, so what reaches the model is a
    Python repr and not JSON. Reading it with `json.loads` silently yields an
    empty list, which would make every write look as though no neighbour had
    been offered.
    """
    matches = _FENCED.findall(prompt)
    for block in reversed(matches):  # the resident memories are the last block
        try:
            rows = ast.literal_eval(block)
        except (ValueError, SyntaxError):
            continue
        out = [
            {"id": str(row["id"]), "text": str(row.get("text", ""))}
            for row in rows
            if isinstance(row, dict) and "id" in row
        ]
        if out:
            return out
    return []
