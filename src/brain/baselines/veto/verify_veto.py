"""Arm B of the veto experiment: recollection veto, one candidate at a time (I4).

The mechanism under test, from proposal-v3 I4:

    For each retrieved candidate, attempt to retrieve the specific trace of the
    corresponding TRUE item; a registered MISMATCH vetoes the candidate. Output
    is a VETO, not a graded score, and rejection is driven by positive evidence
    about the correct item rather than the impostor's low similarity.

Four properties are enforced in the code path, because each one is a way this
arm could silently collapse back into arm A:

1. **One candidate per call, in isolation.** The prompt carries the question and
   exactly ONE note, so the model can never compare candidates and cannot rank.
2. **The judgement is on ONE attribute, not on relevance.** The model must name
   the attribute the question keys on (trigger, outcome, cause, fix, artifact,
   date), then read this note's value for that same attribute. "How related is
   this note" is explicitly ruled out in the prompt.
3. **A veto needs positive evidence, checked mechanically.** MISMATCH is honoured
   only when the returned `recalled` value is grounded in this note's own text.
   A MISMATCH with nothing recalled is DOWNGRADED to ABSENT and counted, so
   "this looks unrelated" can never become a veto.
4. **The output is a boolean.** No score, no confidence, no margin anywhere in
   the return path. Survivors keep their base order exactly; vetoed candidates
   are appended below in their base order. A FILTER, never a re-scorer.

Failure to recall is not evidence: ABSENT never vetoes. That is the faithful
reading of recall-to-reject, where rejecting a lure requires recollecting the
conflicting detail rather than merely failing to recognise it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from brain.baselines.bm25 import tokenize
from brain.baselines.llm import DEFAULT_MODEL, LlmUnavailable
from brain.baselines.vectors import VectorCache
from brain.baselines.veto.llm_pick import LlmCandidateArm
from brain.config import DATA_DIR

__all__ = ["VerifyVetoRetriever", "VetoDecision", "VETO_CACHE_DIR", "VETO_SYSTEM_HEADER"]

VETO_CACHE_DIR = DATA_DIR / "cache" / "verify_veto"
_MAX_OUTPUT_TOKENS = 400
#: A recalled span counts as grounded when at least this fraction of its tokens
#: occur in the note. Fixed before any measurement; it is a guard against an
#: ungrounded veto, not a tuning knob on the result.
_GROUNDING = 0.5
_VERDICTS = ("MATCH", "MISMATCH", "ABSENT")

VETO_SYSTEM_HEADER = (
    "You are checking ONE stored engineering session note against a question "
    "about a specific past event.\n\n"
    "The question identifies the event the asker remembers by naming a "
    "distinguishing detail about it: what triggered it, what its outcome was, "
    "what the root cause turned out to be, which change fixed it, which PR or "
    "branch carried the fix, or when it happened.\n\n"
    "Your job is RECALL, not similarity. Do not judge how related this note is "
    "to the question. Do exactly this:\n"
    "1. Read the question and name the ATTRIBUTE it keys on (one of: trigger, "
    "outcome, cause, fix, artifact, date) and the specific value it asserts.\n"
    "2. Look up what THIS NOTE records for that SAME attribute and quote that "
    "value from the note. If the note does not record that attribute, leave it "
    "empty.\n"
    "3. Decide:\n"
    "   MATCH    - the note records that attribute and its value is the one the "
    "question asserts (a paraphrase of the same fact counts as MATCH).\n"
    "   MISMATCH - the note records that attribute and its value is a DIFFERENT, "
    "conflicting fact from the one the question asserts.\n"
    "   ABSENT   - the note does not record that attribute, or you cannot tell.\n\n"
    "Answer MISMATCH only when you have read a conflicting value out of this "
    "note and can quote it. Never answer MISMATCH because the note merely looks "
    "unrelated to the question.\n\n"
    "NOTE\n"
)


@dataclass(frozen=True, slots=True)
class VetoDecision:
    """One candidate's boolean verdict, plus the evidence that justified it."""

    trace_id: str
    verdict: str
    attribute: str
    cue: str
    recalled: str
    grounded: bool

    @property
    def veto(self) -> bool:
        """The whole output of the mechanism: one boolean, no score."""
        return self.verdict == "MISMATCH" and self.grounded

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "verdict": self.verdict,
            "attribute": self.attribute,
            "cue": self.cue,
            "recalled": self.recalled,
            "grounded": bool(self.grounded),
            "veto": bool(self.veto),
        }


def grounded(recalled: str, note: str) -> bool:
    """Is `recalled` actually read out of `note`, rather than invented?"""
    tokens = tokenize(str(recalled))
    if not tokens:
        return False
    present = set(tokenize(str(note)))
    return (sum(1 for token in tokens if token in present) / len(tokens)) >= _GROUNDING


class VerifyVetoRetriever(LlmCandidateArm):
    """Arm B: up to k independent calls per query, each returning one boolean."""

    name = "verify_veto"

    def __init__(
        self,
        *,
        base: Any | None = None,
        cache: VectorCache | None = None,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        cache_dir: Path | str = VETO_CACHE_DIR,
        name: str | None = None,
    ) -> None:
        super().__init__(
            base=base,
            cache=cache,
            client=client,
            model=model,
            temperature=temperature,
            cache_dir=cache_dir,
            name=name,
            params={
                "max_output_tokens": _MAX_OUTPUT_TOKENS,
                "calls_per_query": "one per candidate (k)",
                "output": "boolean veto per candidate; survivors keep base order",
                "veto_rule": "MISMATCH and the recalled value is grounded in the note",
                "grounding_threshold": _GROUNDING,
                "absent_vetoes": False,
            },
        )
        #: query text -> {trace_id: VetoDecision}. The experiment reads this to
        #: score the correct-item rejection axis straight off the mechanism.
        self.decisions: dict[str, dict[str, VetoDecision]] = {}
        self.stats.update(
            {
                "vetoes": 0,
                "ungrounded_vetoes_downgraded": 0,
                "candidates_judged": 0,
                "all_vetoed_queries": 0,
            }
        )

    def _system(self, trace_id: str) -> str:
        return f"{VETO_SYSTEM_HEADER}[[{trace_id}]]\n{self._texts[trace_id]}\n"

    @staticmethod
    def _user(query: str) -> str:
        return (
            f"QUESTION\n{query}\n\n"
            "Name the attribute the question keys on and the value it asserts, "
            "quote what this note records for that same attribute (empty if the "
            "note does not record it), then give the verdict."
        )

    @staticmethod
    def _schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "attribute": {"type": "string"},
                "cue": {"type": "string"},
                "recalled": {"type": "string"},
                "verdict": {"type": "string", "enum": list(_VERDICTS)},
            },
            "required": ["attribute", "cue", "recalled", "verdict"],
            "additionalProperties": False,
        }

    def _ask(self, query: str, trace_id: str) -> Any:
        return self.request(
            self._system(trace_id),
            self._user(query),
            max_tokens=_MAX_OUTPUT_TOKENS,
            schema=self._schema(),
        )

    def judge(self, query: str, trace_id: str) -> VetoDecision:
        """One candidate, judged in isolation. The unit of the mechanism."""
        try:
            payload = json.loads(str(self._ask(query, trace_id).text).strip() or "{}")
            payload = payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            payload = {}
        verdict = str(payload.get("verdict", "")).strip().upper()
        if verdict not in _VERDICTS:
            self.stats["parse_failures"] = int(self.stats["parse_failures"]) + 1
            verdict = "ABSENT"
        recalled = str(payload.get("recalled", "") or "")
        is_grounded = grounded(recalled, self._texts[trace_id])
        if verdict == "MISMATCH" and not is_grounded:
            # Positive evidence is the whole point: a veto with nothing recalled
            # out of the note is exactly the similarity judgement this arm must
            # not make, so it is refused rather than trusted.
            self.stats["ungrounded_vetoes_downgraded"] = (
                int(self.stats["ungrounded_vetoes_downgraded"]) + 1
            )
        return VetoDecision(
            trace_id=str(trace_id),
            verdict=verdict,
            attribute=str(payload.get("attribute", "") or ""),
            cue=str(payload.get("cue", "") or ""),
            recalled=recalled,
            grounded=bool(is_grounded),
        )

    def _decide(self, query: str, candidate_ids: Sequence[str]) -> list[str]:
        decisions = {tid: self.judge(query, tid) for tid in candidate_ids}
        self.decisions[query] = decisions
        vetoed = [tid for tid in candidate_ids if decisions[tid].veto]
        survivors = [tid for tid in candidate_ids if not decisions[tid].veto]
        self.stats["candidates_judged"] = (
            int(self.stats["candidates_judged"]) + len(candidate_ids)
        )
        self.stats["vetoes"] = int(self.stats["vetoes"]) + len(vetoed)
        if candidate_ids and not survivors:
            self.stats["all_vetoed_queries"] = int(self.stats["all_vetoed_queries"]) + 1
        # A pure filter: survivors keep base order, vetoed keep theirs and go
        # below. Nothing is re-scored, so any gain is attributable to the
        # boolean and to nothing else.
        return survivors + vetoed

    def prefetch(self, queries: Sequence[str], k: int, *, workers: int = 8) -> int:
        """Warm every (query, candidate) pair. Returned ids are unchanged."""
        pairs = [(query, trace_id) for query, ids in self._pending(queries, k) for trace_id in ids]
        return self._run_warm(pairs, lambda pair: self._warm(*pair), workers)

    def _warm(self, query: str, trace_id: str) -> None:
        try:
            self._ask(query, trace_id)
        except LlmUnavailable:
            self._count_failure()


def _smoke() -> None:
    """Tiny path: python -m brain.baselines.veto.verify_veto."""
    from brain.baselines.smoke import run_smoke

    arm = VerifyVetoRetriever()
    if not arm.is_available():
        print(f"verify_veto BLOCKED: {arm.blocked_reason}")
        return
    with arm:
        run_smoke(arm, k=3)


if __name__ == "__main__":
    _smoke()
