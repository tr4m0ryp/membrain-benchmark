"""The entailment judge behind memory precision (I5a), cached on disk.

One LLM call per (transcript, assertion) pair returns a three-way verdict.
The third label is the point of the design: an invention that *sounds* right
is the interesting failure of a memory pipeline, and folding it into either
"entailed" or "contradicted" would hide exactly the case the metric exists to
count. All three counts are reported; the headline metric treats
`NOT_ENTAILED` and `UNSUPPORTED_BUT_PLAUSIBLE` together as invented.

**Caching.** Every response is written to `data/cache/entailment/` under a
hash of the exact request, following `data/cache/full_context/`. The prompt
version is part of the system text, so it is part of the key: editing a prompt
invalidates the cache instead of silently mixing two instruments. A re-run of
an already-judged set makes zero API calls.

**Accounting** lives next door in `accounting`, and every call goes through it:
prompt tokens are summed across all three buckets and two independent cost
figures are carried, because reading only `input_tokens` already produced a
near-zero cost figure on this project once.

Measured on this machine (claude-haiku-4-5 via the CLI): about 970 prompt
tokens, 290 output tokens, 0.003 USD and 6.4 s per uncached call.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from brain.baselines.llm import DEFAULT_MODEL, CachedClient, LlmUnavailable, select_client
from brain.config import DATA_DIR
from brain.eval.precision.accounting import JudgeStats

__all__ = [
    "ENTAILED",
    "NOT_ENTAILED",
    "UNSUPPORTED_BUT_PLAUSIBLE",
    "UNPARSED",
    "VERDICTS",
    "PROMPT_VERSION",
    "DEFAULT_CACHE_DIR",
    "EntailmentJudge",
]

ENTAILED = "ENTAILED"
NOT_ENTAILED = "NOT_ENTAILED"
UNSUPPORTED_BUT_PLAUSIBLE = "UNSUPPORTED_BUT_PLAUSIBLE"
#: Not a verdict: the judge answered something unparseable twice. Counted and
#: excluded from the denominator, never coerced into a label.
UNPARSED = "UNPARSED"
VERDICTS: tuple[str, ...] = (ENTAILED, NOT_ENTAILED, UNSUPPORTED_BUT_PLAUSIBLE)

#: Bumped whenever a prompt changes. It is embedded in the system text, so it is
#: inside the cache key: an edited prompt cannot replay old answers.
PROMPT_VERSION = "entailment-v1"
DEFAULT_CACHE_DIR: Path = DATA_DIR / "cache" / "entailment"
_MAX_OUTPUT_TOKENS = 300
_MAX_EXTRACT_TOKENS = 700

_JUDGE_SYSTEM = f"""[{PROMPT_VERSION}] You are a strict entailment judge auditing a memory system.

You get a TRANSCRIPT, which is the only evidence that exists, and one ASSERTION a
memory pipeline stored after reading it. Decide whether the transcript supports the
assertion.

ENTAILED: the transcript states the assertion, or the assertion follows from it with
certainty. Paraphrase counts. Restating a number in other units counts. Something
that must be true given what the transcript says counts.

UNSUPPORTED_BUT_PLAUSIBLE: the assertion fits the transcript and may well be true,
but the transcript does not establish it. A likely inference, a typical detail, or a
cause the transcript merely hints at belongs here. This is the label for a good guess.

NOT_ENTAILED: the transcript contradicts the assertion, or the assertion is about
something the transcript never raises at all.

Rules. Judge only against the transcript, never against what is usually true of
software. Wording may differ freely; only the claim matters. If part of the assertion
is supported and part is not, judge the unsupported part. Return the tool call with
one label and a reason of at most 25 words."""

_EXTRACT_SYSTEM = f"""[{PROMPT_VERSION}] You are the capture step of an agent memory system.

Read the session note and store what should be remembered about it as short
standalone assertions, one fact each. Each assertion is a single sentence that a
reader who never saw the note could understand on its own. Return the tool call."""


def _verdict_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }


def _extract_schema(max_assertions: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assertions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": int(max_assertions),
            }
        },
        "required": ["assertions"],
        "additionalProperties": False,
    }


def _judge_user(transcript: str, assertion: str, *, repair: bool = False) -> str:
    text = f"TRANSCRIPT\n{transcript.strip()}\n\nASSERTION\n{assertion.strip()}\n"
    if repair:
        text += (
            '\nReturn ONLY JSON of the form {"verdict": "ENTAILED", "reason": "..."} '
            f"with verdict one of {', '.join(VERDICTS)}."
        )
    return text


def _parse_verdict(text: str) -> tuple[str, str]:
    """Verdict and reason from a structured answer, or from loose text."""
    payload = (text or "").strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        verdict = str(parsed.get("verdict", "")).strip().upper()
        if verdict in VERDICTS:
            return verdict, str(parsed.get("reason", ""))[:400]
    # Longest label first: NOT_ENTAILED contains ENTAILED as a substring.
    for label in sorted(VERDICTS, key=len, reverse=True):
        if re.search(rf"\b{label}\b", payload.upper()):
            return label, payload[:400]
    return UNPARSED, payload[:400]


def _parse_assertions(text: str) -> list[str]:
    payload = (text or "").strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    values = parsed.get("assertions", []) if isinstance(parsed, dict) else parsed
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]


class EntailmentJudge:
    """Cached three-way entailment judge, plus the capture step it audits.

    `extract` is deliberately here rather than in the metric: it is the pipeline
    under audit (one bounded Haiku call per transcript, the shape the spec fixes
    for capture in T2), and the metric must stay a function of assertions and a
    transcript so it can be pointed at any other system's output.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ) -> None:
        backend, reason = (client, None) if client is not None else select_client()
        self._client = CachedClient(backend, cache_dir)
        self._cache_dir = Path(cache_dir)
        self._backend_reason = reason
        self.model = str(model)
        self.temperature = float(temperature)
        self.stats = JudgeStats()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def blocked_reason(self) -> str | None:
        """None when the judge can answer, live or entirely from cache."""
        if self._client.inner is not None:
            return None
        if self._cache_dir.is_dir() and any(self._cache_dir.glob("*.json")):
            return None
        return self._backend_reason or "no LLM backend configured"

    def _complete(self, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> Any:
        response = self._client.complete(
            system=system,
            user=user,
            model=self.model,
            temperature=self.temperature,
            max_tokens=int(max_tokens),
            schema=schema,
        )
        self.stats.record(response)
        return response

    def judge(self, transcript: str, assertion: str) -> tuple[str, str]:
        """One verdict. Retries once with a repair nudge before giving up."""
        schema = _verdict_schema()
        response = self._complete(
            _JUDGE_SYSTEM, _judge_user(transcript, assertion), schema, _MAX_OUTPUT_TOKENS
        )
        verdict, reason = _parse_verdict(response.text)
        if verdict == UNPARSED:
            self.stats.note_parse_failure()
            response = self._complete(
                _JUDGE_SYSTEM,
                _judge_user(transcript, assertion, repair=True),
                schema,
                _MAX_OUTPUT_TOKENS,
            )
            verdict, reason = _parse_verdict(response.text)
        return verdict, reason

    def judge_many(
        self, pairs: Sequence[tuple[str, str]], *, workers: int = 6
    ) -> list[tuple[str, str]]:
        """Judge (transcript, assertion) pairs, concurrently when a backend is live.

        The calls are independent and the cache is keyed by the exact request, so
        concurrency changes the wall clock and nothing else.
        """
        items = list(pairs)
        if not items:
            return []
        if int(workers) <= 1 or self._client.inner is None:
            return [self.judge(t, a) for t, a in items]
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            return list(pool.map(lambda pair: self.judge(pair[0], pair[1]), items))

    def extract(self, transcript: str, *, max_assertions: int = 8) -> list[str]:
        """The audited capture step: stored assertions for one transcript."""
        response = self._complete(
            _EXTRACT_SYSTEM,
            f"SESSION NOTE\n{transcript.strip()}\n\nStore 4 to {max_assertions} assertions.",
            _extract_schema(max_assertions),
            _MAX_EXTRACT_TOKENS,
        )
        found = _parse_assertions(response.text)
        if not found:
            self.stats.note_parse_failure()
        return found[: int(max_assertions)]

    def extract_many(
        self, transcripts: Sequence[str], *, max_assertions: int = 8, workers: int = 6
    ) -> list[list[str]]:
        items = list(transcripts)
        if not items:
            return []
        if int(workers) <= 1 or self._client.inner is None:
            return [self.extract(t, max_assertions=max_assertions) for t in items]
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            return list(pool.map(lambda t: self.extract(t, max_assertions=max_assertions), items))


def _smoke() -> None:
    """Tiny path: python -m brain.eval.precision.entailment (two cached calls)."""
    judge = EntailmentJudge()
    if judge.blocked_reason:
        print(f"entailment judge BLOCKED: {judge.blocked_reason}")
        return
    transcript = (
        "[2026-02-03] Fixed the flapping flag in the evaluator: a cache entry was never "
        "invalidated on write. Invalidated it inside the write transaction, PR #9166."
    )
    for assertion in (
        "A cache entry was not invalidated on write.",
        "The evaluator is written in Rust.",
    ):
        verdict, reason = judge.judge(transcript, assertion)
        print(f"{verdict:26s} {assertion}  <- {reason[:80]}")
    print(json.dumps(judge.stats.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        _smoke()
    except LlmUnavailable as exc:  # pragma: no cover - depends on the environment
        print(f"entailment judge unavailable: {exc}")
