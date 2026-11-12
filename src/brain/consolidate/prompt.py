"""The merge prompt, its answer schema, its parser, and the cache key.

Separated so that what the baseline asks the model, and what it accepts back,
can be reviewed on one page. The wording is deliberately generous to the
baseline: a merge prompt that loses detail because it was written carelessly
would not be evidence about consolidation, only about our prompting.

Two things here are load-bearing for determinism.

1. **The prompt version is inside the prompt.** `PROMPT_VERSION` is the first
   line of the system message AND part of the cache key, so a revised prompt
   cannot replay answers written for the old one.
2. **The seed enters the key only when it can change the request.** At
   temperature 0 the request is a pure function of (group texts, prompt, model),
   so keying on the seed would multiply a five-seed pass's cost by five for
   byte-identical prompts. The rule is recorded in the key payload itself
   (`seed: null` at temperature 0) rather than left as a convention.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from brain.baselines.llm import CachedClient

__all__ = [
    "PROMPT_VERSION",
    "MAX_OUTPUT_TOKENS",
    "block_for",
    "merge_system_prompt",
    "merge_user_prompt",
    "merge_schema",
    "parse_merged",
    "merge_cache_key",
]

#: Bumped whenever the prompt or the schema changes. Part of the cache key, so
#: a bump invalidates every cached merge rather than mixing two prompt eras.
PROMPT_VERSION = "consolidate-baseline-v1"

#: Generous output cap: a faithful merge of eight long episodes must not be
#: truncated into a lossy one, and unused output tokens cost nothing.
MAX_OUTPUT_TOKENS = 4096

_SYSTEM_BODY = """You consolidate an engineering agent's long-term memory.

You are given a group of stored session notes that a similarity search judged to
be about the same thing. Merge them into ONE consolidated memory that will
replace the group in the store. A later question must be answerable from your
consolidated memory alone.

Preserve every distinguishing detail:
- dates, files and components, triggers, error signatures, root causes, fixes,
  PR or branch names, and measured outcomes.
- If the notes describe SEVERAL DIFFERENT occasions, say so and record each
  occasion separately with its own details. Do not average them into one
  blended account.
- If the notes restate the SAME occasion, state it once.
- If a later note revises an earlier one, keep the current value and record what
  it replaced and when.

Never invent a detail that is not in the notes, and never replace a specific
value with a vague summary. Length is not a concern; loss of detail is.

Each note starts with its id on its own line as [[id]].

NOTES
"""


def block_for(trace_id: str, text: str) -> str:
    """One note in the prompt: its id on its own line, then the full text."""
    return f"[[{trace_id}]]\n{text}"


def merge_system_prompt(blocks: Sequence[str]) -> str:
    """The merge instruction plus the group, with the prompt version pinned."""
    return f"PROMPT VERSION: {PROMPT_VERSION}\n\n" + _SYSTEM_BODY + "\n\n".join(blocks)


def merge_user_prompt(n_notes: int, *, repair: bool = False) -> str:
    """The request itself, with an optional nudge after an unparseable answer."""
    text = (
        f"Merge the {n_notes} notes above into one consolidated memory that can "
        "replace all of them. Keep every detail that tells them apart."
    )
    if repair:
        text += '\n\nReturn ONLY JSON of the form {"consolidated_memory": "..."}.'
    return text


def merge_schema() -> dict[str, Any]:
    """Structured output: one consolidated memory, nothing else to parse."""
    return {
        "type": "object",
        "properties": {"consolidated_memory": {"type": "string", "minLength": 1}},
        "required": ["consolidated_memory"],
        "additionalProperties": False,
    }


def parse_merged(text: str) -> str:
    """The consolidated memory from a structured answer, or the raw prose.

    Falling back to raw prose is deliberate generosity to the baseline: a model
    that writes a good merge but ignores the schema still contributes a usable
    consolidated memory instead of scoring as a failure. The fallback is counted
    in `stats["schema_bypassed"]`, so it is never invisible.
    """
    payload = text.strip()
    if not payload:
        return ""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    if isinstance(parsed, dict):
        value = parsed.get("consolidated_memory")
        return str(value).strip() if isinstance(value, str) else ""
    return "" if isinstance(parsed, list) else str(parsed).strip()


def merge_cache_key(
    *,
    member_ids: Sequence[str],
    member_shas: Sequence[str],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    system: str,
    user: str,
) -> str:
    """Complete key for one merge request.

    Covers the group identity (sorted ids plus their content hashes), the prompt
    version, the rendered prompt, the model, the temperature and the output cap.
    Ids alone would collide across two stores holding different text under the
    same id, which is why the content hashes are in the key as well.
    """
    return CachedClient.key(
        {
            "kind": "consolidate.merge",
            "prompt_version": PROMPT_VERSION,
            "member_ids": [str(i) for i in member_ids],
            "member_shas": [str(s) for s in member_shas],
            "model": str(model),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "seed": int(seed) if float(temperature) > 0.0 else None,
            "system": system,
            "user": user,
        }
    )
