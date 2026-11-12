"""Prompt construction, output schema and answer parsing for the LLM arm.

Separated from the arm so that what the model is asked, and how its answer is
read, can be reviewed on one page. The wording is deliberately generous to the
baseline: it names the interference structure and the axes that resolve it,
because a full-context arm that loses through a vague prompt would not be
evidence about full context.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "SYSTEM_HEADER",
    "block_for",
    "system_prompt",
    "user_prompt",
    "answer_schema",
    "estimate_tokens",
    "parse_ids",
]

SYSTEM_HEADER = (
    "You identify which stored engineering session note answers a question.\n"
    "The notes below are deliberately similar: several describe the same file and "
    "the same symptom on different occasions. Tell them apart by their specific "
    "details: the date, what triggered the event, the root cause, the fix, the "
    "artifact (PR or branch) and the outcome.\n"
    "Each note starts with its id on its own line as [[id]].\n\n"
    "NOTES\n"
)

#: Characters per token. Used ONLY for the pre-call budget check; the reported
#: numbers always come from the backend's own usage accounting.
_CHARS_PER_TOKEN = 4.0
_ARRAY = re.compile(r"\[[^\[\]]*\]")


def block_for(trace_id: str, text: str) -> str:
    """One note in the prompt: its id on its own line, then the full text."""
    return f"[[{trace_id}]]\n{text}"


def system_prompt(blocks: list[str]) -> str:
    return SYSTEM_HEADER + "\n\n".join(blocks)


def user_prompt(query: str, k: int, *, repair: bool = False) -> str:
    """The question plus the ranking instruction, with an optional repair nudge."""
    text = (
        f"QUESTION\n{query}\n\n"
        f"Return exactly {k} note ids, best first, most likely answer first. "
        "Copy the ids exactly as they appear between the double brackets. "
        "Rank on the specific details, not on the shared file or symptom."
    )
    if repair:
        text += '\n\nReturn ONLY JSON of the form {"ranked_ids": ["id1", "id2"]}.'
    return text


def answer_schema(k: int) -> dict[str, Any]:
    """Structured-output schema: exactly k ids, so the arm produces a ranking.

    Exactly k rather than at most k, so this arm is scored on the same shape of
    output as every other arm instead of on a short list padded from outside the
    model.
    """
    return {
        "type": "object",
        "properties": {
            "ranked_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": int(k),
                "maxItems": int(k),
            }
        },
        "required": ["ranked_ids"],
        "additionalProperties": False,
    }


def estimate_tokens(text: str) -> int:
    """Rough token count for the budget check only. Never reported as usage."""
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def fit_to_budget(
    ids: list[str], blocks: list[str], budget: int, order: list[int]
) -> tuple[list[str], list[str], int]:
    """Keep the blocks that fit, visiting them in `order`. Returns what was dropped.

    Corpus order is restored inside the kept set, so the prompt layout does not
    leak the selection ranking to the model. At least one block is always kept,
    otherwise a budget smaller than a single note would produce an empty prompt.
    """
    kept: set[int] = set()
    used = estimate_tokens(SYSTEM_HEADER)
    for position in order:
        cost = estimate_tokens(blocks[position])
        if used + cost > budget and kept:
            break
        used += cost
        kept.add(position)
    positions = [i for i in range(len(ids)) if i in kept]
    return (
        [ids[i] for i in positions],
        [blocks[i] for i in positions],
        len(ids) - len(positions),
    )


def parse_ids(text: str) -> list[str]:
    """Ids from a structured answer, or from the last JSON array in free text."""
    payload = text.strip()
    if not payload:
        return []
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        for candidate in reversed(_ARRAY.findall(payload)):
            try:
                values = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(values, list):
                return [str(v) for v in values if isinstance(v, (str, int))]
        return []
    values = parsed.get("ranked_ids", []) if isinstance(parsed, dict) else parsed
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if isinstance(v, (str, int))]
