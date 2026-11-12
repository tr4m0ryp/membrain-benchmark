"""Shared fixtures for the baseline-arm tests.

The stub LLM client keeps every test offline: it answers by reading the ids back
out of the prompt it was given, and records the exact request, which is how the
prompt-budget and token-accounting behaviour is checked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brain.baselines.bm25 import Bm25Retriever
from brain.baselines.full_context import FullContextRetriever
from brain.baselines.llm import LlmResponse
from brain.baselines.plain_vector import PlainVectorRetriever
from brain.baselines.smoke import SMOKE_QUERY, smoke_traces
from brain.baselines.write_time_mmr import WriteTimeMmrRetriever
from brain.store import Trace

QUERY = SMOKE_QUERY


@pytest.fixture(scope="module")
def traces() -> list[Trace]:
    return smoke_traces()


class StubClient:
    """Answers by reading the ids back out of the prompt. Records every call."""

    name = "stub"

    def __init__(self, replies: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies = list(replies or [])

    def complete(self, *, system: str, user: str, model: str, temperature: float,
                 max_tokens: int, schema: dict[str, Any] | None) -> LlmResponse:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        want = int(schema["properties"]["ranked_ids"]["maxItems"]) if schema else 1
        ids = [line[2:-2] for line in system.splitlines() if line.startswith("[[")]
        text = self.replies.pop(0) if self.replies else json.dumps({"ranked_ids": ids[:want]})
        return LlmResponse(text=text, input_tokens=len(system) // 4, output_tokens=11,
                           cost_usd=0.001, source="api")


def full_context(tmp_path: Path, client: Any | None = None, **kwargs: Any) -> FullContextRetriever:
    return FullContextRetriever(client=client or StubClient(), cache_dir=tmp_path / "llm", **kwargs)


def all_arms(tmp_path: Path) -> list[Any]:
    return [
        PlainVectorRetriever(),
        Bm25Retriever(),
        WriteTimeMmrRetriever(),
        full_context(tmp_path),
    ]
