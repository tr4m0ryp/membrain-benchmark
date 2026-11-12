"""The full-context arm: prompt content, token accounting, degradation, blocking.

Offline throughout: every call goes to the stub client from `baseline_arms`.
The live backend was validated separately with
`python -m brain.baselines.full_context`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.baselines.bm25 import Bm25Retriever
from brain.baselines.full_context import FullContextRetriever
from brain.baselines.llm import CachedClient, LlmResponse, LlmUnavailable
from brain.store import Trace
from tests.baseline_arms import QUERY, StubClient, full_context, traces  # noqa: F401


def test_full_context_puts_every_episode_in_the_prompt(tmp_path: Path, traces: list[Trace]) -> None:
    client = StubClient()
    with full_context(tmp_path, client) as arm:
        arm.index(traces)
        arm.retrieve(QUERY, 2)
        system = client.calls[0]["system"]
        for trace in traces:
            assert f"[[{trace.id}]]" in system
            assert trace.text in system  # full text, never truncated
        assert client.calls[0]["temperature"] == 0.0
        assert arm.stats["degraded_queries"] == 0


def test_full_context_records_its_token_usage(tmp_path: Path, traces: list[Trace]) -> None:
    with full_context(tmp_path) as arm:
        arm.index(traces)
        arm.retrieve(QUERY, 2)
        assert arm.stats["api_calls"] == 1
        assert arm.stats["live_prompt_tokens"] > 0
        assert arm.stats["live_output_tokens"] > 0
        assert arm.stats["cost_usd"] > 0.0
        assert arm.stats["corpus_prompt_tokens_estimate"] > 0
        assert arm.stats["fits_budget"] is True


def test_prompt_tokens_count_the_cache_buckets_too() -> None:
    # A cache-aware backend reports a 20k corpus under cache_creation, not
    # input_tokens. Summing only the latter would under-report by 1000x.
    response = LlmResponse(text="", input_tokens=10, cache_creation_input_tokens=20953,
                           cached_input_tokens=7)
    assert response.prompt_tokens == 20970


def test_full_context_prefetch_warms_the_cache(tmp_path: Path, traces: list[Trace]) -> None:
    queries = [QUERY, "truncated secret", QUERY]  # a repeat must not call twice
    client = StubClient()
    with full_context(tmp_path, client) as arm:
        arm.index(traces)
        assert arm.prefetch(queries, 2, workers=3) == 2
        assert len(client.calls) == 2
        hits = [arm.retrieve(query, 2) for query in queries]
        assert arm.stats["api_calls"] == 2  # every retrieve came from the cache
        assert arm.stats["cache_hits"] == 3
    with full_context(tmp_path, StubClient()) as replay:  # same cache dir
        replay.index(traces)
        assert [replay.retrieve(query, 2) for query in queries] == hits


def test_prefetch_is_a_no_op_for_local_arms(traces: list[Trace]) -> None:
    with Bm25Retriever() as arm:
        arm.index(traces)
        assert arm.prefetch([QUERY], 3) == 0


def test_full_context_degrades_explicitly_when_over_budget(
    tmp_path: Path, traces: list[Trace]
) -> None:
    client = StubClient()
    with full_context(tmp_path, client, max_prompt_tokens=60) as arm:
        arm.index(traces)
        hits = arm.retrieve(QUERY, 2)
        assert arm.stats["fits_budget"] is False
        assert arm.stats["degraded_queries"] == 1
        assert arm.stats["max_dropped_episodes"] > 0
        kept = [line for line in client.calls[0]["system"].splitlines() if line.startswith("[[")]
        assert 0 < len(kept) < len(traces)  # fewer notes, and the arm said so
        assert len(hits) == 2


def test_full_context_replays_from_its_disk_cache(tmp_path: Path, traces: list[Trace]) -> None:
    client = StubClient()
    with full_context(tmp_path, client) as first:
        first.index(traces)
        expected = first.retrieve(QUERY, 3)
    with full_context(tmp_path, StubClient(replies=["broken"])) as second:
        second.index(traces)
        assert second.retrieve(QUERY, 3) == expected  # identical, no new call
        assert second.stats["api_calls"] == 0
        assert second.stats["cache_hits"] == 1
        assert second.stats["replayed_prompt_tokens"] > 0


def test_full_context_survives_an_unparseable_answer(tmp_path: Path, traces: list[Trace]) -> None:
    client = StubClient(replies=["I cannot answer that.", "still not json"])
    with full_context(tmp_path, client) as arm:
        arm.index(traces)
        hits = arm.retrieve(QUERY, 2)
        assert len(hits) == 2  # padded in prompt order, never by similarity
        assert arm.stats["parse_failures"] == 1
        assert arm.stats["padded_slots"] == 2
        assert len(client.calls) == 2  # one repair retry


def test_full_context_parses_a_bare_json_array(tmp_path: Path, traces: list[Trace]) -> None:
    client = StubClient(replies=['Best first: ["c0-e1", "c0-e0"]'])
    with full_context(tmp_path, client) as arm:
        arm.index(traces)
        assert arm.retrieve(QUERY, 2) == ["c0-e1", "c0-e0"]


def test_full_context_reports_blocked_instead_of_guessing(
    tmp_path: Path, traces: list[Trace]
) -> None:
    arm = FullContextRetriever(client=None, cache_dir=tmp_path / "empty")
    arm._client = CachedClient(None, tmp_path / "empty")  # no backend, no cache
    arm.index(traces)
    assert arm.blocked_reason is not None
    assert arm.is_available() is False
    with pytest.raises(LlmUnavailable):
        arm.retrieve(QUERY, 2)


