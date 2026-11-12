"""Full-context arm: the whole history in the prompt, the naive-but-hard baseline.

This is the arm published memory systems lose to (Evaluation Plan; Mem0 and Zep
both lost to it), so it is built to win if it can:

* **Every episode, in full, every query.** No summarization, no chunking, no
  truncation of a note's tail. The corpus is presented in one block with stable
  ids, and the model is told the notes are near-duplicates and which detail
  axes tell them apart. It sees strictly more evidence than any retrieval arm.
* **A ranking, not a single guess.** The model returns the k best ids, best
  first, so it is scored by exactly the metric the other arms are scored by
  instead of being handicapped into a top-1-only arm.
* **Honest accounting.** Input and output tokens are recorded per query from
  the backend's own usage numbers, along with cost. If the corpus does not fit
  the budget the arm degrades EXPLICITLY: it keeps the episodes nearest the
  query, marks the query as degraded and records how many were dropped. It
  never silently truncates its way to a better-looking number.

Determinism: temperature 0 where the backend allows it, plus an on-disk
response cache keyed by the exact request, so repeated seeds replay identical
answers rather than resampling. The arm itself has no randomness, so its result
is seed-independent by construction; that is a property of the arm, not a bug.

Measured on the frozen corpus (120 episodes, claude-haiku-4-5 via the CLI, a
15-query probe): 21.9k prompt tokens per call (98.9 percent cache reads), 6.4k
output tokens, 0.036 USD and 13 s per query at 6 prefetch workers, and 15 of 15
correct against 6 of 15 for plain vector. Budget a cold 360-query pass at about
13 USD and an hour, once; later seeds replay from cache.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from brain.baselines.base import BaseRetriever, rank_hits
from brain.baselines.llm import DEFAULT_MODEL, CachedClient, LlmUnavailable, select_client
from brain.baselines.prompt import (
    answer_schema,
    block_for,
    estimate_tokens,
    fit_to_budget,
    parse_ids,
    system_prompt,
    user_prompt,
)
from brain.baselines.vectors import VectorCache, shared_vector_cache
from brain.config import DATA_DIR
from brain.store import Trace

__all__ = ["FullContextRetriever", "DEFAULT_CACHE_DIR", "DEFAULT_MAX_PROMPT_TOKENS"]

#: Haiku 4.5 carries a 200k context window. The ceiling leaves room for the
#: instructions, the question and the answer, and is the point at which the arm
#: degrades loudly instead of overflowing.
DEFAULT_MAX_PROMPT_TOKENS = 150_000
#: Cached responses live next to the corpus so a rerun is offline and identical.
DEFAULT_CACHE_DIR = DATA_DIR / "cache" / "full_context"
_MAX_OUTPUT_TOKENS = 1024


class FullContextRetriever(BaseRetriever):
    """Put the episodes in the prompt and let the model rank them."""

    name = "full_context"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        vector_cache: VectorCache | None = None,
        name: str | None = None,
    ) -> None:
        backend, reason = (client, None) if client is not None else select_client()
        super().__init__(
            name=name,
            params={
                "model": model,
                "temperature": float(temperature),
                "max_prompt_tokens": int(max_prompt_tokens),
                # Output cap and temperature are honoured by the SDK backend
                # only; the CLI exposes neither, which `stats` records per run.
                "max_output_tokens": _MAX_OUTPUT_TOKENS,
                "backend": getattr(backend, "name", None),
                "episode_order": "corpus order (id-sorted upstream)",
                "degradation": "keep the episodes nearest the query, mark and count",
            },
        )
        self._model = str(model)
        self._temperature = float(temperature)
        self._budget = int(max_prompt_tokens)
        self._backend_reason = reason
        self._client = CachedClient(backend, cache_dir)
        self._cache_dir = Path(cache_dir)
        self._vectors = vector_cache
        self._blocks: list[str] = []
        self._texts: dict[str, str] = {}
        self._lock = Lock()
        self.stats.update(
            {
                "api_calls": 0,
                "cache_hits": 0,
                # Everything the model read on a live call: fresh plus
                # cache-written plus cache-read prompt tokens.
                "live_prompt_tokens": 0,
                "live_cache_creation_tokens": 0,
                "live_cache_read_tokens": 0,
                "live_output_tokens": 0,
                "replayed_prompt_tokens": 0,
                "cost_usd": 0.0,
                "degraded_queries": 0,
                "max_dropped_episodes": 0,
                "parse_failures": 0,
                "padded_slots": 0,
                "temperature_enforced": True,
            }
        )

    # -- availability ------------------------------------------------------

    @property
    def blocked_reason(self) -> str | None:
        """None when the arm can answer, live or from cache."""
        if self._client.inner is not None:
            return None
        if self._cache_dir.is_dir() and any(self._cache_dir.glob("*.json")):
            return None
        return self._backend_reason or "no LLM backend configured"

    # -- write path --------------------------------------------------------

    def _index(self, traces: Sequence[Trace]) -> None:
        self._texts = {trace.id: trace.text for trace in traces}
        self._blocks = [block_for(trace.id, trace.text) for trace in traces]
        self.stats["corpus_prompt_tokens_estimate"] = estimate_tokens(system_prompt(self._blocks))
        self.stats["fits_budget"] = self.stats["corpus_prompt_tokens_estimate"] <= self._budget

    # -- read path ---------------------------------------------------------

    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        ids, blocks, dropped = self._within_budget(query)
        if dropped:
            self.stats["degraded_queries"] = int(self.stats["degraded_queries"]) + 1
            self.stats["max_dropped_episodes"] = max(int(self.stats["max_dropped_episodes"]), dropped)
        system = system_prompt(blocks)
        ranked = self._ask(system, query, k)
        if not ranked:
            self.stats["parse_failures"] = int(self.stats["parse_failures"]) + 1
            ranked = self._ask(system, query, k, repair=True)
        chosen = [trace_id for trace_id in ranked if trace_id in self._texts][:k]
        seen = set(chosen)
        if len(chosen) < k:
            # Pad in prompt order first, then the rest of the corpus (only
            # reachable on a degraded query). Never by similarity: borrowing the
            # vector arm's ranking here would quietly turn this into a hybrid,
            # and the slots are counted so a padded result is visible.
            self.stats["padded_slots"] = int(self.stats["padded_slots"]) + (k - len(chosen))
            for trace_id in [*ids, *self._texts]:
                if len(chosen) >= k:
                    break
                if trace_id not in seen:
                    chosen.append(trace_id)
                    seen.add(trace_id)
        # Descending integer scores: the model returns an order, not a calibrated
        # score, and this keeps that order through the shared tie-break.
        return rank_hits(
            ((trace_id, float(len(chosen) - rank)) for rank, trace_id in enumerate(chosen)), k
        )

    def _within_budget(self, query: str) -> tuple[list[str], list[str], int]:
        """All episodes when they fit; otherwise the nearest ones, and what was dropped."""
        ids = list(self._texts)
        if self.stats.get("fits_budget", True):
            return ids, list(self._blocks), 0
        return fit_to_budget(ids, self._blocks, self._budget, self._nearest_first(query))

    def _nearest_first(self, query: str) -> list[int]:
        cache = self._vectors if self._vectors is not None else shared_vector_cache()
        matrix = cache.matrix([self._texts[trace_id] for trace_id in self._texts])
        sims = matrix @ cache.vector(query)
        # Ascending id as the tie-break, matching every other arm.
        ids = list(self._texts)
        return sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))

    def prefetch(self, queries: Sequence[str], k: int, *, workers: int = 6) -> int:
        """Warm the response cache in parallel. Optional, and results are unchanged.

        A cold pass over the frozen corpus is 360 sequential calls of 20 to 100
        seconds each. The calls are independent and the cache is keyed by the
        exact request, so running them concurrently changes the wall clock and
        nothing else. The degraded path builds a query-dependent prompt from the
        embedder, so it stays sequential rather than sharing the embedder across
        threads.
        """
        if not self._indexed or self._client.inner is None:
            return 0
        pending = [q for q in dict.fromkeys(queries) if isinstance(q, str)]
        if not pending:
            return 0
        want = min(int(k), len(self._traces))
        if want <= 0:
            return 0
        if not self.stats.get("fits_budget", True) or int(workers) <= 1:
            for query in pending:
                _, blocks, _ = self._within_budget(query)
                self._warm(system_prompt(blocks), query, want)
            return len(pending)
        system = system_prompt(self._blocks)
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            list(pool.map(lambda query: self._warm(system, query, want), pending))
        return len(pending)

    def _warm(self, system: str, query: str, k: int) -> None:
        """One cache-filling call. A failure here is retried by the real pass."""
        try:
            self._request(system, user_prompt(query, k), k)
        except LlmUnavailable:
            with self._lock:
                self.stats["prefetch_failures"] = int(self.stats.get("prefetch_failures", 0)) + 1

    def _ask(self, system: str, query: str, k: int, *, repair: bool = False) -> list[str]:
        response = self._request(system, user_prompt(query, k, repair=repair), k)
        return parse_ids(response.text)

    def _request(self, system: str, user: str, k: int) -> Any:
        # An unavailable backend propagates as LlmUnavailable: Task 008 reports
        # the arm BLOCKED rather than scoring it as if it had answered badly.
        response = self._client.complete(
            system=system,
            user=user,
            model=self._model,
            temperature=self._temperature,
            max_tokens=_MAX_OUTPUT_TOKENS,
            schema=answer_schema(k),
        )
        self._account(response)
        return response

    def _account(self, response: Any) -> None:
        with self._lock:  # prefetch accounts from several threads
            if response.source == "cache":
                self.stats["cache_hits"] = int(self.stats["cache_hits"]) + 1
                self.stats["replayed_prompt_tokens"] = (
                    int(self.stats["replayed_prompt_tokens"]) + response.prompt_tokens
                )
            else:
                self.stats["api_calls"] = int(self.stats["api_calls"]) + 1
                self.stats["live_prompt_tokens"] = (
                    int(self.stats["live_prompt_tokens"]) + response.prompt_tokens
                )
                self.stats["live_cache_creation_tokens"] = (
                    int(self.stats["live_cache_creation_tokens"])
                    + int(response.cache_creation_input_tokens)
                )
                self.stats["live_cache_read_tokens"] = (
                    int(self.stats["live_cache_read_tokens"]) + int(response.cached_input_tokens)
                )
                self.stats["live_output_tokens"] = (
                    int(self.stats["live_output_tokens"]) + int(response.output_tokens)
                )
                self.stats["cost_usd"] = float(self.stats["cost_usd"]) + float(response.cost_usd)
            if not response.temperature_enforced:
                self.stats["temperature_enforced"] = False


def _smoke() -> None:
    """Tiny path: python -m brain.baselines.full_context (one live or cached call)."""
    from brain.baselines.smoke import run_smoke

    arm = FullContextRetriever()
    if not arm.is_available():
        print(f"full_context BLOCKED: {arm.blocked_reason}")
        return
    run_smoke(arm, k=2)


if __name__ == "__main__":
    _smoke()
