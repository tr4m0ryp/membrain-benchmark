"""Arm A of the veto experiment: give the model the top-k and let it pick (I4).

This is the arm to beat, so it is built to win. Handicapping it would repeat the
exact error this project has criticised in vendor benchmarks. It sees every
candidate at once (comparative evidence, which arm B is denied by construction);
it returns a RANKING rather than one id, which dominates a single pick on the
metric it is scored by because the first element IS the pick; and the prompt names
the interference structure in the full-context arm's wording, so it loses on the
problem rather than on framing.

Two base classes live here because arm A defines the shared contracts and arm B
imports them. `CandidateArm` fixes the candidate set: BOTH arms draw their k
candidates from one deterministic plain-vector base and differ only in what they
do with that list, which is what makes the comparison paired rather than
confounded by two retrieval front ends. `LlmCandidateArm` adds the cached
transport and the token accounting, once, for both.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from brain.baselines.base import BaseRetriever, rank_hits
from brain.baselines.llm import DEFAULT_MODEL, CachedClient, LlmUnavailable, select_client
from brain.baselines.plain_vector import PlainVectorRetriever
from brain.baselines.prompt import block_for, parse_ids
from brain.baselines.vectors import VectorCache
from brain.config import DATA_DIR
from brain.store import Trace

__all__ = ["CandidateArm", "LlmCandidateArm", "LlmPickRetriever", "PICK_CACHE_DIR",
           "PICK_SYSTEM_HEADER"]

#: Cached responses live next to the corpus so a rerun is offline and identical.
PICK_CACHE_DIR = DATA_DIR / "cache" / "llm_pick"
_MAX_OUTPUT_TOKENS = 512

PICK_SYSTEM_HEADER = (
    "You identify which stored engineering session note answers a question.\n"
    "The notes below are deliberately similar: several describe the same file and "
    "the same symptom on different occasions. Tell them apart by their specific "
    "details: the date, what triggered the event, the root cause, the fix, the "
    "artifact (PR or branch) and the outcome.\n"
    "Each note starts with its id on its own line as [[id]].\n\n"
    "NOTES\n"
)
#: Counters every LLM arm reports, in one place so none of the three prompt-token
#: buckets can be quietly omitted.
_COUNTERS = ("api_calls", "cache_hits", "parse_failures", "unique_requests",
             "live_prompt_tokens", "live_cache_creation_tokens", "live_cache_read_tokens",
             "live_output_tokens", "replayed_prompt_tokens", "replayed_output_tokens")


class CandidateArm(BaseRetriever):
    """An arm that post-processes a fixed plain-vector candidate set.

    The base retriever is owned, indexed and queried here, so every arm built on
    `CandidateArm` sees byte-identical candidates for a given query. Subclasses
    implement `_decide` and never touch retrieval.
    """

    name = "candidate_arm"

    def __init__(self, *, base: Any | None = None, cache: VectorCache | None = None,
                 name: str | None = None, params: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, params=params)
        self._base = base if base is not None else PlainVectorRetriever(cache=cache)
        self.params.setdefault("candidate_source", getattr(self._base, "name", "plain_vector"))
        self._texts: dict[str, str] = {}
        #: query text -> candidate ids in base order. Read by the experiment to
        #: PROVE both arms were fed the same list, not merely believe it.
        self.candidates: dict[str, list[str]] = {}

    @property
    def base(self) -> Any:
        return self._base

    def close(self) -> None:
        if hasattr(self._base, "close"):
            self._base.close()

    def _index(self, traces: Sequence[Trace]) -> None:
        self._base.index(traces)
        self._texts = {trace.id: trace.text for trace in traces}

    def candidate_ids(self, query: str, k: int) -> list[str]:
        """The shared top-k, in base order. Recorded for the pairing check."""
        ids = [trace_id for trace_id, _ in self._base.retrieve_scored(query, k)]
        self.candidates[query] = list(ids)
        return ids

    def _retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        ordered = self._decide(query, self.candidate_ids(query, k))
        # Descending integer scores keep the arm's chosen ORDER through the
        # shared tie-break; no arm here emits a calibrated score.
        return rank_hits(
            ((tid, float(len(ordered) - rank)) for rank, tid in enumerate(ordered)), k)

    def _decide(self, query: str, candidate_ids: Sequence[str]) -> list[str]:
        """Reorder (never extend) the candidate list. Subclass hook."""
        raise NotImplementedError


class LlmCandidateArm(CandidateArm):
    """A `CandidateArm` with a cached LLM transport and honest token accounting.

    Prompt tokens arrive in three buckets and all three are prompt: the CLI puts
    a large prompt under `cache_creation_input_tokens`, so summing only
    `input_tokens` under-reports the context by orders of magnitude. That bug
    already shipped once, so the sum happens in exactly one place here."""

    def __init__(self, *, base: Any | None, cache: VectorCache | None, client: Any | None,
                 model: str, temperature: float, cache_dir: Path | str, name: str | None,
                 params: dict[str, Any]) -> None:
        backend, reason = (client, None) if client is not None else select_client()
        super().__init__(
            base=base, cache=cache, name=name,
            params={"model": str(model), "temperature": float(temperature),
                    "backend": getattr(backend, "name", None), **params},
        )
        self._model = str(model)
        self._temperature = float(temperature)
        self._backend_reason = reason
        self._client = CachedClient(backend, cache_dir)
        self._cache_dir = Path(cache_dir)
        self._lock = Lock()
        #: Distinct requests, cache hit or not: a warm replay repeats the same
        #: request, so `api_calls` alone cannot size a cold pass but this can.
        self._requests: set[int] = set()
        self.stats.update(dict.fromkeys(_COUNTERS, 0))
        self.stats.update({"cost_usd": 0.0, "replayed_cost_usd": 0.0,
                           "temperature_enforced": True})

    @property
    def blocked_reason(self) -> str | None:
        """None when the arm can answer, live or from cache."""
        if self._client.inner is not None:
            return None
        if self._cache_dir.is_dir() and any(self._cache_dir.glob("*.json")):
            return None
        return self._backend_reason or "no LLM backend configured"

    def request(self, system: str, user: str, *, max_tokens: int, schema: dict[str, Any]) -> Any:
        """One completion, cached on disk and accounted for."""
        response = self._client.complete(
            system=system, user=user, model=self._model, temperature=self._temperature,
            max_tokens=int(max_tokens), schema=schema,
        )
        with self._lock:  # prefetch accounts from several threads
            self._requests.add(hash((system, user)))
            self.stats["unique_requests"] = len(self._requests)
            # Before the cache branch: the CLI cannot pin temperature, and a
            # replay must not report a guarantee the answer never had.
            if not response.temperature_enforced:
                self.stats["temperature_enforced"] = False
            if response.source == "cache":
                # A replay spends nothing, but what it replays had a price.
                self.stats["cache_hits"] = int(self.stats["cache_hits"]) + 1
                self.stats["replayed_prompt_tokens"] += int(response.prompt_tokens)
                self.stats["replayed_output_tokens"] += int(response.output_tokens)
                self.stats["replayed_cost_usd"] += float(response.cost_usd)
                return response
            self.stats["api_calls"] = int(self.stats["api_calls"]) + 1
            for key, value in (
                ("live_prompt_tokens", response.prompt_tokens),
                ("live_cache_creation_tokens", response.cache_creation_input_tokens),
                ("live_cache_read_tokens", response.cached_input_tokens),
                ("live_output_tokens", response.output_tokens),
            ):
                self.stats[key] = int(self.stats[key]) + int(value)
            self.stats["cost_usd"] = float(self.stats["cost_usd"]) + float(response.cost_usd)
        return response

    def _pending(self, queries: Sequence[str], k: int) -> list[tuple[str, list[str]]]:
        """De-duplicated queries with their candidate ids, resolved HERE.

        The store is a SQLite connection owned by the thread that opened it, so
        every k-NN lookup happens on this thread before the pool starts. Warming
        threads then only make HTTP calls, which is what they are for.
        """
        if not self._indexed or self._client.inner is None:
            return []
        want = min(int(k), len(self._traces))
        if want <= 0:
            return []
        return [(query, [tid for tid, _ in self._base.retrieve_scored(query, want)])
                for query in dict.fromkeys(queries) if isinstance(query, str)]

    def _run_warm(self, jobs: Sequence[Any], warm: Any, workers: int) -> int:
        """Warm `jobs` concurrently. Calls are independent and the cache is keyed
        by the exact request, so this changes the wall clock and nothing else."""
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            list(pool.map(warm, jobs))
        return len(jobs)

    def _count_failure(self) -> None:
        with self._lock:
            self.stats["prefetch_failures"] = int(self.stats.get("prefetch_failures", 0)) + 1


class LlmPickRetriever(LlmCandidateArm):
    """Arm A: one call per query, all k candidates in the prompt, a ranking out."""
    name = "llm_pick"

    def __init__(self, *, base: Any | None = None, cache: VectorCache | None = None,
                 client: Any | None = None, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, cache_dir: Path | str = PICK_CACHE_DIR,
                 name: str | None = None) -> None:
        super().__init__(
            base=base, cache=cache, client=client, model=model, temperature=temperature,
            cache_dir=cache_dir, name=name,
            params={"max_output_tokens": _MAX_OUTPUT_TOKENS, "calls_per_query": 1,
                    "output": "ranking of the k candidate ids, best first"},
        )
        self.stats["padded_slots"] = 0

    def _system(self, candidate_ids: Sequence[str]) -> str:
        return PICK_SYSTEM_HEADER + "\n\n".join(
            block_for(tid, self._texts[tid]) for tid in candidate_ids
        )

    @staticmethod
    def _user(query: str, k: int) -> str:
        return (
            f"QUESTION\n{query}\n\n"
            f"Return exactly {k} note ids, best first, most likely answer first. "
            "Copy the ids exactly as they appear between the double brackets. "
            "Rank on the specific details, not on the shared file or symptom."
        )

    @staticmethod
    def _schema(k: int) -> dict[str, Any]:
        # Exactly k, so the arm is scored on the same shape of output as every
        # other arm instead of on a short list padded from outside the model.
        ids = {"type": "array", "items": {"type": "string"}, "minItems": k, "maxItems": k}
        return {
            "type": "object",
            "properties": {"ranked_ids": ids},
            "required": ["ranked_ids"],
            "additionalProperties": False,
        }

    def _ask(self, query: str, allowed: Sequence[str]) -> Any:
        return self.request(
            self._system(allowed),
            self._user(query, len(allowed)),
            max_tokens=_MAX_OUTPUT_TOKENS,
            schema=self._schema(len(allowed)),
        )

    def _decide(self, query: str, candidate_ids: Sequence[str]) -> list[str]:
        allowed = list(candidate_ids)
        if not allowed:
            return []
        response = self._ask(query, allowed)
        known = set(allowed)
        ordered: list[str] = []
        seen: set[str] = set()
        for trace_id in parse_ids(response.text):
            if trace_id in known and trace_id not in seen:
                ordered.append(trace_id)
                seen.add(trace_id)
        if not ordered:
            self.stats["parse_failures"] = int(self.stats["parse_failures"]) + 1
        # Pad in BASE order, never by re-scoring: a short answer falls back to
        # the control's ranking, and the borrowed slots are counted.
        missing = [tid for tid in allowed if tid not in seen]
        if missing:
            self.stats["padded_slots"] = int(self.stats["padded_slots"]) + len(missing)
        return ordered + missing

    def prefetch(self, queries: Sequence[str], k: int, *, workers: int = 6) -> int:
        """Fill the response cache in parallel. Returned ids are unchanged."""
        jobs = [job for job in self._pending(queries, k) if job[1]]
        return self._run_warm(jobs, lambda job: self._warm(*job), workers)

    def _warm(self, query: str, ids: Sequence[str]) -> None:
        try:
            self._ask(query, list(ids))
        except LlmUnavailable:
            self._count_failure()


def _smoke() -> None:
    """Tiny path: python -m brain.baselines.veto.llm_pick."""
    from brain.baselines.smoke import run_smoke

    arm = LlmPickRetriever()
    if not arm.is_available():
        print(f"llm_pick BLOCKED: {arm.blocked_reason}")
        return
    with arm:
        run_smoke(arm, k=3)


if __name__ == "__main__":
    _smoke()
