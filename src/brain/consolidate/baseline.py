"""The consolidation pass itself: group, merge with one LLM call, write a copy.

Grouping lives in `grouping.py`, the prompt in `prompt.py`, the response cache
in `cache.py`, the store writes in `writer.py`. What is here is the policy.

The source store is READ-ONLY throughout: consolidation always builds a new
`EpisodicStore`, so the frozen corpus and any store built from it survive the
pass byte-identical. `destructive=True` retires each group's members;
`destructive=False` leaves them valid and only adds the summary, which is what
lets task 005 attribute a measured loss to merging rather than to summarising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from brain.baselines.llm import DEFAULT_MODEL, LlmResponse, select_client
from brain.baselines.vectors import VectorCache, shared_vector_cache
from brain.config import DATA_DIR, SEED
from brain.consolidate.cache import MergeCache
from brain.consolidate.grouping import groups_for
from brain.consolidate.prompt import (
    MAX_OUTPUT_TOKENS,
    PROMPT_VERSION,
    block_for,
    merge_cache_key,
    merge_schema,
    merge_system_prompt,
    merge_user_prompt,
    parse_merged,
)
from brain.consolidate.types import (
    DEFAULT_LINKAGE,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    MAX_SPEND_USD,
    ConsolidationResult,
    ConsolidationSpend,
    MergeGroup,
)
from brain.consolidate.writer import (
    count_collapsed,
    merged_trace,
    replacement_map,
    retire_members,
    write_merges,
)
from brain.store import EpisodicStore, Trace

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ConsolidationBudgetExceeded",
    "BaselineConsolidator",
    "consolidate_baseline",
    "run_consolidation",
    "sweep_thresholds",
]

#: Cached merges live next to the corpus, so a re-run is offline and identical.
DEFAULT_CACHE_DIR = DATA_DIR / "cache" / "consolidate"


class ConsolidationBudgetExceeded(RuntimeError):
    """Live spend passed the budget. Raised instead of quietly continuing."""


class BaselineConsolidator:
    """Similarity grouping plus one merge call per group, onto a copy."""

    name = "consolidate_baseline"

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        destructive: bool = True,
        linkage: str = DEFAULT_LINKAGE,
        seed: int = SEED,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        vectors: VectorCache | None = None,
        max_groups: int | None = None,
        budget_usd: float = MAX_SPEND_USD,
    ) -> None:
        backend, reason = (client, None) if client is not None else select_client()
        self._threshold = float(threshold)
        self._destructive = bool(destructive)
        self._linkage = str(linkage)
        self._seed = int(seed)
        self._model = str(model)
        self._temperature = float(temperature)
        self._budget = float(budget_usd)
        self._max_groups = None if max_groups is None else int(max_groups)
        self._client = MergeCache(backend, cache_dir)
        self._backend_reason = reason
        self._vectors = vectors
        self.spend = ConsolidationSpend()
        self.params: dict[str, Any] = {
            "threshold": self._threshold,
            "destructive": self._destructive,
            "linkage": self._linkage,
            "seed": self._seed,
            "model": self._model,
            "temperature": self._temperature,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "prompt_version": PROMPT_VERSION,
            "backend": getattr(backend, "name", None),
            "cache_dir": str(cache_dir),
            # False at temperature 0 on purpose: the request cannot depend on
            # the seed there, so keying on it would cost 5x for 5 seeds.
            "seed_in_cache_key": self._temperature > 0.0,
        }
        self.stats: dict[str, Any] = {
            "parse_failures": 0,
            "schema_bypassed": 0,
            "empty_merges": 0,
            "collapsed_merges": 0,
        }

    def __repr__(self) -> str:
        return (
            f"BaselineConsolidator(threshold={self._threshold}, "
            f"destructive={self._destructive}, linkage={self._linkage!r})"
        )

    @property
    def blocked_reason(self) -> str | None:
        """None when the pass can run, live or entirely from cache."""
        if self._client.inner is not None:
            return None
        if self._client.cache_dir.is_dir() and any(self._client.cache_dir.glob("*.json")):
            return None
        return self._backend_reason or "no LLM backend configured"

    def is_available(self) -> bool:
        return self.blocked_reason is None

    # -- main entry point --------------------------------------------------

    def run(
        self, store: EpisodicStore, *, db_path: str | Path = ":memory:"
    ) -> ConsolidationResult:
        """Consolidate into a NEW store. `store` is never written to."""
        source = store.all_traces(valid_only=False)
        live = [trace for trace in source if trace.is_valid]
        by_id = {trace.id: trace for trace in live}
        groups = groups_for(
            live,
            store.get_vectors([trace.id for trace in live]),
            threshold=self._threshold,
            linkage=self._linkage,
        )
        if self._max_groups is not None:
            groups = groups[: self._max_groups]

        merged: list[tuple[MergeGroup, Trace]] = []
        for group in groups:
            text = self._merge(group, by_id)
            if not text:
                self.stats["empty_merges"] = int(self.stats["empty_merges"]) + 1
                continue
            members = [by_id[i] for i in group.member_ids]
            merged.append((group, merged_trace(group, members, text)))

        new_store = EpisodicStore(db_path, store.dim)
        new_store.add_many(source, store.get_vectors([trace.id for trace in source]))
        cache = self._vectors if self._vectors is not None else shared_vector_cache()
        written = write_merges(new_store, merged, cache)
        if self._destructive:
            retired, collapsed = retire_members(new_store, merged, written)
        else:
            retired, collapsed = [], count_collapsed(merged, written)
        self.stats["collapsed_merges"] = collapsed

        traces = new_store.all_traces(valid_only=True)
        self.stats.update(
            {
                "n_source": len(source),
                "n_source_valid": len(live),
                "n_groups": len(groups),
                "n_grouped_episodes": sum(group.size for group in groups),
                "n_merged": len(merged),
                "n_retired": len(retired),
                "n_after": len(traces),
                "groups_spanning_clusters": sum(1 for g in groups if g.spans_clusters),
                "clusters_touched": len({c for g in groups for c in g.cluster_ids if c}),
                "api_calls": self._client.calls,
                "cache_hits": self._client.hits,
            }
        )
        return ConsolidationResult(
            store=new_store,
            traces=traces,
            groups=groups,
            retired_ids=tuple(sorted(retired)),
            merged_ids=tuple(sorted(written)),
            replacements=replacement_map(merged, written),
            params=dict(self.params),
            stats=dict(self.stats),
            spend=self.spend,
        )

    # -- one merge ---------------------------------------------------------

    def _merge(self, group: MergeGroup, by_id: dict[str, Trace]) -> str:
        blocks = [block_for(i, by_id[i].text) for i in group.member_ids]
        system = merge_system_prompt(blocks)
        raw = self._request(group, by_id, system, merge_user_prompt(len(blocks))).text
        text = parse_merged(raw)
        if not text:
            self.stats["parse_failures"] = int(self.stats["parse_failures"]) + 1
            raw = self._request(
                group, by_id, system, merge_user_prompt(len(blocks), repair=True)
            ).text
            text = parse_merged(raw)
        if text and not raw.strip().startswith("{"):
            self.stats["schema_bypassed"] = int(self.stats["schema_bypassed"]) + 1
        return text

    def _request(
        self, group: MergeGroup, by_id: dict[str, Trace], system: str, user: str
    ) -> LlmResponse:
        digest = merge_cache_key(
            member_ids=group.member_ids,
            member_shas=[by_id[i].content_sha for i in group.member_ids],
            model=self._model,
            temperature=self._temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            seed=self._seed,
            system=system,
            user=user,
        )
        response = self._client.complete(
            digest=digest,
            system=system,
            user=user,
            model=self._model,
            temperature=self._temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            schema=merge_schema(),
        )
        self.spend.add(response, self._model)
        if self.spend.cost_usd > self._budget:
            raise ConsolidationBudgetExceeded(
                f"live spend {self.spend.cost_usd:.2f} USD passed the "
                f"{self._budget:.2f} USD budget after {self.spend.api_calls} calls"
            )
        return response


# -- public surface (task 005 consumes these) -----------------------------

def run_consolidation(
    store: EpisodicStore,
    seed: int = SEED,
    *,
    db_path: str | Path = ":memory:",
    **kwargs: Any,
) -> ConsolidationResult:
    """Full result: the new store, its traces, the groups, the stats and spend."""
    return BaselineConsolidator(seed=seed, **kwargs).run(store, db_path=db_path)


def consolidate_baseline(
    store: EpisodicStore,
    seed: int = SEED,
    *,
    db_path: str | Path = ":memory:",
    **kwargs: Any,
) -> EpisodicStore:
    """`consolidate_baseline(store, seed) -> new_store`, the task 003 contract.

    The returned store has the unchanged read interface: same schema, same
    `knn` / `all_traces` / `get`. Use `run_consolidation` when the groups, the
    stats or the spend are needed as well.
    """
    return run_consolidation(store, seed, db_path=db_path, **kwargs).store


def sweep_thresholds(
    store: EpisodicStore,
    seed: int = SEED,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    **kwargs: Any,
) -> dict[float, ConsolidationResult]:
    """One result per threshold. Every mode and seed replays the same merges."""
    return {
        float(tau): run_consolidation(store, seed, threshold=float(tau), **kwargs)
        for tau in sorted({float(t) for t in thresholds})
    }


if __name__ == "__main__":  # pragma: no cover - manual entry point
    from brain.consolidate.smoke import main

    main()
