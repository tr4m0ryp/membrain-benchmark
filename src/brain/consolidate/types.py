"""Records, parameters and accounting for the baseline consolidation pass.

Everything a reader has to agree with before a number produced by this package
means anything, except the prompt (`prompt.py`) and the algorithm
(`grouping.py`, `baseline.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LINKAGES",
    "DEFAULT_LINKAGE",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "MAX_SPEND_USD",
    "MergeGroup",
    "ConsolidationSpend",
    "ConsolidationResult",
    "price_usd",
]

#: Grouping linkages, from the most merge-happy to the most conservative.
LINKAGES: tuple[str, ...] = ("single", "average", "complete")
#: Complete linkage by default: every pair inside a merged group is itself above
#: tau, so the baseline never merges a pair its own policy did not judge similar.
#: `single` reproduces the naive transitive closure, which chains distant
#: episodes together and would inflate the measured harm in our favour.
DEFAULT_LINKAGE = "complete"

#: A common shipped merge threshold. F17 measures harm at exactly this value.
DEFAULT_THRESHOLD = 0.85
#: The swept grid. The threshold is a parameter, never a constant: task 005
#: reports consolidation delta as a curve over this sweep.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.95)

#: Hard stop from rules.md. A pass whose projection exceeds this is reported,
#: not spent.
MAX_SPEND_USD = 25.0

#: USD per million tokens, (input, output). Cache writes bill at 1.25x input and
#: cache reads at 0.1x input.
_PRICE_USD_PER_MTOK: dict[str, tuple[float, float]] = {"claude-haiku-4-5": (1.0, 5.0)}
_FALLBACK_PRICE = _PRICE_USD_PER_MTOK["claude-haiku-4-5"]


def price_usd(
    model: str,
    *,
    input_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """List-price cost of one call, used when the backend reports none.

    All three prompt buckets are priced. Summing only `input_tokens` reads as
    near-zero on a cache-aware backend, which has already shipped once on this
    project (rules.md, token accounting).
    """
    price_in, price_out = _PRICE_USD_PER_MTOK.get(model, _FALLBACK_PRICE)
    return (
        int(input_tokens) * price_in
        + int(cache_creation_tokens) * price_in * 1.25
        + int(cache_read_tokens) * price_in * 0.1
        + int(output_tokens) * price_out
    ) / 1e6


@dataclass(frozen=True, slots=True)
class MergeGroup:
    """One group the grouping step proposed, before any model was called."""

    member_ids: tuple[str, ...]
    score: float
    cluster_ids: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.member_ids)

    @property
    def spans_clusters(self) -> bool:
        """True when the group crosses corpus clusters, i.e. a coarse merge."""
        return len({c for c in self.cluster_ids if c}) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_ids": list(self.member_ids),
            "score": round(float(self.score), 6),
            "cluster_ids": list(self.cluster_ids),
            "spans_clusters": self.spans_clusters,
        }


@dataclass
class ConsolidationSpend:
    """Token and money accounting, live calls kept apart from cache replays."""

    api_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    replayed_prompt_tokens: int = 0
    temperature_enforced: bool = True

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read on live calls, all three buckets summed."""
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def cost_per_call_usd(self) -> float:
        return self.cost_usd / self.api_calls if self.api_calls else 0.0

    def add(self, response: Any, model: str) -> None:
        """Fold one response in. A cached replay costs nothing and is counted so."""
        # Checked before the cache short-circuit: a replayed answer was still
        # produced by a backend that could not pin temperature, and a fully
        # cached run must not report the limitation as absent.
        if not getattr(response, "temperature_enforced", True):
            self.temperature_enforced = False
        if getattr(response, "source", "") == "cache":
            self.cache_hits += 1
            self.replayed_prompt_tokens += int(response.prompt_tokens)
            return
        self.api_calls += 1
        self.input_tokens += int(response.input_tokens)
        self.cache_creation_tokens += int(response.cache_creation_input_tokens)
        self.cache_read_tokens += int(response.cached_input_tokens)
        self.output_tokens += int(response.output_tokens)
        # The CLI reports its own cost; the SDK does not, so fall back to list
        # price over the same three buckets rather than recording zero.
        self.cost_usd += float(response.cost_usd) or price_usd(
            model,
            input_tokens=response.input_tokens,
            cache_creation_tokens=response.cache_creation_input_tokens,
            cache_read_tokens=response.cached_input_tokens,
            output_tokens=response.output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "live_prompt_tokens": self.prompt_tokens,
            "live_input_tokens": self.input_tokens,
            "live_cache_creation_tokens": self.cache_creation_tokens,
            "live_cache_read_tokens": self.cache_read_tokens,
            "live_output_tokens": self.output_tokens,
            "replayed_prompt_tokens": self.replayed_prompt_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_per_call_usd": round(self.cost_per_call_usd, 6),
            "temperature_enforced": self.temperature_enforced,
        }


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """A consolidated store plus everything needed to attribute a number to it.

    `store` reads exactly like the store that went in: same schema, same
    `knn` / `all_traces` / `get`. `traces` is the post-pass valid set, which is
    what a retrieval arm should be indexed on.
    """

    store: Any
    traces: list[Any]
    groups: list[MergeGroup] = field(default_factory=list)
    retired_ids: tuple[str, ...] = ()
    merged_ids: tuple[str, ...] = ()
    #: original episode id -> the consolidated episode standing for it, so a
    #: downstream metric can credit a merge instead of scoring an id change.
    replacements: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    spend: ConsolidationSpend = field(default_factory=ConsolidationSpend)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def trace_ids(self) -> list[str]:
        return [trace.id for trace in self.traces]

    def describe(self) -> dict[str, Any]:
        """Same shape as a baseline arm's `describe`, so results files match."""
        return {
            "name": "consolidate_baseline",
            "params": dict(self.params),
            "stats": dict(self.stats),
            "spend": self.spend.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
        }
