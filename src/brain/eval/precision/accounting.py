"""Token and cost accounting for the judge, and the pre-spend extrapolation.

Split out of `entailment` because it is the part that has already been wrong on
this project and deserves to be readable on one page.

**Prompt tokens arrive in three buckets and all three are prompt.** A large
prompt sent through a cache-aware backend reports `input_tokens = 10` and
`cache_creation_input_tokens = 20953`; summing only the first under-reports the
work by three orders of magnitude and once produced a near-zero cost figure
here. `prompt_tokens` sums all three, and nothing else is reported as the
prompt size.

**Two cost numbers, never one.** The CLI backend reports its own
`total_cost_usd` per call; the SDK backend reports none. Carrying only the
reported figure would make an SDK run look free, and carrying only the estimate
would ignore the backend's own accounting. Both are kept, `cost_usd` prefers
the reported one, and it falls back to the price table rather than to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

__all__ = ["PRICE_USD_PER_MTOK", "JudgeStats", "extrapolate_cost"]

#: claude-haiku-4-5 list price, USD per million tokens. Used for the independent
#: estimate and the pre-spend extrapolation only; the reported spend prefers the
#: backend's own number when it gives one.
PRICE_USD_PER_MTOK: dict[str, float] = {
    "input": 1.00,
    "output": 5.00,
    "cache_write": 1.25,
    "cache_read": 0.10,
}


@dataclass(slots=True)
class JudgeStats:
    """Calls, tokens and cost for one judge. Safe to update from several threads."""

    api_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    reported_cost_usd: float = 0.0
    parse_failures: int = 0
    temperature_enforced: bool = True
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read: fresh, cache-written and cache-read."""
        return int(self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens)

    @property
    def estimated_cost_usd(self) -> float:
        """Price-table estimate, independent of what the backend reported."""
        return (
            self.input_tokens * PRICE_USD_PER_MTOK["input"]
            + self.cache_creation_tokens * PRICE_USD_PER_MTOK["cache_write"]
            + self.cache_read_tokens * PRICE_USD_PER_MTOK["cache_read"]
            + self.output_tokens * PRICE_USD_PER_MTOK["output"]
        ) / 1_000_000.0

    @property
    def cost_usd(self) -> float:
        """Reported spend, falling back to the estimate when none was reported."""
        reported = float(self.reported_cost_usd)
        return reported if reported > 0.0 else self.estimated_cost_usd

    def record(self, response: Any) -> None:
        """Account for one response. A cache replay costs nothing and is counted apart."""
        with self._lock:  # judge_many accounts from several threads
            if response.source == "cache":
                self.cache_hits += 1
                return
            self.api_calls += 1
            self.input_tokens += int(response.input_tokens)
            self.cache_creation_tokens += int(response.cache_creation_input_tokens)
            self.cache_read_tokens += int(response.cached_input_tokens)
            self.output_tokens += int(response.output_tokens)
            self.reported_cost_usd += float(response.cost_usd)
            if not response.temperature_enforced:
                self.temperature_enforced = False

    def note_parse_failure(self) -> None:
        with self._lock:
            self.parse_failures += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "reported_cost_usd": round(self.reported_cost_usd, 6),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "cost_usd": round(self.cost_usd, 6),
            "parse_failures": self.parse_failures,
            "temperature_enforced": self.temperature_enforced,
        }


#: Prior for a judge call when the smoke was served entirely from cache, so
#: nothing live was measured. Rounded up from 234 cold calls at 0.00518 USD
#: each (claude-haiku-4-5 through the CLI, 2026-07-25). A cached smoke must not
#: project a cost of zero: the pass it is gating may still be mostly cold, and a
#: zero projection would disable the budget check exactly when it is needed.
PRIOR_COST_PER_CALL_USD = 0.0055
_PRIOR_PROMPT_TOKENS = 1400
_PRIOR_OUTPUT_TOKENS = 600


def extrapolate_cost(stats: JudgeStats, remaining_calls: int) -> dict[str, Any]:
    """Cost and tokens for `remaining_calls` more live calls.

    Measured from the smoke when the smoke made live calls, and from the
    recorded prior when it did not. `source` says which, so a reader can tell a
    measurement from an assumption.
    """
    live = max(int(stats.api_calls), 0)
    n = max(int(remaining_calls), 0)
    if live == 0:
        return {
            "source": "prior",
            "per_call_usd": PRIOR_COST_PER_CALL_USD,
            "remaining_calls": n,
            "projected_usd": round(PRIOR_COST_PER_CALL_USD * n, 4),
            "projected_prompt_tokens": _PRIOR_PROMPT_TOKENS * n,
            "projected_output_tokens": _PRIOR_OUTPUT_TOKENS * n,
        }
    per_call = stats.cost_usd / live
    return {
        "source": "measured",
        "per_call_usd": round(per_call, 6),
        "remaining_calls": n,
        "projected_usd": round(per_call * n, 4),
        "projected_prompt_tokens": int(stats.prompt_tokens / live * n),
        "projected_output_tokens": int(stats.output_tokens / live * n),
    }
