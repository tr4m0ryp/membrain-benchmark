"""Response record and client contract for the full-context arm.

Prompt tokens arrive in three buckets and all three are prompt. Keeping
them separate here, and summing them in one place, is what stops an arm
from reporting a 20k-token context as 10 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["DEFAULT_MODEL", "LlmUnavailable", "LlmResponse", "LlmClient"]

DEFAULT_MODEL = "claude-haiku-4-5"


class LlmUnavailable(RuntimeError):
    """No usable backend and no cached answer for this request."""


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """One completion plus the accounting the task contract demands.

    Prompt tokens arrive in three buckets and ALL of them are prompt: a 20k
    corpus block sent through a cache-aware backend reports `input_tokens = 10`
    and `cache_creation_input_tokens = 20953`. Summing only the first would
    under-report the arm's context by three orders of magnitude, so
    `prompt_tokens` is what gets reported and the buckets are kept for cost.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    source: str = "api"  # api | cli | cache | stub
    temperature_enforced: bool = True

    @property
    def prompt_tokens(self) -> int:
        """Everything the model actually read: fresh, cache-written and cache-read."""
        return (
            int(self.input_tokens)
            + int(self.cached_input_tokens)
            + int(self.cache_creation_input_tokens)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "cached_input_tokens": int(self.cached_input_tokens),
            "cache_creation_input_tokens": int(self.cache_creation_input_tokens),
            "cost_usd": float(self.cost_usd),
            "source": self.source,
            "temperature_enforced": bool(self.temperature_enforced),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LlmResponse":
        return cls(
            text=str(raw["text"]),
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
            cached_input_tokens=int(raw.get("cached_input_tokens", 0)),
            cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens", 0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
            source="cache",
            temperature_enforced=bool(raw.get("temperature_enforced", True)),
        )


class LlmClient(Protocol):  # pragma: no cover - structural typing only
    name: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> LlmResponse: ...

