"""LLM transport for the full-context arm: two backends and a disk cache.

Split out of the arm so its logic (budget, prompt, parsing) is readable without
the transport underneath it. Nothing else in `brain` talks to an LLM.
"""

from brain.baselines.llm.backends import AnthropicClient, ClaudeCliClient, select_client
from brain.baselines.llm.cache import CachedClient
from brain.baselines.llm.response import (
    DEFAULT_MODEL,
    LlmClient,
    LlmResponse,
    LlmUnavailable,
)

__all__ = [
    "DEFAULT_MODEL",
    "LlmClient",
    "LlmResponse",
    "LlmUnavailable",
    "AnthropicClient",
    "ClaudeCliClient",
    "CachedClient",
    "select_client",
]
