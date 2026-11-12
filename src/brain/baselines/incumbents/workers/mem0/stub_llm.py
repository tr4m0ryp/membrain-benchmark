"""A Mem0 LLM provider that answers from a judge instead of from a model.

Registered into `mem0.utils.factory.LlmFactory` under the provider name
``brain_stub``. Mem0 constructs it exactly as it constructs any other provider,
and every call goes through Mem0's real prompt building and its real response
parsing; only the transport is replaced. That is what makes the probe a
measurement of Mem0's write path rather than of a model.

The judge is a process-wide singleton because the factory owns construction and
gives us no hook to pass one in.
"""

from __future__ import annotations

from typing import Any

#: Set by the driver before `Memory` is constructed. See `judges.py`.
ACTIVE: Any = None


class MissingJudge(RuntimeError):
    """Mem0 called the LLM before a judge was installed."""


class BrainStubLLM:
    """Duck-typed Mem0 LLM provider. No network, no key, no model."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        response_format: Any = None,
        tools: Any = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> str:
        if ACTIVE is None:
            raise MissingJudge("no judge installed; set stub_llm.ACTIVE before Memory()")
        return ACTIVE.respond(list(messages or []), response_format)


#: Slots tried, in order. Mem0 ships `LlmFactory.register_provider`, but its
#: `LlmConfig` validator checks the provider name against a hard-coded tuple in
#: a function body, so a genuinely new name cannot be registered without editing
#: Mem0. An existing slot is taken over instead, and which one is recorded in
#: the result file. `langchain` is preferred because it is Mem0's own
#: bring-your-own-LLM-object slot, so nothing about the write path changes.
_SLOTS = ("langchain", "litellm", "openai")


def install(judge: Any) -> str:
    """Make `judge` active, claim a provider slot, and return the slot name."""
    global ACTIVE
    ACTIVE = judge

    from mem0.llms.configs import LlmConfig
    from mem0.utils.factory import LlmFactory

    entry: Any
    try:  # v2 factory: (class path, config class)
        from mem0.configs.llms.base import BaseLlmConfig

        entry = ("stub_llm.BrainStubLLM", BaseLlmConfig)
    except ImportError:  # v0.1 factory: class path only
        entry = "stub_llm.BrainStubLLM"

    for slot in _SLOTS:
        if slot not in LlmFactory.provider_to_class:
            continue
        try:
            LlmConfig(provider=slot, config={})
        except Exception:
            continue
        LlmFactory.provider_to_class[slot] = entry
        return slot
    raise RuntimeError(f"no Mem0 provider slot could be claimed; tried {_SLOTS}")
