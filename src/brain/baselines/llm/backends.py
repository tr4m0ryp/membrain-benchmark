"""The two transports: the SDK (preferred) and the `claude -p` CLI (fallback).

1. `AnthropicClient` is preferred because it is the only one that can pin
   `temperature=0`, which the task contract requires, and because it marks
   the corpus block cacheable so 360 queries over the same 120 episodes are
   not 360 full-price prompts.
2. `ClaudeCliClient` is the fallback that works on a machine with a
   subscription and no API key, which is this machine. It CANNOT set
   temperature, so the arm records `temperature_enforced: false` when it
   runs this way. That is a disclosed limitation, not a silent one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from brain.baselines.llm.response import LlmResponse, LlmUnavailable

__all__ = ["AnthropicClient", "ClaudeCliClient", "select_client"]

_CLI_TIMEOUT_S = 300


class AnthropicClient:
    """Messages API at temperature 0, with the corpus block marked cacheable."""

    name = "anthropic-sdk"

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise LlmUnavailable("anthropic SDK not installed (pip install -e '.[llm]')") from exc
        self._client = anthropic.Anthropic()

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> LlmResponse:  # pragma: no cover - requires network and a key
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            # Prompt caching on the corpus block: every query in a run repeats
            # the same episodes, so this is a large discount, not a shortcut.
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user}],
        }
        if schema is not None:
            kwargs["tools"] = [
                {
                    "name": "submit_ranking",
                    "description": "Return the ranked note ids.",
                    "input_schema": schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "submit_ranking"}
        message = self._client.messages.create(**kwargs)
        if schema is not None:
            text = "".join(
                json.dumps(block.input) for block in message.content if block.type == "tool_use"
            )
        else:
            text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        return LlmResponse(
            text=text.strip(),
            input_tokens=int(getattr(usage, "input_tokens", 0)),
            output_tokens=int(getattr(usage, "output_tokens", 0)),
            cached_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            source="api",
            temperature_enforced=True,
        )


class ClaudeCliClient:
    """`claude -p` headless. Structured output via --json-schema.

    Customization is stripped (no tools, no MCP, no settings, no session
    persistence) so the arm is the model plus the prompt, not this machine's
    configuration.
    """

    name = "claude-cli"

    def __init__(self, binary: str | None = None, timeout_s: int = _CLI_TIMEOUT_S) -> None:
        found = binary or os.environ.get("BRAIN_CLAUDE_BIN") or shutil.which("claude")
        if not found:
            raise LlmUnavailable("claude CLI not found on PATH")
        self._binary = str(found)
        self._timeout_s = int(timeout_s)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> LlmResponse:  # pragma: no cover - requires the CLI and network
        args = [
            self._binary, "-p",
            "--model", model,
            "--output-format", "json",
            "--system-prompt", system,
            "--tools", "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
            "--setting-sources", "",
        ]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]
        try:
            done = subprocess.run(
                args, input=user, capture_output=True, text=True, timeout=self._timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            raise LlmUnavailable(f"claude CLI timed out after {self._timeout_s}s") from exc
        if done.returncode != 0:
            raise LlmUnavailable(f"claude CLI failed ({done.returncode}): {done.stderr.strip()[:400]}")
        try:
            payload = json.loads(done.stdout)
        except json.JSONDecodeError as exc:
            raise LlmUnavailable(f"claude CLI returned non-JSON output: {done.stdout[:200]}") from exc
        if payload.get("is_error"):
            raise LlmUnavailable(f"claude CLI error result: {str(payload.get('result'))[:400]}")
        usage = payload.get("usage") or {}
        return LlmResponse(
            text=str(payload.get("result", "")).strip(),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cached_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            # A large system prompt is reported here, not in input_tokens.
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
            cost_usd=float(payload.get("total_cost_usd", 0.0) or 0.0),
            source="cli",
            # The CLI exposes no temperature flag, so determinism rests on the
            # response cache instead. Recorded, never hidden.
            temperature_enforced=False,
        )


def select_client() -> tuple[Any | None, str | None]:
    """Return (client, reason_unavailable). SDK first, CLI second.

    `BRAIN_LLM_BACKEND` forces a choice (`anthropic`, `cli`, `none`) so a run
    can be pinned to one transport for reproducibility.
    """
    choice = (os.environ.get("BRAIN_LLM_BACKEND") or "auto").strip().lower()
    if choice == "none":
        return None, "BRAIN_LLM_BACKEND=none"
    reasons: list[str] = []
    if choice in {"auto", "anthropic"}:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        if not has_key and choice == "auto":
            reasons.append("no ANTHROPIC_API_KEY")
        else:
            try:
                return AnthropicClient(), None
            except LlmUnavailable as exc:
                reasons.append(str(exc))
    if choice in {"auto", "cli"}:
        try:
            return ClaudeCliClient(), None
        except LlmUnavailable as exc:
            reasons.append(str(exc))
    return None, "; ".join(reasons) or f"unknown backend {choice!r}"
