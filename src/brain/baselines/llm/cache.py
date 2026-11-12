"""Disk cache in front of any backend.

Every response is cached under a hash of the exact request, so a rerun of
the same seeds replays byte-identical answers. That is what makes an LLM arm
reproducible at all, and it is why cached replays are counted separately
from live calls: a report can then state both what the run cost and what a
cold run would cost.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from brain.baselines.llm.response import LlmResponse, LlmUnavailable

__all__ = ["CachedClient"]


class CachedClient:
    """Disk cache in front of any client. A miss without a client is fatal."""

    def __init__(self, inner: Any | None, cache_dir: Path | str) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self.name = f"cached({getattr(inner, 'name', 'none')})"
        self.hits = 0
        self.calls = 0

    @property
    def inner(self) -> Any | None:
        """The live backend, or None when only cached replay is possible."""
        return self._inner

    @property
    def cache_dir(self) -> Path:
        return self._dir

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> LlmResponse:
        digest = self.key(
            {
                "model": model,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "schema": schema,
                "system": system,
                "user": user,
            }
        )
        path = self._dir / f"{digest}.json"
        if path.is_file():
            self.hits += 1
            return LlmResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if self._inner is None:
            raise LlmUnavailable(f"no LLM backend and no cached answer ({digest[:12]})")
        response = self._inner.complete(
            system=system,
            user=user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            schema=schema,
        )
        self.calls += 1
        self._dir.mkdir(parents=True, exist_ok=True)
        # Write then rename: prefetch calls this from several threads, and a
        # half-written cache entry would poison every later run.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(response.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return response


