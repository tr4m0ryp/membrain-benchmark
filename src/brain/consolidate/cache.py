"""Disk cache for merge responses, in the style of `data/cache/full_context/`.

One JSON file per sha256, written to a temporary name and renamed into place so
a half-written entry can never poison a later run. The difference from
`baselines.llm.CachedClient` is where the key comes from: this one takes the
digest from the caller, which is what lets the key carry the group identity, the
prompt version and the seed rule (`prompt.merge_cache_key`) rather than only the
rendered request.

A re-run of the same groups is therefore free, which is the property the LLM
cost discipline in rules.md depends on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from brain.baselines.llm import LlmResponse, LlmUnavailable

__all__ = ["MergeCache"]


class MergeCache:
    """Disk cache in front of any backend. A miss without a client is fatal."""

    def __init__(self, inner: Any | None, cache_dir: Path | str) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self._lock = Lock()  # `prefetch` calls this from several threads
        self.hits = 0
        self.calls = 0

    def __repr__(self) -> str:
        return f"MergeCache(inner={getattr(self._inner, 'name', None)!r}, dir={self._dir})"

    @property
    def inner(self) -> Any | None:
        """The live backend, or None when only cached replay is possible."""
        return self._inner

    @property
    def cache_dir(self) -> Path:
        return self._dir

    def has(self, digest: str) -> bool:
        return (self._dir / f"{digest}.json").is_file()

    def complete(self, *, digest: str, **request: Any) -> LlmResponse:
        """Replay `digest` if it is on disk, otherwise call and record it."""
        path = self._dir / f"{digest}.json"
        if path.is_file():
            with self._lock:
                self.hits += 1
            return LlmResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if self._inner is None:
            raise LlmUnavailable(f"no LLM backend and no cached merge ({digest[:12]})")
        response = self._inner.complete(**request)
        with self._lock:
            self.calls += 1
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(response.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return response
