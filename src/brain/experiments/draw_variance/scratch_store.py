"""The on-disk `{seed: draw_dict}` cache that makes a crashed live-LLM sweep
(F21, F23) resumable instead of a full re-pay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brain.config import ensure_dir
from brain.eval.results import jsonable

__all__ = ["ScratchStore"]


class ScratchStore:
    """One JSON file per statistic under a scratch directory: `{seed: draw_dict}`."""

    def __init__(self, directory: Path | str, statistic_id: str) -> None:
        self.path = ensure_dir(Path(directory)) / f"{statistic_id}.json"
        self._data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def get(self, seed: int) -> dict[str, Any] | None:
        return self._data.get(str(int(seed)))

    def put(self, seed: int, draw: dict[str, Any]) -> None:
        self._data[str(int(seed))] = draw
        self.path.write_text(json.dumps(jsonable(self._data), indent=2, sort_keys=True) + "\n")
