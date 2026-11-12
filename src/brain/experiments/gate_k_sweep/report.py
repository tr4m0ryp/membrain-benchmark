"""Write `results/k-sweep/ksweep-<UTC>.json` and the matching `.md`.

Mirrors `eval/results.py`'s conventions (UTC second-resolution stamp, sorted
keys, non-finite floats mapped to `null` via `jsonable`) without importing its
arm-named `write_result` helper, whose filename shape (`<stamp>-<arm>.json`)
does not fit a multi-arm sweep report.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from brain.config import ensure_dir
from brain.eval.results import environment, jsonable

from brain.experiments.gate_k_sweep.markdown import render_markdown
from brain.experiments.gate_k_sweep.sweep import SweepResult

__all__ = ["write_sweep_report"]


def write_sweep_report(
    result: SweepResult, *, command: str, directory: Path | str
) -> tuple[Path, Path, dict[str, Any]]:
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"ksweep-{stamp}.json"
    md_path = root / f"ksweep-{stamp}.md"
    suffix = 1
    while json_path.exists() or md_path.exists():
        json_path = root / f"ksweep-{stamp}-{suffix}.json"
        md_path = root / f"ksweep-{stamp}-{suffix}.md"
        suffix += 1

    payload = result.to_dict()
    payload["command"] = command
    payload["environment"] = environment()

    body = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    json_path.write_text(body + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    return json_path, md_path, payload
