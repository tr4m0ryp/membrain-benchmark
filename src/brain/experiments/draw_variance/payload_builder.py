"""Turn per-seed draws (in scratch or freshly run) into the result payload.

A statistic with fewer than N draws is reported honestly as `pending` or
`in_progress` rather than padded, extrapolated, or silently summarized as if
N had been reached: rank and z-score are only ever computed over the draws
that actually ran (task hard constraint: never fabricate a number).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from brain.config import ensure_dir
from brain.eval.results import environment, jsonable
from brain.experiments.draw_variance import integrity
from brain.experiments.draw_variance.preregistration import SEEDS, build_preregistration
from brain.experiments.draw_variance.report_markdown import render_markdown
from brain.experiments.draw_variance.scratch_store import ScratchStore
from brain.experiments.draw_variance.stats_summary import summarize_draws

__all__ = ["SCRATCH_NAMES", "load_all_from_scratch", "assemble_payload", "write_report"]

#: statistic key (as used in `results`/`--only`) -> its ScratchStore filename.
SCRATCH_NAMES = {
    "f17a": "F17a_merge_harm_synthetic", "f17b": "F17b_merge_harm_relations_v2",
    "f23": "F23_consolidation_delta", "f21": "F21_memory_precision",
}


def load_all_from_scratch(scratch_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Every seed any process (this one or a concurrent sibling) has written so
    far, for every statistic -- so a checkpoint always reflects the combined
    state even when only a subset of statistics runs in THIS process."""
    out: dict[str, list[dict[str, Any]]] = {}
    for key, name in SCRATCH_NAMES.items():
        store = ScratchStore(scratch_dir, name)
        draws = [store.get(seed) for seed in SEEDS]
        out[key] = [d for d in draws if d is not None]
    return out


def _summary_or_progress(
    statistic_id: str, draws: list[dict[str, Any]], value_key: str, n_target: int
) -> dict[str, Any]:
    if len(draws) < 2:
        return {"status": "pending", "n_draws_so_far": len(draws), "n_target": n_target}
    seeds = [d["seed"] for d in draws]
    values = [d[value_key] for d in draws]
    summary = summarize_draws(statistic_id, seeds, values, frozen_seed=0).to_dict()
    summary["status"] = "complete" if len(draws) >= n_target else "in_progress"
    summary["n_target"] = n_target
    return summary


def assemble_payload(
    *, before: dict[str, str], results: dict[str, list[dict[str, Any]]],
    f21_blocked: str | None, started_at: str, command: str, n_target: int = len(SEEDS),
    verify_integrity: bool = True,
) -> dict[str, Any]:
    integrity_report = (
        integrity.assert_unchanged(before) if verify_integrity
        else {"status": "not yet verified (partial write, mid-run)"}
    )
    summaries: dict[str, Any] = {
        "F17a_merge_harm_synthetic": _summary_or_progress(
            "F17a_merge_harm_synthetic", results.get("f17a", []), "rate", n_target),
        "F17b_merge_harm_relations_v2": _summary_or_progress(
            "F17b_merge_harm_relations_v2", results.get("f17b", []), "rate", n_target),
        "F23_consolidation_delta": _summary_or_progress(
            "F23_consolidation_delta", results.get("f23", []), "delta_mean", n_target),
    }
    f21_draws = results.get("f21", [])
    if f21_blocked is None and f21_draws:
        summaries["F21_memory_precision"] = _summary_or_progress(
            "F21_memory_precision", f21_draws, "score", n_target)
    else:
        summaries["F21_memory_precision"] = {
            "status": "BLOCKED" if f21_blocked else "pending",
            "reason": f21_blocked, "partial_draws": f21_draws,
        }
    return {
        "schema": "draw-variance-result-v1",
        "experiment": "Draw-to-draw variance for F17, F21, F23 (F23 prioritised)",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "started_at": started_at, "command": command,
        "preregistration": build_preregistration(),
        "raw_draws": results,
        "summaries": summaries,
        "corpus_integrity": integrity_report,
        "environment": environment(),
    }


def write_report(payload: dict[str, Any], directory: Path | str, stamp: str) -> tuple[Path, Path]:
    """Writes (or overwrites, for the same `stamp`) the JSON and markdown report."""
    root = ensure_dir(Path(directory))
    json_path, md_path = root / f"draw-variance-{stamp}.json", root / f"draw-variance-{stamp}.md"
    json_path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    md_path.write_text(render_markdown(payload))
    return json_path, md_path
