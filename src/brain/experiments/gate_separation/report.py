"""The gate's two artifacts: `results/gate-<stamp>.json` and `gate-<stamp>.md`.

The JSON is the record (every arm, every seed, every interval, the provenance
that ties them to a corpus by content hash). The markdown is the same numbers
laid out for a reader, headed by the verdict line so it cannot be missed.

The headline verdict is deliberately the one computed at the BEST-performing
sweep configuration. That is the most generous reading available to the
mechanism, so a killed headline is conservative: the mechanism failed even at
the setting picked, with hindsight, to suit it.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from brain.config import ensure_dir
from brain.eval.results import corpus_provenance, environment, jsonable
from brain.eval.verdict import REQUIRED_BASELINES
from brain.experiments.gate_separation.gate import GateRun
from brain.experiments.gate_separation.markdown import render_markdown

__all__ = [
    "VERDICT_LINES",
    "unavailable_arms",
    "verdict_line",
    "build_payload",
    "write_gate_report",
]

SCHEMA = "gate-result-v1"

VERDICT_LINES = {
    "survives": "VERDICT: H-A SURVIVES",
    "killed": "VERDICT: H-A KILLED",
}


def unavailable_arms(run: GateRun) -> list[str]:
    """Arms the criterion names that produced no number, blocked or never run.

    Not run and blocked are the same thing for the verdict: either way the
    pre-registered comparison could not be made.
    """
    missing = [name for name in REQUIRED_BASELINES if name not in run.results]
    return sorted(set(missing) | set(run.blocked))


def verdict_line(run: GateRun) -> str:
    """The one line the spec asks the gate to print."""
    verdict = run.verdicts.get("selected")
    if verdict is not None and verdict.status in VERDICT_LINES:
        return VERDICT_LINES[verdict.status]
    arms = unavailable_arms(run)
    if verdict is None and not arms:
        arms = ["no separation arm produced a result"]
    return f"VERDICT: INCONCLUSIVE (blocked arms: {arms or ['none; see the reasons below']})"


def _sweep_rows(run: GateRun) -> list[dict[str, Any]]:
    control = run.results.get("plain_vector")
    geometry = {row["arm"]: row for row in run.diagnostics.get("stored_geometry", []) or []}
    rows: list[dict[str, Any]] = []
    for name in run.sweep:
        result = run.results[name]
        config = dict(run.specs[name].config)
        geo = geometry.get(name, {})
        rows.append(
            {
                "arm": name,
                "mode": config.get("mode"),
                "s": config.get("s"),
                "aggregation": config.get("aggregation"),
                "calibrated_tau": result.describe.get("params", {}).get("tau"),
                "disambiguation": float(result.disambiguation),
                "ci_lower": float(result.bootstrap.lower),
                "ci_upper": float(result.bootstrap.upper),
                "delta_vs_plain_vector_points": (
                    None
                    if control is None
                    else round((result.disambiguation - control.disambiguation) * 100.0, 4)
                ),
                "accuracy_by_cluster_size": {
                    str(size): round(point.accuracy, 6)
                    for size, point in sorted(result.curve.items())
                },
                "decays": bool(result.decay.get("decays", False)),
                "sibling_cosine": geo.get("sibling_cosine"),
                "non_sibling_cosine": geo.get("non_sibling_cosine"),
                "contrast": geo.get("contrast"),
                "separating_writes": geo.get("separating_writes"),
            }
        )
    return rows


def build_payload(run: GateRun, *, command: str) -> dict[str, Any]:
    """Everything the run produced, in one JSON-safe record."""
    manifest = getattr(run.corpus, "manifest", {}) or {}
    verdict = run.verdicts.get("selected")
    return {
        "schema": SCHEMA,
        "experiment": "Task 008 -- minimal falsifying experiment for H-A (THE GATE)",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "verdict_line": verdict_line(run),
        "verdict_status": None if verdict is None else verdict.status,
        "unavailable_required_arms": unavailable_arms(run),
        "verdicts": {name: value.to_dict() for name, value in sorted(run.verdicts.items())},
        "selection": run.selection,
        "sweep_verdicts": run.sweep_verdicts,
        "config": {
            "k": run.k,
            "seeds": list(run.seeds),
            "primary_aggregation": run.diagnostics.get("primary_aggregation"),
            "aggregation_note": (
                "sequential is the literal reading of M-A and the recorded spec decision; "
                "net is reported as a secondary diagnostic only"
            ),
            "tau_note": (
                "tau is quoted in raw embedding units and calibrated into each stored "
                "geometry, so the arms are compared at the same collision CRITERION, not "
                "at the same number"
            ),
        },
        "corpus": {
            # corpus_provenance already carries the scalars (hash, generator
            # name, similarity setting); the bulky manifest blocks go beside
            # them under their own keys so a reader is not handed a wall of
            # diagnostics where a hash was expected.
            **corpus_provenance(run.corpus),
            "generator_params": manifest.get("generator"),
            "counts": manifest.get("counts"),
            "similarity_diagnostics": manifest.get("similarity"),
        },
        "arms": {name: result.to_dict() for name, result in sorted(run.results.items())},
        "blocked_arms": dict(sorted(run.blocked.items())),
        "sweep": _sweep_rows(run),
        "accuracy_by_cluster_size": {
            name: {str(size): point.to_dict() for size, point in sorted(result.curve.items())}
            for name, result in sorted(run.results.items())
        },
        "diagnostics": run.diagnostics,
        "cost_usd_reported": round(run.cost_usd(), 4),
        "llm_accounting": run.llm_accounting(),
        "wall_clock_seconds": round(float(run.wall_clock), 2),
        "environment": environment(),
    }


def write_gate_report(
    run: GateRun, *, command: str, directory: Path | str
) -> tuple[Path, Path, dict[str, Any]]:
    """Write both artifacts under one timestamp and return their paths."""
    payload = build_payload(run, command=command)
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"gate-{stamp}.json"
    md_path = root / f"gate-{stamp}.md"
    suffix = 1
    while json_path.exists() or md_path.exists():
        json_path = root / f"gate-{stamp}-{suffix}.json"
        md_path = root / f"gate-{stamp}-{suffix}.md"
        suffix += 1
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(run, payload), encoding="utf-8")
    return json_path, md_path, payload
