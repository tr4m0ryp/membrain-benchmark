"""The result record, its provenance, and its on-disk form.

A number without its origin is not a result, so `ArmResult` carries what was
measured, on which corpus BYTES, with which arm parameters, under which seeds,
and in which environment. The corpus content hash is the load-bearing field: it
makes it impossible to attribute a number to a corpus that has since drifted,
which is the failure the frozen-corpus rule exists to prevent.

Non-finite floats are written as `null` rather than as bare `NaN`. `NaN` is not
JSON, and a file that only some parsers accept is not an artifact.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from brain.config import SEED, ensure_dir
from brain.eval.curve import curve_to_dict
from brain.eval.metrics import QueryOutcome
from brain.eval.stats import Interval
from brain.eval.verdict import ArmStats

__all__ = [
    "RESULT_SCHEMA",
    "SeedRun",
    "ArmResult",
    "corpus_provenance",
    "environment",
    "jsonable",
    "write_result",
]

#: Bumped when the result file shape changes.
RESULT_SCHEMA = "eval-result-v1"


@dataclass(frozen=True, slots=True)
class SeedRun:
    """One arm, one seed."""

    seed: int
    metrics: dict[str, float]
    outcomes: list[QueryOutcome]
    index_seconds: float
    retrieve_seconds: float
    prefetched: int = 0
    arm_stats: dict[str, Any] = field(default_factory=dict)
    describe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "metrics": dict(self.metrics),
            "index_seconds": round(float(self.index_seconds), 4),
            "retrieve_seconds": round(float(self.retrieve_seconds), 4),
            "prefetched_queries": int(self.prefetched),
            "arm_stats": dict(self.arm_stats),
        }


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Per-seed values plus the aggregates, and the provenance they hold for."""

    arm: str
    k: int
    seeds: tuple[int, ...]
    seeds_run: tuple[int, ...]
    seed_independent: bool
    runs: tuple[SeedRun, ...]
    aggregate: dict[str, Interval]
    bootstrap: Interval
    clusters: dict[str, list[float]]
    curve: dict[int, Any]
    decay: dict[str, Any]
    describe: dict[str, Any]
    corpus: dict[str, Any]
    config: dict[str, Any]
    wall_clock: float
    path: Path | None = None

    @property
    def disambiguation(self) -> float:
        """Mean disambiguation accuracy across the seeds that ran."""
        return float(self.bootstrap.mean)

    def stats(self) -> ArmStats:
        """The shape `evaluate_kill_criterion` consumes (cluster interval, not seed)."""
        return ArmStats(
            name=self.arm,
            accuracy=self.disambiguation,
            ci=self.bootstrap,
            clusters=self.clusters,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "arm": self.arm,
            "k": int(self.k),
            "seeds_declared": list(self.seeds),
            "seeds_run": list(self.seeds_run),
            "seed_independent": bool(self.seed_independent),
            "metric_definitions": {
                "disambiguation": (
                    "fraction of queries with >= 1 sibling where the correct trace outranks "
                    "every sibling; an unretrieved sibling ranks below a retrieved correct trace"
                ),
                "recall_at_k": "fraction of queries whose correct trace is in the top k",
                "precision_at_k": "recall_at_k / k (exactly one relevant trace per query)",
                "mrr": "mean reciprocal rank of the correct trace, 0 when it is not in the top k",
            },
            "per_seed": [run.to_dict() for run in self.runs],
            "aggregate_across_seeds": {
                name: interval.to_dict() for name, interval in sorted(self.aggregate.items())
            },
            "disambiguation_cluster_bootstrap": self.bootstrap.to_dict(),
            "accuracy_by_cluster_size": curve_to_dict(self.curve),
            "cluster_size_decay": self.decay,
            "arm_describe": self.describe,
            "corpus": self.corpus,
            "config": self.config,
            "wall_clock_seconds": round(float(self.wall_clock), 3),
            "environment": environment(),
        }


def corpus_provenance(corpus: Any) -> dict[str, Any]:
    """Everything needed to prove which corpus a number came from."""
    manifest: Mapping[str, Any] = getattr(corpus, "manifest", {}) or {}
    generator: Mapping[str, Any] = manifest.get("generator", {}) or {}
    return {
        "content_sha256": manifest.get("content_sha256"),
        "episodes_sha256": manifest.get("episodes_sha256"),
        "queries_sha256": manifest.get("queries_sha256"),
        "schema_version": manifest.get("schema_version"),
        "created_at": manifest.get("created_at"),
        "generation_seed": manifest.get("seed", generator.get("seed")),
        "generator": generator.get("generator"),
        "similarity": generator.get("similarity"),
        "n_episodes": len(getattr(corpus, "episodes", []) or []),
        "n_queries": len(getattr(corpus, "queries", []) or []),
        "n_clusters": len(corpus.clusters()) if hasattr(corpus, "clusters") else None,
        "cluster_sizes": (manifest.get("counts", {}) or {}).get("cluster_size_histogram"),
    }


def environment() -> dict[str, Any]:
    """The versions that decide whether a rerun can reproduce this file."""
    import numpy
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "brain_seed_env": int(SEED),
    }


def jsonable(value: Any) -> Any:
    """Recursively convert to JSON-safe types, mapping non-finite floats to null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = sorted(value) if isinstance(value, set) else list(value)
        return [jsonable(item) for item in items]
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if hasattr(value, "item"):  # numpy scalar
        return jsonable(value.item())
    return str(value)


def write_result(payload: Mapping[str, Any], *, arm: str, directory: Path | str) -> Path:
    """Write `results/<timestamp>-<arm>.json` and return the path.

    The timestamp is UTC and second-resolution, so two runs of the same arm in
    the same second would collide; a counter suffix keeps both rather than
    letting one silently overwrite the other.
    """
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in str(arm))
    path = root / f"{stamp}-{safe}.json"
    suffix = 1
    while path.exists():
        path = root / f"{stamp}-{safe}-{suffix}.json"
        suffix += 1
    body = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(body + "\n", encoding="utf-8")
    return path
