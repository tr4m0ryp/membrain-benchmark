"""Where each figure's numbers come from.

Sources are pinned by exact filename rather than resolved by glob. A figure in
the paper must not silently change because a later run landed in `results/`, and
pinning is what makes the manifest a provenance record rather than a listing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# Pinned canonical runs, each under the directory of the experiment that made it.
# Every number the paper states is read off one of these.
SOURCES: dict[str, str] = {
    # The gate experiment: separation sweep, arm accuracies, cluster-size curves.
    "gate": "gate-separation/gate-20260724T224755Z.json",
    # Task 007 on the v2 relational corpus. This is F25, and it supersedes the
    # v1 threshold run whose 0.8686 long-regime ceiling was draw-lucky.
    "threshold": "threshold-ceiling/threshold-20260725T185408Z.json",
    # Consolidation delta, full 5-seed run over the frozen synthetic corpus.
    "consolidation": "consolidation/20260725T114045Z-consolidation_delta.json",
    # Neighbourhood-conditional pooling, 320 cells.
    "ncp": "ncp/ncp-20260725T215915Z.json",
    # The veto experiment (K-I4). Named "smoke" by the harness; it is the full run.
    "veto": "veto/veto-smoke-20260725T144603Z.json",
    # The k sweep. Supplies the saturated recall ceiling; 0.958 is a k=5 artifact.
    "ksweep": "k-sweep/ksweep-20260801T184604Z.json",
    # Memory precision over the resident store.
    "precision": "precision/memory_precision.json",
}


class MissingSource(FileNotFoundError):
    """A pinned result file is not on disk."""


def source_path(key: str) -> Path:
    """Absolute path of a pinned result file."""
    try:
        rel = SOURCES[key]
    except KeyError:
        raise KeyError(f"unknown figure source {key!r}; known: {sorted(SOURCES)}") from None
    path = RESULTS_DIR / rel
    if not path.is_file():
        raise MissingSource(
            f"pinned result {rel} is missing. Figures are built from frozen runs, "
            f"so regenerate that run rather than repointing the source."
        )
    return path


def load(key: str) -> dict[str, Any]:
    """Parse a pinned result file."""
    with source_path(key).open(encoding="utf-8") as handle:
        return json.load(handle)


def rel_source(key: str) -> str:
    """Repo-relative path, for the manifest."""
    return str(source_path(key).relative_to(REPO_ROOT))


def interval(node: dict[str, Any]) -> tuple[float, float, float]:
    """Pull (mean, lower, upper) out of this project's interval records."""
    return float(node["mean"]), float(node["lower"]), float(node["upper"])


def yerr(node: dict[str, Any]) -> list[list[float]]:
    """Asymmetric error bars for matplotlib, from one interval record."""
    mean, lower, upper = interval(node)
    return [[max(0.0, mean - lower)], [max(0.0, upper - mean)]]
