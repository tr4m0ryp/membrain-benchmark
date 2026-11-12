"""The figure registry and its provenance record.

`figures/manifest.json` is the record of which frozen run each figure was drawn
from, so a number in the paper can be traced from the plot back to the result
file that produced it. It is written by the builder, never by hand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import io
from .panels import ceiling, competition, consolidation, pooling, separation


@dataclass(frozen=True)
class Figure:
    """One figure: where it goes, what it draws, and what it was drawn from."""

    key: str
    label: str
    build: Callable
    caption: str
    sources: tuple[str, ...]


REGISTRY: tuple[Figure, ...] = (
    Figure(
        key="competition-gap",
        label="fig:competition",
        build=competition.build,
        caption=competition.CAPTION,
        sources=("ncp", "veto"),
    ),
    Figure(
        key="separation-sweep",
        label="fig:separation",
        build=separation.build,
        caption=separation.CAPTION,
        sources=("gate",),
    ),
    Figure(
        key="threshold-ceiling",
        label="fig:ceiling",
        build=ceiling.build,
        caption=ceiling.CAPTION,
        sources=("threshold",),
    ),
    Figure(
        key="consolidation-cost",
        label="fig:consolidation",
        build=consolidation.build,
        caption=consolidation.CAPTION,
        sources=("consolidation",),
    ),
    Figure(
        key="pooling-surface",
        label="fig:pooling",
        build=pooling.build,
        caption=pooling.CAPTION,
        sources=("ncp",),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write(entries: list[dict], out_dir: Path) -> Path:
    """Write the manifest beside the rendered figures."""
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def entry(figure: Figure, pdf_path: Path) -> dict:
    """The provenance record for one rendered figure."""
    return {
        "id": figure.key,
        "file": str(pdf_path.relative_to(io.REPO_ROOT)),
        "label": figure.label,
        "caption": figure.caption,
        "sources": [io.rel_source(key) for key in figure.sources],
        "sha256": _sha256(pdf_path),
        "generated_by": "python -m brain.figures",
    }
