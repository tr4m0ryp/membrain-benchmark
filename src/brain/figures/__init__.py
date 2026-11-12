"""Paper figures, rendered from the frozen result files.

Every figure is drawn from a pinned run in `results/`, never from a number typed
into the plotting code, so a figure cannot drift away from the measurement it
reports. `build_all()` renders the set and rewrites the provenance manifest.

    python -m brain.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import io, manifest, style
from .manifest import REGISTRY, Figure

__all__ = ["REGISTRY", "Figure", "build_all", "build_one"]


#: Suppressing the embedded build timestamp makes a rebuild byte-identical when
#: the underlying result file has not changed. Without it every rebuild rewrites
#: all five PDFs, and the diff stops being able to say whether a number moved.
_REPRODUCIBLE = {"CreationDate": None}


def build_one(figure: Figure, out_dir: Path) -> Path:
    """Render one figure to PDF and return the path."""
    style.apply()
    fig = figure.build()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{figure.key}.pdf"
    fig.savefig(path, format="pdf", metadata=_REPRODUCIBLE)
    plt.close(fig)
    return path


def build_all(out_dir: Path | None = None) -> list[dict]:
    """Render every registered figure and rewrite `figures/manifest.json`."""
    out_dir = out_dir or io.FIGURES_DIR
    entries = []
    for figure in REGISTRY:
        path = build_one(figure, out_dir)
        entries.append(manifest.entry(figure, path))
    manifest.write(entries, out_dir)
    return entries
