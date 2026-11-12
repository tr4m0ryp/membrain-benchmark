"""Print style for the paper figures.

One place decides type, colour and size, so every figure in the paper reads as
one system. The serif stack matches the NeurIPS Times body text; TeX Gyre Termes
is the metric-compatible clone and the rest are fallbacks for other machines.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

# NeurIPS 2026 \textwidth is 5.5in. Figures are sized in fractions of it so a
# \includegraphics[width=\textwidth] never rescales the type.
TEXT_WIDTH_IN = 5.5

SERIF_STACK = ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]

# Restrained, colourblind-safe. Hue carries meaning and is reused across figures:
# OURS for a mechanism of ours, HARM for a measured cost, GOOD for a measured
# gain, PAID for anything that bills a model call, MUTED for controls.
INK = "#1a1a1a"
OURS = "#1f4e79"
HARM = "#b03a2e"
GOOD = "#2e7d5b"
PAID = "#8e6c1f"
MUTED = "#7f8c8d"
FAINT = "#c8ccce"
GRID = "#e4e7e9"


def apply() -> None:
    """Install the paper's rcParams. Idempotent."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def figsize(width_frac: float = 1.0, height_in: float = 2.2) -> tuple[float, float]:
    """A figure size in fractions of the NeurIPS text width."""
    return (TEXT_WIDTH_IN * width_frac, height_in)


def annotate(ax, text: str, xy, xytext, color: str = INK, **kw) -> None:
    """A leader-line annotation in the paper's voice: small, grey, no box."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=6.4,
        color=color,
        arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": color, "shrinkA": 0, "shrinkB": 1.5},
        **kw,
    )
