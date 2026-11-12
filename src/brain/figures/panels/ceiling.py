"""Figure 3: the single-threshold ceiling, as a picture.

Theorem 1 says one similarity scalar cannot express the four-class action map.
The two regimes show it directly: on session-length notes the two classes sit on
top of each other, on terse assertions they come apart, and the thresholds that
serve them are 0.2389 apart, so a pooled rule must abandon one of them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .. import io
from ..style import FAINT, HARM, INK, MUTED, OURS, annotate, figsize

CAPTION = (
    "Why one similarity threshold cannot do this job. Each bar spans the 5th to 95th percentile "
    "of surface similarity for a class, with the median marked and the full support as a thin "
    "line; $\\tau^{\\ast}$ is the threshold reaching that regime's ceiling. On session-length "
    "notes the classes overlap over $[0.7884, 0.9698]$, which holds $93.2\\%$ of the supersession "
    "mass and $80.1\\%$ of the interference mass, and the ceiling is $0.7820$. On terse assertions "
    "the same measurement separates cleanly and reaches $0.9589$. The optimal thresholds differ by "
    "$0.2389$, which is why the pooled ceiling of $0.7391$ falls below both of the regimes it pools."
)

_CLASSES = (("SUPERSESSION", OURS), ("INTERFERENCE", HARM))
_REGIMES = (("long", "session-length notes"), ("short", "terse assertions"))


def _dist(threshold: dict, regime: str, cls: str) -> dict:
    for row in threshold["distributions"]["derivable"]:
        if row["regime"] == regime and row["cls"] == cls:
            return row
    raise KeyError(f"no distribution for {regime}/{cls}")


def build(axes=None):
    threshold = io.load("threshold")
    same_topic = threshold["ceilings"]["derivable"]["same_topic"]

    own_axes = axes is None
    if own_axes:
        fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, 2.05), sharex=True)
    else:
        fig = axes[0].figure

    for ax, (regime, title) in zip(axes, _REGIMES):
        node = same_topic[regime]
        ceiling = node["ceiling"]["balanced_accuracy"]
        tau = node["ceiling"]["tau_star"]
        band = node["overlap"]["interval"]

        ax.axvspan(band[0], band[1], color=FAINT, alpha=0.5, linewidth=0, zorder=0)
        ax.axvline(tau, color=INK, linestyle=(0, (4, 2)), linewidth=0.9, zorder=4)

        for y, (cls, colour) in zip((1.0, 0.0), _CLASSES):
            d = _dist(threshold, regime, cls)
            ax.plot([d["min"], d["max"]], [y, y], color=colour, linewidth=0.7, alpha=0.55, zorder=2)
            ax.barh(
                y,
                d["p95"] - d["p05"],
                left=d["p05"],
                height=0.34,
                color=colour,
                alpha=0.55,
                edgecolor=colour,
                linewidth=0.7,
                zorder=3,
            )
            ax.plot([d["p50"]], [y], marker="|", markersize=9, color=INK, markeredgewidth=1.1, zorder=5)

        sup_mass = node["overlap"]["supersession_mass_in_overlap"]
        int_mass = node["overlap"]["interference_mass_in_overlap"]
        ax.text(
            -0.12,
            -0.52,
            f"shaded band holds {sup_mass:.1%} of the supersession mass\n"
            f"and {int_mass:.1%} of the interference mass",
            ha="left",
            va="center",
            fontsize=6.2,
            color=MUTED,
        )
        ax.text(
            tau,
            1.44,
            f"$\\tau^{{\\ast}}$",
            ha="center",
            va="center",
            fontsize=7.0,
            color=INK,
        )

        ax.set_yticks([1.0, 0.0], ["supersession", "interference"])
        ax.set_ylim(-0.92, 1.62)
        ax.set_xlim(-0.14, 1.05)
        ax.set_xlabel("surface similarity")
        ax.set_title(f"{title}\nceiling {ceiling:.4f} at $\\tau^{{\\ast}} = {tau:.4f}$", pad=5)
        ax.grid(axis="y", visible=False)

    pooled = same_topic["pooled"]["ceiling"]
    if own_axes:
        fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.90))
        fig.text(
            0.5,
            0.955,
            f"one rule for both regimes reaches {pooled['balanced_accuracy']:.4f}, "
            f"below either of the regimes it pools",
            ha="center",
            va="center",
            fontsize=7.2,
            color=MUTED,
        )
    return fig
