"""Figure 2: pushing vectors apart is not telling them apart.

The gate experiment's whole result in one frame. Sibling cosine falls by two
thirds across the sweep and disambiguation accuracy does not follow it up. The
five-point bar is drawn so a reader can see that no cell reaches it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .. import io
from ..style import FAINT, GOOD, HARM, INK, MUTED, OURS, annotate, figsize

CAPTION = (
    "Write-time separation strength against what it was built to buy. The mechanism fires: "
    "sibling cosine falls from $0.824$ to $0.290$ across the sweep. Disambiguation accuracy does "
    "not follow, never reaches the five-point bar, and collapses once the push is large enough to "
    "rotate a vector away from its own queries. The pre-registered setting $s = 0.25$ is worse "
    "than doing nothing. Shaded band is the cluster-BCa interval on the control."
)


def _sweep(gate: dict, mode: str) -> list[dict]:
    rows = [r for r in gate["sweep"] if r["mode"] == mode and r["aggregation"] == "sequential"]
    return sorted(rows, key=lambda r: r["s"])


def build(ax=None):
    gate = io.load("gate")
    dec = _sweep(gate, "decorrelate")
    exp = _sweep(gate, "expand_sparsify")

    control = next(r for r in dec if r["s"] == 0.0)
    ctrl_acc = control["disambiguation"]
    bar = ctrl_acc + 0.05

    own_axes = ax is None
    if own_axes:
        fig, ax = plt.subplots(figsize=figsize(1.0, 2.5))
    else:
        fig = ax.figure

    xs = [r["s"] for r in dec]
    acc = [r["disambiguation"] for r in dec]
    cos = [r["sibling_cosine"] for r in dec]

    ax.axhspan(control["ci_lower"], control["ci_upper"], color=FAINT, alpha=0.28, linewidth=0, zorder=0)
    ax.axhline(ctrl_acc, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    ax.axhline(bar, color=GOOD, linewidth=0.9, linestyle=(0, (1, 1.6)), zorder=1)

    ax.plot(xs, acc, marker="o", color=OURS, label="disambiguation, decorrelate", zorder=3)
    ax.plot(
        [r["s"] for r in exp],
        [r["disambiguation"] for r in exp],
        marker="s",
        markerfacecolor="white",
        linestyle=(0, (3, 1.5)),
        color=HARM,
        label="disambiguation, expand and sparsify",
        zorder=3,
    )

    twin = ax.twinx()
    twin.plot(xs, cos, marker="^", color=MUTED, linewidth=1.0, label="sibling cosine", zorder=2)
    twin.set_ylabel("sibling cosine", color=MUTED)
    twin.tick_params(axis="y", colors=MUTED)
    twin.set_ylim(0.0, 0.95)
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(MUTED)
    twin.spines["right"].set_linewidth(0.6)

    ax.set_xlabel("separation strength $s$")
    ax.set_ylabel("disambiguation accuracy")
    ax.set_ylim(0.42, 0.72)
    ax.set_xlim(-0.03, 0.86)
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)

    # Both reference lines are labelled at the right edge, clear of every series.
    ax.text(0.855, bar + 0.006, f"+5 pt bar ({bar:.4f})", ha="right", fontsize=6.4, color=GOOD)
    ax.text(0.855, ctrl_acc + 0.006, f"control ({ctrl_acc:.4f})", ha="right", fontsize=6.4, color=MUTED)
    pre = next(r for r in dec if r["s"] == 0.25)
    annotate(
        ax,
        "pre-registered $s = 0.25$,\nworse than doing nothing",
        xy=(0.25, pre["disambiguation"]),
        xytext=(0.30, 0.452),
        color=INK,
    )

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(
        handles + h2,
        labels + l2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.2,
    )

    if own_axes:
        fig.tight_layout()
    return fig
