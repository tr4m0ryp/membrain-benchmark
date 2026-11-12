"""Figure 1: the competition gap.

The diagnostic that redirected the project. Every arm that makes no model call
sits far below the retrieval ceiling, which means the correct memory is retrieved
and then out-ranked. Colour carries the argument: what is free, and what bills.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .. import io
from ..style import FAINT, GOOD, HARM, INK, MUTED, OURS, PAID, figsize

CAPTION = (
    "Disambiguation accuracy against the retrieval ceiling. Every arm that makes no model "
    "call sits at least 29 points below the share of queries whose correct memory is retrieved "
    "at all, so the loss is competition rather than retrieval. Two ceilings are drawn because "
    "the first is a parameter rather than a law: recall is $0.958$ at $k = 5$ and saturates at "
    "$1.000$ by $k = 10$, and the headroom figures elsewhere in this paper are stated against "
    "the saturated one. Free arms are measured over all 360 queries and the arms that bill a "
    "model call over the 180-query stratified subsample, which is the basis their kill criterion "
    "was written on; the control is $0.6194$ on the first basis and $0.6000$ on the second, a "
    "difference well inside its interval. Intervals are cluster-BCa at $B = 10000$."
)


def _best_local_cell(ncp: dict) -> dict:
    local = [c for c in ncp["cells"] if c["arm"].startswith("local-")]
    return max(local, key=lambda c: c["metrics"]["disambiguation"])


def _best_family(ncp: dict, prefix: str) -> dict:
    fam = [c for c in ncp["cells"] if c["arm"].startswith(prefix)]
    return max(fam, key=lambda c: c["metrics"]["disambiguation"])


def build(ax=None):
    ncp = io.load("ncp")
    veto = io.load("veto")

    base = ncp["baselines"]
    rows = [
        ("plain vector", base["plain_vector"]["bootstrap"], MUTED, "free"),
        ("BM25 (1994)", base["bm25"]["bootstrap"], MUTED, "free"),
        ("global IDF", _best_family(ncp, "global-")["bootstrap"], FAINT, "free"),
        ("SIF (2017)", _best_family(ncp, "sif-")["bootstrap"], FAINT, "free"),
        ("NCP, best cell", _best_local_cell(ncp)["bootstrap"], OURS, "free"),
        ("verify and veto", veto["axes"]["verify_veto"]["disambiguation"], HARM, "paid"),
        ("one model call", veto["axes"]["llm_pick"]["disambiguation"], PAID, "paid"),
        ("full context", base["full_context"]["bootstrap"], GOOD, "paid"),
    ]

    own_axes = ax is None
    if own_axes:
        fig, ax = plt.subplots(figsize=figsize(1.0, 2.45))
    else:
        fig = ax.figure

    # Two ceilings, because the first one is the mistake this figure used to make.
    # recall@5 is not where retrieval tops out; sweeping k puts it at 1.0 by k=10.
    ceiling = float(base["plain_vector"]["metrics"]["recall_at_k"])
    ksweep = io.load("ksweep")
    saturated = float(
        ksweep["results"]["by_arm"]["plain_vector"]["10"]["metrics"]["recall_at_k"]["mean"]
    )
    top = len(rows) - 1
    ax.axvline(ceiling, color=MUTED, linestyle=(0, (2, 2)), linewidth=0.7, zorder=1)
    ax.axvline(saturated, color=INK, linestyle=(0, (4, 2)), linewidth=0.9, zorder=1)
    ax.axvspan(0.0, 0.275, color=FAINT, alpha=0.35, linewidth=0, zorder=0)

    ypos = list(range(len(rows)))[::-1]
    for y, (label, node, colour, basis) in zip(ypos, rows):
        mean, lower, upper = io.interval(node)
        ax.barh(
            y,
            mean,
            height=0.62,
            color=colour,
            alpha=0.9 if basis == "free" else 0.75,
            edgecolor=INK,
            linewidth=0.5,
            hatch="" if basis == "free" else "///",
            zorder=2,
        )
        ax.plot([lower, upper], [y, y], color=INK, linewidth=0.9, zorder=3)
        ax.plot([lower, upper], [y, y], marker="|", markersize=3, color=INK, linewidth=0, zorder=3)
        # Value labels live in a reserved right-hand column so nothing collides
        # with an interval, however wide that interval turns out to be.
        ax.text(1.10, y, f"{mean:.4f}", va="center", ha="right", fontsize=6.8, color=INK, zorder=4)

    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.set_xlabel("disambiguation accuracy")
    ax.set_xlim(0.0, 1.12)
    ax.set_ylim(-0.62, top + 1.75)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", visible=False)

    ax.text(
        saturated - 0.012,
        top + 1.42,
        f"retrieval ceiling: recall@10 = {saturated:.3f}",
        ha="right",
        va="center",
        fontsize=6.6,
        color=INK,
    )
    ax.text(
        ceiling - 0.012,
        top + 0.92,
        f"recall@5 = {ceiling:.3f}, not the ceiling",
        ha="right",
        va="center",
        fontsize=6.2,
        color=MUTED,
    )
    ax.text(0.283, -0.45, "chance floor", ha="left", va="center", fontsize=6.4, color=MUTED)
    ax.text(
        1.10,
        top + 1.42,
        "accuracy",
        ha="right",
        va="center",
        fontsize=6.6,
        color=MUTED,
    )
    # A key rather than a leader line: there is no clear space inside the bars
    # to put one, and every bar starts at zero.
    key = Rectangle(
        (0.012, top + 0.25),
        0.055,
        0.40,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.5,
        hatch="///",
        transform=ax.transData,
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(key)
    ax.text(
        0.080,
        top + 0.45,
        "bills model calls: $\\approx$\\$0.014 one call, $\\approx$\\$0.062 five-call veto",
        va="center",
        fontsize=6.4,
        color=PAID,
    )

    if own_axes:
        fig.tight_layout()
    return fig
