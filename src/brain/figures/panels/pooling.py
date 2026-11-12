"""Figure 5: neighbourhood-conditional pooling, and the claim it did not earn.

Left, the pre-registered surface: the mechanism reduces exactly to mean pooling
at gamma = 0, peaks, then destroys the topic signal it needs. Right, the strict
one-variable ablation against global weighting, which is the comparison the
novelty claim actually rests on and the one it does not win.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .. import io
from ..style import FAINT, GOOD, HARM, INK, MUTED, OURS, annotate, figsize

CAPTION = (
    "\\textbf{(a)} Disambiguation over the pre-registered $(\\gamma, \\alpha)$ surface at "
    "$k = 8$ with an IDF query side. At $\\gamma = 0$ the scheme is mean pooling exactly and "
    "reproduces the control, which is the correctness test the design gives for free. Accuracy "
    "peaks near $\\gamma = 0.5$ and falls once the local rule starts suppressing tokens shared "
    "with the cluster itself. Every best cell sits at $\\alpha = 0$, so the second channel did "
    "not earn its place. \\textbf{(b)} The strict one-variable ablation, at the baseline encoder "
    "and $k = 8$ only: same $\\gamma$, same $\\alpha$, same query side, same $k$, and only the "
    "place the document frequency is measured changes. Local conditioning wins conclusively in 8 "
    "of 70 matched cells and loses conclusively in 1, with a mean advantage of $+0.27$ points, so "
    "the local rule does not separate from the global one it was built to beat. The encoder and "
    "neighbourhood sweep is reported separately; this panel varies neither."
)

_PRIMARY_K = 8
_PRIMARY_QUERY_SIDE = "idf"


def _surface(ncp: dict):
    cells = [
        c
        for c in ncp["cells"]
        if c["arm"].startswith("local-")
        and c["k_neighbours"] == _PRIMARY_K
        and c["query_side"] == _PRIMARY_QUERY_SIDE
    ]
    by_alpha: dict[float, list[tuple[float, float]]] = {}
    for c in cells:
        by_alpha.setdefault(c["alpha"], []).append((c["gamma"], c["metrics"]["disambiguation"]))
    return {a: sorted(v) for a, v in sorted(by_alpha.items())}


def _matched(ncp: dict):
    rows = ncp["paired_deltas"]["local_minus_global_matched"]
    out = [io.interval(row["delta"]) for row in rows]
    return sorted(out, key=lambda r: r[0])


def build(axes=None):
    ncp = io.load("ncp")
    control = ncp["baselines"]["plain_vector"]["metrics"]["disambiguation"]

    own_axes = axes is None
    if own_axes:
        fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, 2.25))
    else:
        fig = axes[0].figure

    ax_a, ax_b = axes
    surface = _surface(ncp)
    # The gamma grid is geometric with a zero in it, so it is plotted as an
    # ordered grid rather than on a scale; the spacing carries no meaning.
    gammas = sorted({g for pts in surface.values() for g, _ in pts})
    index = {g: i for i, g in enumerate(gammas)}
    shades = [OURS, "#3d6f9e", "#6b93b8", "#9db8cf", MUTED]
    for (alpha, points), colour in zip(surface.items(), shades):
        xs = [index[p[0]] for p in points]
        ys = [p[1] for p in points]
        ax_a.plot(xs, ys, marker="o", color=colour, label=f"$\\alpha = {alpha:g}$", zorder=3)

    ax_a.axhline(control, color=INK, linestyle=(0, (4, 2)), linewidth=0.8, zorder=2)
    ax_a.axhline(control + 0.05, color=GOOD, linestyle=(0, (1, 1.6)), linewidth=0.9, zorder=2)
    ax_a.set_xticks(range(len(gammas)), [f"{g:g}" for g in gammas])
    ax_a.set_xlim(-0.35, len(gammas) - 0.65)
    ax_a.set_xlabel("suppression exponent $\\gamma$")
    ax_a.set_ylabel("disambiguation accuracy")
    ax_a.set_title("(a) the pre-registered surface", pad=17)
    ax_a.set_ylim(control - 0.048, control + 0.10)
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        borderaxespad=0.0,
        handlelength=1.4,
        columnspacing=0.7,
        handletextpad=0.35,
    )
    right = len(gammas) - 0.7
    ax_a.text(right, control + 0.053, "+5 pt bar", ha="right", fontsize=6.4, color=GOOD)
    ax_a.text(right, control + 0.011, "control", ha="right", fontsize=6.4, color=MUTED)
    ax_a.text(
        -0.25,
        control - 0.037,
        "$\\gamma = 0$ is mean pooling exactly,\nand reproduces the control",
        fontsize=6.4,
        color=MUTED,
    )

    matched = _matched(ncp)
    xs = range(len(matched))
    for x, (mean, lower, upper) in zip(xs, matched):
        conclusive = lower > 0.0 or upper < 0.0
        colour = OURS if (conclusive and mean > 0) else (HARM if conclusive else FAINT)
        ax_b.plot([x, x], [lower * 100, upper * 100], color=colour, linewidth=0.8, zorder=2)
        ax_b.plot([x], [mean * 100], marker="o", markersize=2.2, color=colour, zorder=3)
    ax_b.axhline(0.0, color=INK, linewidth=0.7, zorder=4)

    paired = ncp["paired_deltas"]
    wins = paired["matched_cells_where_local_beats_global_conclusively"]
    losses = paired["matched_cells_where_global_beats_local_conclusively"]
    mean_delta = paired["mean_matched_delta_points"]
    ax_b.set_xlabel(f"each of {len(matched)} matched cells, sorted by delta")
    ax_b.set_ylabel("local minus global (points)")
    ax_b.set_title("(b) local conditioning against global", pad=17)
    ax_b.set_xticks([])
    ax_b.set_xlim(-2, len(matched) + 1)
    ax_b.set_ylim(-16.5, 10.5)
    ax_b.grid(axis="x", visible=False)
    ax_b.text(
        -1,
        10.0,
        f"{wins} conclusive wins for local, {losses} for global,\nmean advantage {mean_delta:+.2f} points",
        fontsize=6.4,
        color=MUTED,
        va="top",
    )

    if own_axes:
        fig.tight_layout(w_pad=1.6)
    return fig
