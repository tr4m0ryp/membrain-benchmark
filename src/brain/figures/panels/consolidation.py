"""Figure 4: what one consolidation pass costs, and what decides that.

Three panels. The cost against the merge threshold, the two metrics moving in
opposite directions, which is the addressability argument, and the crediting
convention, which is the part of this result that is not replication.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .. import io
from ..style import FAINT, GOOD, HARM, INK, MUTED, OURS, annotate, figsize

CAPTION = (
    "One consolidation pass over a 120-episode store. \\textbf{(a)} Destructive consolidation is "
    "negative at every threshold that merges anything, non-destructive consolidation is slightly "
    "positive, and a length placebo straddles zero, so the loss is the replacement step and not "
    "the summarising. \\textbf{(b)} Under destructive merging lexical recall at rank one rises "
    "while disambiguation falls: the words survive and what is lost is the binding between them "
    "and the occasion they came from. \\textbf{(c)} The same destructive cell at $\\tau = 0.85$ "
    "under three conventions for crediting a record that absorbed several events. On this corpus the "
    "convention decides the sign and the generous one is what a vendor would reach for; on a public "
    "benchmark whose questions each rest on one piece of evidence the three coincide, so the lever "
    "is specific to corpora containing interference. Intervals are "
    "paired cluster bootstrap at $B = 10000$."
)

_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
_PRIMARY = "expected@full"
_ARMS = (
    ("destructive@", "destructive", HARM, "o", "-"),
    ("non_destructive@", "non-destructive", GOOD, "s", (0, (3, 1.5))),
    ("placebo_self@", "length placebo", MUTED, "^", (0, (1, 1.6))),
)
_CONVENTIONS = (
    ("credited@full", "generous\n(credit every\nabsorbed event)", MUTED),
    ("expected@full", "expected\n$1/(1+a)$", OURS),
    ("strict@full", "strict\n(originating\nevent only)", HARM),
)


def _delta(cons: dict, condition: str, metric: str) -> dict:
    return cons["conditions"][condition]["metrics"][metric]["delta_vs_before"]


def _series(cons: dict, prefix: str, metric: str):
    xs, means, los, his = [], [], [], []
    for tau in _THRESHOLDS:
        key = f"{prefix}{tau:g}"
        if key not in cons["conditions"]:
            continue
        mean, lower, upper = io.interval(_delta(cons, key, metric))
        xs.append(tau)
        means.append(mean)
        los.append(lower)
        his.append(upper)
    return xs, means, los, his


def build(axes=None):
    cons = io.load("consolidation")

    own_axes = axes is None
    if own_axes:
        fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, 2.15))
    else:
        fig = axes[0].figure

    ax_a, ax_b, ax_c = axes

    for prefix, label, colour, marker, dash in _ARMS:
        xs, means, los, his = _series(cons, prefix, _PRIMARY)
        ax_a.plot(xs, means, marker=marker, color=colour, linestyle=dash, label=label, zorder=3)
        ax_a.fill_between(xs, los, his, color=colour, alpha=0.16, linewidth=0, zorder=1)
    ax_a.axhline(0.0, color=INK, linewidth=0.7, zorder=2)
    ax_a.set_xlabel("merge threshold $\\tau$")
    ax_a.set_ylabel("change in disambiguation")
    ax_a.set_title("(a) the cost of the pass")
    ax_a.legend(loc="lower right", borderaxespad=0.3)
    ax_a.set_xticks(list(_THRESHOLDS))
    ax_a.set_ylim(-0.42, 0.10)

    xs, dis, _, _ = _series(cons, "destructive@", _PRIMARY)
    _, lex, _, _ = _series(cons, "destructive@", "answer_recall@1")
    ax_b.axhline(0.0, color=INK, linewidth=0.7, zorder=2)
    ax_b.plot(xs, lex, marker="o", color=GOOD, label="lexical recall@1", zorder=3)
    ax_b.plot(xs, dis, marker="s", color=HARM, linestyle=(0, (3, 1.5)), label="disambiguation", zorder=3)
    ax_b.set_xlabel("merge threshold $\\tau$")
    ax_b.set_title("(b) what merging destroys", pad=5)
    ax_b.set_xticks(list(_THRESHOLDS))
    ax_b.set_ylim(-0.44, 0.36)
    ax_b.text(0.752, lex[0] + 0.042, "lexical recall@1,\nthe words survive", fontsize=6.4, color=GOOD)
    ax_b.text(0.752, dis[0] - 0.085, "disambiguation,\nthe addressing does not", fontsize=6.4, color=HARM)

    ax_c.axhspan(0.0, 0.17, color=FAINT, alpha=0.45, linewidth=0, zorder=0)
    ax_c.axhline(0.0, color=INK, linewidth=0.7, zorder=2)
    for x, (metric, label, colour) in enumerate(_CONVENTIONS):
        mean, lower, upper = io.interval(_delta(cons, "destructive@0.85", metric))
        ax_c.plot([x, x], [lower, upper], color=colour, linewidth=1.1, zorder=3)
        ax_c.plot([x], [mean], marker="o", color=colour, markersize=4.5, zorder=4)
        ax_c.text(x + 0.14, mean, f"{mean:+.3f}", va="center", fontsize=6.6, color=colour, zorder=5)
    ax_c.set_xticks(range(len(_CONVENTIONS)), [c[1] for c in _CONVENTIONS], fontsize=6.0)
    ax_c.set_xlim(-0.55, len(_CONVENTIONS) - 0.22)
    ax_c.set_ylim(-0.45, 0.19)
    ax_c.set_title("(c) the convention decides the sign", pad=5)
    ax_c.grid(axis="x", visible=False)
    ax_c.text(-0.45, 0.145, "reads as beneficial", fontsize=6.4, color=MUTED, va="center")

    if own_axes:
        fig.tight_layout(w_pad=1.4)
    return fig
