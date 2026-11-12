"""Render the README figures from the frozen result files.

Docs only: the harness never imports this file, and matplotlib is an optional extra
(`pip install -e '.[figures]'`). extract.py first rewrites data.json from results/,
which names the source file and JSON key of every plotted value; this file then draws.

    python docs/figures/plot.py              # data.json, PNG (light and dark) and SVG next to this file
    python docs/figures/plot.py --pdf DIR    # also a light PDF per figure into DIR
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
              "quiet": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
             "quiet": "#52514e"},
}


def style(t):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": t["ink2"], "lines.linewidth": 2, "lines.markersize": 6,
    })


def v(points):
    return [p["value"] for p in points]


def band(points):
    return [p["lo"] for p in points], [p["hi"] for p in points]


def xerr(p):
    return [[p["value"] - p["lo"]], [p["hi"] - p["value"]]]


def recall_vs_k(d, t):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    k = d["k"]["value"]
    for arm, ls, face in (("plain_vector", "-", None), ("bm25", (0, (4, 2)), t["surface"])):
        rows = d["arms"][arm]
        for metric, col, name in (("recall_at_k", t["s1"], "recall@k"), ("disambiguation", t["s2"], "disambiguation")):
            lo, hi = band(rows[metric])
            if arm == "plain_vector":
                a.fill_between(k, lo, hi, color=col, alpha=0.13, lw=0)
            a.plot(k, v(rows[metric]), ls=ls, marker="o", color=col, mfc=face or col, label=f"{name}, {arm}")
    a.set_xscale("log")
    a.set_xticks(k, [str(x) for x in k])
    a.minorticks_off()
    a.set_ylim(0.45, 1.06)
    a.set_xlabel("k, episodes retrieved per query")
    a.set_ylabel("share of 360 queries")
    a.set_title("(a) Recall saturates, disambiguation does not move")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    pre = d["precondition_plain_vector"]
    kk = [p["k"] for p in pre]
    for key, col, name in (("sibling_coverage", t["s1"], "sibling coverage (all lookalikes in the top k)"),
                           ("competitor_purity", t["s2"], "competitor purity (share of the top k that are lookalikes)")):
        pts = [p[key] for p in pre]
        lo, hi = band(pts)
        b.fill_between(kk, lo, hi, color=col, alpha=0.13, lw=0)
        b.plot(kk, v(pts), marker="o", color=col, label=name)
    b.set_xscale("log")
    b.set_xticks(kk, [str(x) for x in kk])
    b.minorticks_off()
    b.set_ylim(-0.03, 1.12)
    b.set_xlabel("k, episodes retrieved per query (plain_vector)")
    b.set_title("(b) The veto precondition holds at no k")
    b.annotate(f"k=10: coverage {pre[6]['sibling_coverage']['value']:.4f},\npurity {pre[6]['competitor_purity']['value']:.4f}",
               (10, pre[6]["competitor_purity"]["value"]), (12, 0.62), color=t["ink2"], fontsize=8,
               arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 0.8})
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
    fig.tight_layout(w_pad=2.5)
    return fig


def separation(d, t):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True)
    modes = {"decorrelate": (t["s1"], "o"), "expand_sparsify": (t["s2"], "s")}
    for mode, (col, mk) in modes.items():
        m = d["modes"][mode]
        s = v(m["s"])
        a.plot(s, v(m["sibling_cosine"]), marker=mk, color=col, label=mode)
        b.errorbar(s, v(m["disambiguation"]), yerr=[[p["value"] - p["lo"] for p in m["disambiguation"]],
                                                     [p["hi"] - p["value"] for p in m["disambiguation"]]],
                   marker=mk, color=col, ecolor=col, elinewidth=1, capsize=0, alpha=0.95, label=mode)
    a.set_ylim(0, 0.9)
    a.set_ylabel("mean cosine between lookalike episodes")
    a.set_title("(a) The transform moves lookalikes apart")
    a.legend(loc="upper right")
    c = d["control"]
    bar = c["value"] + d["margin"]["value"]
    b.axhspan(c["lo"], c["hi"], color=t["quiet"], alpha=0.35, lw=0)
    b.axhline(c["value"], color=t["muted"], lw=1)
    b.axhline(bar, color=t["muted"], lw=1, ls=(0, (3, 2)))
    b.text(0.81, bar + 0.006, f"5-point bar {bar:.4f}", ha="right", va="bottom", color=t["ink2"], fontsize=8)
    b.text(-0.02, c["lo"] - 0.006, f"plain_vector {c['value']:.4f}, 95% interval", ha="left", va="top",
           color=t["ink2"], fontsize=8)
    b.set_ylim(0.38, 0.75)
    b.set_ylabel("disambiguation (95% interval)")
    n, ok = d["n_configurations"]["value"], d["n_survives"]["value"]
    b.set_title(f"(b) {ok} of {n} configurations clear the bar")
    for ax in (a, b):
        ax.set_xlabel("separation strength s")
    fig.tight_layout(w_pad=2.5)
    return fig


def arms(d, t):
    rows = d["rows"][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bases = list(dict.fromkeys(r["basis"] for r in d["rows"]))
    color = {bases[0]: t["s1"], bases[1]: t["s2"]}
    seen = set()
    for y, r in enumerate(rows):
        lab = None if r["basis"] in seen else r["basis"]
        seen.add(r["basis"])
        ax.errorbar(r["value"], y, xerr=xerr(r), fmt="o", color=color[r["basis"]], ecolor=color[r["basis"]],
                    elinewidth=1.5, capsize=0, label=lab)
        ax.text(r["hi"] + 0.008, y, f"{r['value']:.4f}", ha="left", va="center", color=t["ink2"], fontsize=8)
    for key, txt, ha, dx in (("recall_at_5", "recall@5", "right", -0.004), ("recall_at_10", "recall@10", "left", 0.004)):
        x = d[key]["value"]
        ax.axvline(x, color=t["muted"], lw=1, ls=(0, (3, 2)))
        ax.text(x + dx, len(rows) - 0.4, f"{txt}\n{x:.4f}", ha=ha, va="top", color=t["ink2"], fontsize=7.5)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.40, 1.10)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("disambiguation: right episode ranked above all its lookalikes (95% cluster interval)")
    ax.set_title("Every arm on the synthetic benchmark, k = 5")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, fontsize=8.5)
    fig.tight_layout()
    return fig


def cluster_size(d, t):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    m = d["m"]
    series = (("full_context", t["s3"], "^"), ("bm25", t["s2"], "s"), ("plain_vector", t["s1"], "o"))
    for arm, col, mk in series:
        pts = d["arms"][arm]
        ax.errorbar(m, v(pts), yerr=[[p["value"] - p["lo"] for p in pts], [p["hi"] - p["value"] for p in pts]],
                    marker=mk, color=col, ecolor=col, elinewidth=1, capsize=0, label=arm)
        ax.text(m[-1] + 0.25, pts[-1]["value"], arm, color=t["ink2"], fontsize=8, va="center")
    fit = d["fit_plain_vector"]
    ax.plot(m, v(fit["predicted"]), ls=(0, (3, 2)), color=t["muted"], lw=1.3,
            label=f"1/(1+rho(m-1)), rho={fit['rho']['value']:.3f}, fitted to plain_vector")
    ax.set_xticks(m)
    ax.set_xlim(1.6, 9.6)
    ax.set_ylim(0.2, 1.06)
    ax.set_xlabel("m, episodes in the lookalike cluster (72 queries per size)")
    ax.set_ylabel("disambiguation (95% interval)")
    ax.set_title("Accuracy falls with the number of lookalikes, k = 5")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def threshold(d, t):
    rows = d["rows"][::-1]
    names = {"long": "session-length notes", "short": "terse assertions", "pooled": "both regimes pooled"}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for y, r in enumerate(rows):
        c = r["ceiling"]
        ax.errorbar(c["value"], y, xerr=xerr(c), fmt="o", color=t["s1"], ecolor=t["s1"], elinewidth=1.5, capsize=0)
        ax.text(c["value"], y + 0.2, f"{c['value']:.4f}  (tau* = {r['tau_star']['value']:.4f})", ha="center",
                va="bottom", color=t["ink2"], fontsize=8)
    bar = d["bar"]["value"]
    ax.axvline(bar, color=t["muted"], lw=1, ls=(0, (3, 2)))
    ax.text(bar + 0.004, -0.45, f"pre-registered bar {bar:.2f}", ha="left", va="bottom", color=t["ink2"], fontsize=8)
    ax.set_yticks(range(len(rows)), [names[r["regime"]] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.68, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlabel("best balanced accuracy of any single similarity threshold (95% interval)")
    ax.set_title("Supersession versus interference: the single-threshold ceiling")
    fig.tight_layout()
    return fig


def ncp_encoders(d, t):
    rows = d["rows"][::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    names = [r["encoder"]["value"].split("/")[-1] for r in rows]
    y = range(len(rows))
    for key, col, mk, lab in (("plain_vector", t["muted"], "o", "plain_vector (mean pooling)"),
                              ("best_global", t["s2"], "s", "global-IDF, best cell"),
                              ("best_sif", t["s3"], "^", "SIF, best cell"),
                              ("best_local", t["s1"], "D", "NCP local, best cell")):
        a.plot([r[key]["value"] for r in rows], y, ls="none", marker=mk, color=col, label=lab)
    a.set_yticks(list(y), names)
    a.grid(axis="y", visible=False)
    a.set_xlim(0.6, 0.74)
    a.set_xlabel("disambiguation, all 360 queries")
    a.set_title("(a) The encoder moves the baseline more")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    loc = [r["local_wins"]["value"] for r in rows]
    glo = [r["global_wins"]["value"] for r in rows]
    b.barh(list(y), loc, height=0.5, color=t["s1"], label="local df wins conclusively")
    b.barh(list(y), [-g for g in glo], height=0.5, color=t["s2"], label="global df wins conclusively")
    for i, (l_, g_) in enumerate(zip(loc, glo)):
        b.text(l_ + 1.5, i, str(l_), va="center", color=t["ink2"], fontsize=8)
        b.text(-g_ - 1.5, i, str(g_), va="center", ha="right", color=t["ink2"], fontsize=8)
    b.axvline(0, color=t["axis"], lw=1)
    b.set_xlim(-72, 40)
    b.grid(axis="y", visible=False)
    b.set_xlabel(f"matched cells, of {rows[0]['cells_compared']['value']} per encoder")
    b.set_title("(b) The sign follows the encoder")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=8)
    fig.tight_layout(w_pad=1.5)
    return fig


FIGURES = {"recall-vs-k": ("recall_vs_k", recall_vs_k), "separation": ("separation", separation),
           "arms": ("arms", arms), "cluster-size": ("cluster_size", cluster_size),
           "threshold-ceiling": ("threshold", threshold), "ncp-encoders": ("ncp_encoders", ncp_encoders)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="also write a light PDF of each figure into this directory")
    args = ap.parse_args()
    data = extract.write()
    for sub in ("png", "svg"):
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    if args.pdf:
        os.makedirs(args.pdf, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "membrain-benchmark"
    plt.rcParams["svg.fonttype"] = "none"
    for name, (key, draw) in FIGURES.items():
        for mode, t in THEMES.items():
            style(t)
            fig = draw(data[key], t)
            fig.savefig(os.path.join(HERE, "png", f"{name}-{mode}.png"), dpi=200, bbox_inches="tight")
            if mode == "light":
                fig.savefig(os.path.join(HERE, "svg", f"{name}.svg"), bbox_inches="tight", metadata={"Date": None})
                if args.pdf:
                    fig.savefig(os.path.join(args.pdf, f"{name}.pdf"), bbox_inches="tight",
                                metadata={"CreationDate": None})
            plt.close(fig)
    print("wrote data.json,", ", ".join(FIGURES))


if __name__ == "__main__":
    main()
