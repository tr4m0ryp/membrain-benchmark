"""Build data.json from the frozen result files under results/.

Every plotted value is read here, never typed by hand, and is stored next to the
file it came from and its JSON Pointer (RFC 6901) inside that file:

    {"value": 0.6194, "lo": 0.5273, "hi": 0.6812,
     "source": "results/k-sweep/ksweep-20260801T184604Z.json",
     "key": "/results/by_arm/plain_vector/5/metrics/disambiguation",
     "fields": ["mean", "lower", "upper"]}

`fields` names the members of the object at `key` that became value, lo and hi.
A value without `fields` is the scalar at `key` itself. Nothing is rounded or
recomputed; the figures round only when they print a label.

    python docs/figures/extract.py      # rewrite data.json next to this file
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

KSWEEP = "results/k-sweep/ksweep-20260801T184604Z.json"
KADD = "results/k-sweep/ksweep-addendum-20260801T192231Z.json"
GATE = "results/gate-separation/gate-20260724T224755Z.json"
VETO = "results/veto/veto-smoke-20260725T144603Z.json"
NCP = "results/ncp/ncp-20260725T215915Z.json"
NCPENC = "results/ncp/ncp-encoder-20260801T195813Z.json"
THRESH = "results/threshold-ceiling/threshold-20260725T185408Z.json"
CI = ("mean", "lower", "upper")
_DOCS = {}


def _doc(src):
    if src not in _DOCS:
        with open(os.path.join(ROOT, src), encoding="utf-8") as f:
            _DOCS[src] = json.load(f)
    return _DOCS[src]


def _resolve(src, key):
    node = _doc(src)
    for raw in key.strip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _ptr(*parts):
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def scalar(src, *parts):
    key = _ptr(*parts)
    return {"value": _resolve(src, key), "source": src, "key": key}


def interval(src, *parts, fields=CI):
    key = _ptr(*parts)
    node = _resolve(src, key)
    v, lo, hi = (node[f] for f in fields)
    return {"value": v, "lo": lo, "hi": hi, "source": src, "key": key, "fields": list(fields)}


def _ncp_cell(arm):
    cells = _doc(NCP)["cells"]
    index = [c["arm"] for c in cells].index(arm)
    return interval(NCP, "cells", index, "bootstrap")


def recall_vs_k():
    out = {"k": scalar(KSWEEP, "pre_registration", "k_grid"), "arms": {}}
    for arm in ("plain_vector", "bm25"):
        ks = _resolve(KSWEEP, _ptr("pre_registration", "k_grid"))
        out["arms"][arm] = {
            "recall_at_k": [interval(KSWEEP, "results", "by_arm", arm, k, "metrics", "recall_at_k") for k in ks],
            "disambiguation": [interval(KSWEEP, "results", "by_arm", arm, k, "metrics", "disambiguation")
                               for k in ks],
        }
    pre = []
    for k in (1, 2, 3, 5, 7, 8, 10, 20, 50):
        src, base = (KADD, ("precondition_at_k", "by_arm")) if k in (7, 8) else \
            (KSWEEP, ("results", "precondition_coverage_by_arm"))
        pre.append({"k": k,
                    "sibling_coverage": interval(src, *base, "plain_vector", k, "sibling_coverage"),
                    "competitor_purity": interval(src, *base, "plain_vector", k, "competitor_purity")})
    out["precondition_plain_vector"] = pre
    return out


def separation():
    sweep = _doc(GATE)["sweep"]
    modes = {}
    for i, row in enumerate(sweep):
        m = modes.setdefault(row["mode"], {"s": [], "sibling_cosine": [], "disambiguation": [],
                                          "pair_accuracy_m2": []})
        m["s"].append(scalar(GATE, "sweep", i, "s"))
        m["sibling_cosine"].append(scalar(GATE, "sweep", i, "sibling_cosine"))
        m["disambiguation"].append(interval(GATE, "sweep", i, fields=("disambiguation", "ci_lower", "ci_upper")))
        m["pair_accuracy_m2"].append(scalar(GATE, "sweep", i, "accuracy_by_cluster_size", "2"))
    return {"modes": modes,
            "control": interval(GATE, "arms", "plain_vector", "disambiguation_cluster_bootstrap"),
            "bm25": interval(GATE, "arms", "bm25", "disambiguation_cluster_bootstrap"),
            "margin": scalar(GATE, "verdicts", "preregistered", "margin"),
            "n_configurations": scalar(GATE, "sweep_verdicts", "n_configurations"),
            "n_survives": scalar(GATE, "sweep_verdicts", "n_survives")}


def arms():
    gate = lambda a: interval(GATE, "arms", a, "disambiguation_cluster_bootstrap")  # noqa: E731
    veto = lambda a: interval(VETO, "axes", a, "disambiguation")  # noqa: E731
    full = "all 360 queries"
    sub = "180-query subsample"
    rows = [
        ("write_time_mmr", full, gate("write_time_mmr")),
        ("plain_vector (standard RAG)", full, gate("plain_vector")),
        ("separation, best of 14 (decorrelate s=0.05)", full,
         interval(GATE, "sweep", 2, fields=("disambiguation", "ci_lower", "ci_upper"))),
        ("bm25", full, gate("bm25")),
        ("NCP, pre-registered cell", full, _ncp_cell("local-g1-k8-al0.5-qmean")),
        ("global-IDF pooling, best cell", full, _ncp_cell("global-g4-al0-qidf")),
        ("SIF pooling, best cell", full, _ncp_cell("sif-g0-a0.001-al0-qidf")),
        ("NCP, best of 210 cells", full, _ncp_cell("local-g0.25-k8-al0-qidf")),
        ("full_context (model call)", full, gate("full_context")),
        ("plain_vector (control)", sub, veto("plain_vector")),
        ("verify_veto (model calls)", sub, veto("verify_veto")),
        ("llm_pick (model call)", sub, veto("llm_pick")),
    ]
    return {"rows": [{"label": lab, "basis": basis, **pt} for lab, basis, pt in rows],
            "recall_at_5": interval(KSWEEP, "results", "by_arm", "plain_vector", 5, "metrics", "recall_at_k"),
            "recall_at_10": interval(KSWEEP, "results", "by_arm", "plain_vector", 10, "metrics", "recall_at_k")}


def cluster_size():
    sizes = ["2", "3", "4", "6", "8"]
    out = {"m": [int(m) for m in sizes], "arms": {}}
    for arm in ("plain_vector", "bm25", "full_context"):
        out["arms"][arm] = [interval(GATE, "arms", arm, "accuracy_by_cluster_size", m, "ci") for m in sizes]
    fit = ("arms", "plain_vector", "cluster_size_decay", "m_a_fit")
    out["fit_plain_vector"] = {"rho": scalar(GATE, *fit, "rho"),
                               "predicted": [scalar(GATE, *fit, "predicted", m) for m in sizes],
                               "max_abs_error": scalar(GATE, *fit, "max_abs_error")}
    return out


def threshold():
    rows = []
    for regime in ("long", "short", "pooled"):
        base = ("ceilings", "derivable", "same_topic", regime)
        rows.append({"regime": regime,
                     "ceiling": interval(THRESH, *base, "bootstrap", "interval"),
                     "tau_star": scalar(THRESH, *base, "ceiling", "tau_star"),
                     "cross_validated": scalar(THRESH, *base, "cross_validated", "balanced_accuracy")})
    return {"rows": rows, "bar": scalar(THRESH, "k1_verdicts", 0, "threshold")}


def ncp_encoders():
    rows = []
    for i, s in enumerate(_doc(NCPENC)["encoder_summaries"]):
        wins = ("encoder_summaries", i, "strict_matched_win_counts_all_k")
        rows.append({
            "encoder": scalar(NCPENC, "encoder_summaries", i, "encoder"),
            "plain_vector": scalar(NCPENC, "encoder_summaries", i, "plain_vector_disambiguation"),
            "best_local": scalar(NCPENC, "encoder_summaries", i, "best_local_disambiguation"),
            "best_global": scalar(NCPENC, "encoder_summaries", i, "best_global_disambiguation"),
            "best_sif": scalar(NCPENC, "encoder_summaries", i, "best_sif_disambiguation"),
            "local_wins": scalar(NCPENC, *wins, "local_beats_global_conclusively"),
            "global_wins": scalar(NCPENC, *wins, "global_beats_local_conclusively"),
            "cells_compared": scalar(NCPENC, *wins, "cells_compared"),
        })
    return {"rows": rows}


def build():
    return {
        "about": ("Every value plotted in docs/figures, read by extract.py from the frozen result files "
                  "of this repository. 'source' is the file, 'key' the JSON Pointer inside it, 'fields' the "
                  "members of that object used as value, lo and hi. Synthetic benchmark: 120 episodes, "
                  "360 queries, 33 clusters of lookalike episodes, all-MiniLM-L6-v2 encoder unless an "
                  "encoder is named, seeds 0 to 4, intervals cluster-BCa at B = 10000."),
        "regenerate": "python docs/figures/plot.py",
        "recall_vs_k": recall_vs_k(),
        "separation": separation(),
        "arms": arms(),
        "cluster_size": cluster_size(),
        "threshold": threshold(),
        "ncp_encoders": ncp_encoders(),
    }


def write(path=os.path.join(HERE, "data.json")):
    data = build()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    return data


if __name__ == "__main__":
    write()
    print("wrote", os.path.relpath(os.path.join(HERE, "data.json"), ROOT))
