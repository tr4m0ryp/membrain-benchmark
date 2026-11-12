"""Task 007 self-improvement pass: is v2 the same distribution as v1, and is the
power calculation honest?

Nothing here changes a reported number. Every headline comes from the frozen gate;
this script only asks the four questions the task contract requires an answer to:

1. achieved interval half-width against what the cluster count predicts,
2. whether the bootstrap is still clustered correctly at the larger n,
3. whether growing the class shifted the marginal composition,
4. whether v2 and v1 are draws from the same distribution.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from brain.config import RESULTS_DIR, get_rng
from brain.corpus.relations import INTERFERENCE, SUPERSESSION, load_relations
from brain.corpus.relations.freeze import RELATIONS_V2_DIR
from brain.corpus.relations.loader import RELATIONS_DIR
from brain.experiments.gate_threshold import ceiling as C
from brain.experiments.gate_threshold import distributions as D

SPLITS = {"v1": RELATIONS_DIR, "v2": RELATIONS_V2_DIR}
CELLS = (("same_topic", "long"), ("same_topic", "short"), ("marginal", "long"),
         ("marginal", "short"), ("same_topic", "pooled"), ("marginal", "pooled"))


def load_reports(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(root.glob("threshold-*.json")):
        payload = json.loads(path.read_text())
        split = payload.get("task_007", {}).get("split")
        if split:
            out[split] = payload
    return out


def rows_for(split: str) -> tuple[list, dict]:
    corpus = load_relations(SPLITS[split])
    ids, vectors = D.encode_corpus(corpus)
    labelled = D.all_pair_rows(corpus, ids, D.gram_matrix(vectors))
    return D.population(labelled, "derivable"), {"sha": corpus.content_hash}


def cell_rows(rows: list, stratum: str, regime: str) -> list:
    kept = D.stratify(rows, same_topic_only=(stratum == "same_topic"))
    return kept if regime == "pooled" else [r for r in kept if r.regime == regime]


def bootstrap_se(rows: list, *, unit: str, B: int = 4000, seed: int = 0) -> float:
    """SE of the ceiling under cluster or pair resampling. Diagnostic only.

    The pair-level number is NOT a reportable interval (rules.md forbids it); it
    exists so the design effect can be quoted, which is what says whether the
    clustered bootstrap is still doing work at the larger n.
    """
    rng = get_rng(seed, stream="task007-design-effect")
    if unit == "cluster":
        keys, sup_by, int_by = C._cluster_arrays(rows)
        draws = rng.integers(0, len(keys), size=(B, len(keys)))
        reps = [C._ceiling_value(C._joined(sup_by, d), C._joined(int_by, d)) for d in draws]
    else:
        sup = D.by_class(rows, SUPERSESSION)
        inter = D.by_class(rows, INTERFERENCE)
        reps = [C._ceiling_value(rng.choice(sup, sup.size, replace=True),
                                rng.choice(inter, inter.size, replace=True)) for _ in range(B)]
    finite = np.array([r for r in reps if math.isfinite(r)], dtype=np.float64)
    return float(finite.std(ddof=1))


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(RESULTS_DIR) / "task-007"
    reports = load_reports(root)
    assert set(reports) == {"v1", "v2"}, f"expected both reports in {root}, found {sorted(reports)}"

    print("=" * 100)
    print("1. SIDE BY SIDE, derivable population (the gate's primary path)")
    print(f"{'cell':<22}{'split':<5}{'ceiling':>9}{'lower':>9}{'upper':>9}{'half':>8}"
          f"{'tau*':>8}{'n_S':>7}{'n_I':>7}{'clus':>6}{'loco':>8}  verdict")
    for stratum, regime in CELLS:
        for split in ("v1", "v2"):
            block = reports[split]["ceilings"]["derivable"][stratum][regime]
            ceil, k1, boot = block["ceiling"], block["k1"], block["bootstrap"]
            half = 0.5 * (k1["ci_upper"] - k1["ci_lower"])
            loco = block["cross_validated"]
            cv = loco["balanced_accuracy"] if loco.get("available") else float("nan")
            verdict = ("K1 FIRES" if k1["fires"] else "K1 does not fire") + (
                ", interval conclusive" if k1["interval_conclusive"] else ", INTERVAL STRADDLES 0.90")
            print(f"{stratum + '/' + regime:<22}{split:<5}{ceil['balanced_accuracy']:>9.4f}"
                  f"{k1['ci_lower']:>9.4f}{k1['ci_upper']:>9.4f}{half:>8.4f}{ceil['tau_star']:>8.4f}"
                  f"{ceil['n_supersession']:>7}{ceil['n_interference']:>7}{boot['n_clusters']:>6}"
                  f"{cv:>8.4f}  {verdict}")
        print("")

    print("=" * 100)
    print("2. POWER: achieved half-width against what the design predicts")
    for stratum, regime in (("same_topic", "long"), ("same_topic", "short")):
        one = reports["v1"]["ceilings"]["derivable"][stratum][regime]
        two = reports["v2"]["ceilings"]["derivable"][stratum][regime]
        h1 = 0.5 * (one["k1"]["ci_upper"] - one["k1"]["ci_lower"])
        h2 = 0.5 * (two["k1"]["ci_upper"] - two["k1"]["ci_lower"])
        m1, m2 = one["bootstrap"]["n_clusters"], two["bootstrap"]["n_clusters"]
        s1, s2 = one["ceiling"]["n_supersession"], two["ceiling"]["n_supersession"]
        print(f"\n  {stratum}/{regime}")
        print(f"    v1 half={h1:.4f} clusters={m1} n_S={s1}")
        print(f"    v2 half={h2:.4f} clusters={m2} n_S={s2}   achieved shrink {h2 / h1:.3f}x")
        print(f"    predicted by 1/sqrt(clusters): {h1 * math.sqrt(m1 / m2):.4f} "
              f"(shrink {math.sqrt(m1 / m2):.3f}x)  <- the honest prediction, the bootstrap "
              f"block is the cluster")
        print(f"    predicted by 1/sqrt(n_S):      {h1 * math.sqrt(s1 / s2):.4f} "
              f"(shrink {math.sqrt(s1 / s2):.3f}x)  <- what pair-level counting would claim")
        need = two["ceiling"]["balanced_accuracy"] - 0.90
        print(f"    distance of the v2 point estimate to the bar: {need:+.4f}; "
              f"half-width needed to decide: {abs(need):.4f}")
        if h2 > abs(need) and abs(need) > 0:
            factor = (h2 / abs(need)) ** 2
            print(f"    STILL UNDECIDED: needs {factor:.2f}x the clusters, i.e. about "
                  f"{math.ceil(m2 * factor)} pair-bearing clusters, against a bank of 30")

    print("\n" + "=" * 100)
    print("3. COMPOSITION: did growing the class move the marginal mix?")
    for stratum, regime in (("same_topic", "long"), ("marginal", "long")):
        line = []
        for split in ("v1", "v2"):
            ceil = reports[split]["ceilings"]["derivable"][stratum][regime]["ceiling"]
            ratio = ceil["n_interference"] / ceil["n_supersession"]
            line.append(f"{split}: n_S={ceil['n_supersession']} n_I={ceil['n_interference']} "
                        f"I/S={ratio:.2f}")
        print(f"  {stratum}/{regime}  " + "   ".join(line))
    for split in ("v1", "v2"):
        sizes = reports[split]["population_sizes"]["derivable"]
        print(f"  {split} derivable population sizes: "
              + ", ".join(f"{k}={v}" for k, v in sorted(sizes.items())))

    print("\n" + "=" * 100)
    print("4. SAME DISTRIBUTION? two-sample KS on the per-class score distributions")
    scored = {split: rows_for(split) for split in ("v1", "v2")}
    for stratum, regime in (("same_topic", "long"), ("same_topic", "short")):
        for cls in (SUPERSESSION, INTERFERENCE):
            a = D.by_class(cell_rows(scored["v1"][0], stratum, regime), cls)
            b = D.by_class(cell_rows(scored["v2"][0], stratum, regime), cls)
            ks = stats.ks_2samp(a, b)
            print(f"  {stratum}/{regime:<6} {cls:<13} v1 n={a.size:<5} mean={a.mean():.4f} "
                  f"sd={a.std(ddof=1):.4f} | v2 n={b.size:<5} mean={b.mean():.4f} "
                  f"sd={b.std(ddof=1):.4f} | KS D={ks.statistic:.4f} p={ks.pvalue:.4f} "
                  f"{'SAME (p>0.05)' if ks.pvalue > 0.05 else 'DIFFERENT (p<=0.05)'}")

    print("\n" + "=" * 100)
    print("5. IS THE BOOTSTRAP STILL CLUSTERED? design effect at both n")
    print("   (pair-level SE is a DIAGNOSTIC only; rules.md forbids reporting a pair-level interval)")
    for split in ("v1", "v2"):
        rows = cell_rows(scored[split][0], "same_topic", "long")
        se_cluster = bootstrap_se(rows, unit="cluster")
        se_pair = bootstrap_se(rows, unit="pair")
        print(f"  {split} same_topic/long: SE_cluster={se_cluster:.4f} SE_pair={se_pair:.4f} "
              f"design effect (var ratio)={(se_cluster / se_pair) ** 2:.2f}")

    print("\n" + "=" * 100)
    print("6. OPTIMISM: the ceiling is a maximum over tau, which is biased upward at small n")
    for stratum, regime in (("same_topic", "long"), ("same_topic", "short")):
        for split in ("v1", "v2"):
            block = reports[split]["ceilings"]["derivable"][stratum][regime]
            ceil, loco = block["ceiling"], block["cross_validated"]
            n_s, n_i = ceil["n_supersession"], ceil["n_interference"]
            null_ks = 0.8687 * math.sqrt((n_s + n_i) / (n_s * n_i))  # E[D] under H0, ~0.87*sqrt(..)
            print(f"  {stratum}/{regime:<6} {split}: ceiling={ceil['balanced_accuracy']:.4f} "
                  f"loco={loco.get('balanced_accuracy', float('nan')):.4f} "
                  f"gap={ceil['balanced_accuracy'] - loco.get('balanced_accuracy', float('nan')):+.4f} "
                  f"| null-KS scale of the optimism = {0.5 * null_ks:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
