"""Is the v1-to-v2 move a change of DISTRIBUTION or a change of DRAW?

Generates throwaway corpora at the v1 and v2 sizes under other seeds, runs the
frozen ceiling on the same-topic long cell, and reports the spread. Nothing here
is frozen, nothing here is reported as a result, and the released v2 stays the
seed-0 corpus chosen before any of this ran. The question is only whether the
seed-to-seed spread at the v1 size covers the gap between 0.8686 and 0.7820.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np

from brain.corpus.relations import INTERFERENCE, SUPERSESSION
from brain.corpus.relations.generate import (
    RelationsConfig, generate_relations, scaled_relations_config,
)
from brain.experiments.gate_threshold import ceiling as C
from brain.experiments.gate_threshold import distributions as D


def cell(corpus, embedder):
    ids, vectors = D.encode_corpus(corpus, embedder)
    rows = D.population(D.all_pair_rows(corpus, ids, D.gram_matrix(vectors)), "derivable")
    rows = [r for r in D.stratify(rows, same_topic_only=True) if r.regime == "long"]
    sup, inter = D.by_class(rows, SUPERSESSION), D.by_class(rows, INTERFERENCE)
    found = C.best_threshold(sup, inter)
    return {
        "ceiling": found.balanced_accuracy, "tau": found.tau_star,
        "n_S": int(sup.size), "n_I": int(inter.size),
        "clusters": len({r.cluster_id for r in rows}),
        "mean_S": float(sup.mean()), "mean_I": float(inter.mean()),
    }


def main() -> int:
    from brain.embed.embedder import get_embedder

    embedder = get_embedder()
    plans = [("v1-size", RelationsConfig(), range(0, 8)),
             ("v2-size", scaled_relations_config(3), range(0, 4))]
    for label, base, seeds in plans:
        rows = []
        for seed in seeds:
            corpus = generate_relations(dataclasses.replace(base, seed=int(seed)))
            got = cell(corpus, embedder)
            rows.append(got)
            flag = "  <- FROZEN, the reported corpus" if seed == 0 else ""
            print(f"  {label} seed={seed}: ceiling={got['ceiling']:.4f} tau*={got['tau']:.4f} "
                  f"n_S={got['n_S']:<4} n_I={got['n_I']:<5} clusters={got['clusters']:<3} "
                  f"E[s|S]={got['mean_S']:.4f} E[s|I]={got['mean_I']:.4f}{flag}", flush=True)
        values = np.array([r["ceiling"] for r in rows])
        print(f"  {label}: mean={values.mean():.4f} sd={values.std(ddof=1):.4f} "
              f"min={values.min():.4f} max={values.max():.4f} n_seeds={values.size}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
