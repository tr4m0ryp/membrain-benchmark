"""The F15/I4 precondition, measured directly: is the retrieved top-k a
BOUNDED, ENUMERABLE competitor set with (close to) one-to-one target/lure
correspondence?

The I4 claim: "Requires a bounded, enumerable competitor set
with one-to-one target/lure correspondence. Our top-5 at recall 0.958
satisfies this. Fails when competitors cannot be exhaustively recovered."
That claim was never checked as a function of k. Two things decide it and both
are computed here, per query, from the RAW ranking (not from `QueryOutcome`,
which only records whether a sibling ranks ahead of the correct trace, not
whether it is present in the window at all):

* **coverage(k)**: does the top-k contain EVERY sibling the corpus defines for
  this query's cluster? Below coverage, the competitor set is not exhaustively
  recovered and the precondition fails outright -- a veto could reject on an
  incomplete lure set. The largest frozen cluster has 8 members (7 siblings),
  so coverage cannot reach 1.0 for those queries until k is large enough to
  hold all 7.
* **purity(k)**: of the k-1 non-target slots in the window, what fraction are
  true siblings (lures) rather than unrelated noise from other clusters?
  "One-to-one target/lure correspondence" is a claim about EVERY candidate
  being a deliberate lure (the Mnemonic Similarity Task framing I4 cites);
  once k exceeds the cluster, most of the window is noise and a veto would
  spend a model call adjudicating candidates that were never competitors.

Retrieval here is exact-and-deterministic for both free arms (both score the
whole corpus and truncate with the shared stable tie-break, `baselines/base.py
rank_hits`), so this is computed once and its determinism is asserted, not
assumed, by comparing two different seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from brain.config import seed_everything
from brain.eval.bootstrap import DEFAULT_RESAMPLES, cluster_bootstrap_ci

__all__ = ["CoveragePoint", "raw_rankings", "score_coverage", "coverage_point"]


def raw_rankings(arm_name: str, corpus: Any, k: int, *, seed: int = 0) -> dict[str, list[str]]:
    """query_id -> top-k trace ids, for one arm at one k."""
    from brain.baselines import build_arm

    seed_everything(seed)
    arm = build_arm(arm_name)
    try:
        arm.index(corpus.episodes)
        return {q.query_id: arm.retrieve(q.text, k) for q in sorted(corpus.queries, key=lambda q: q.query_id)}
    finally:
        if hasattr(arm, "close"):
            arm.close()


def score_coverage(query: Any, ranked_ids: Sequence[str]) -> tuple[str, bool, float]:
    """One query's (cluster_id, full_coverage, purity)."""
    window = list(ranked_ids)
    present = set(window)
    siblings = {str(t) for t in query.distractor_ids}
    correct = str(query.correct_trace_id)
    full_coverage = siblings.issubset(present)
    non_target_slots = [tid for tid in window if tid != correct]
    purity = (
        float(len({tid for tid in non_target_slots if tid in siblings})) / len(non_target_slots)
        if non_target_slots
        else float("nan")
    )
    return str(query.cluster_id), bool(full_coverage), purity


@dataclass(frozen=True, slots=True)
class CoveragePoint:
    """The precondition diagnostic for one (arm, k): coverage and purity, both
    with cluster-BCa intervals."""

    arm: str
    k: int
    coverage: Any  # Interval
    purity: Any  # Interval
    max_cluster_siblings: int
    verified_deterministic: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "k": int(self.k),
            "sibling_coverage": self.coverage.to_dict(),
            "competitor_purity": self.purity.to_dict(),
            "max_cluster_siblings": int(self.max_cluster_siblings),
            "precondition_satisfiable": bool(self.coverage.mean >= 1.0 - 1e-9),
            "verified_deterministic_across_seeds": bool(self.verified_deterministic),
        }


def coverage_point(
    arm_name: str,
    corpus: Any,
    k: int,
    *,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> CoveragePoint:
    """Coverage and purity for one (arm, k), cluster-bootstrapped."""
    by_id = {q.query_id: q for q in corpus.queries}
    rankings_a = raw_rankings(arm_name, corpus, k, seed=0)
    rankings_b = raw_rankings(arm_name, corpus, k, seed=1)
    verified_deterministic = rankings_a == rankings_b

    coverage_groups: dict[str, list[float]] = {}
    purity_groups: dict[str, list[float]] = {}
    for query_id in sorted(rankings_a):
        cluster_id, covered, purity = score_coverage(by_id[query_id], rankings_a[query_id])
        coverage_groups.setdefault(cluster_id, []).append(float(covered))
        if purity == purity:  # skip NaN (k=0 window, never hit on this grid)
            purity_groups.setdefault(cluster_id, []).append(purity)

    generator = rng if rng is not None else np.random.default_rng(0)
    coverage_ci = cluster_bootstrap_ci(
        coverage_groups, alpha=alpha, n_resamples=n_resamples, rng=generator
    )
    purity_ci = cluster_bootstrap_ci(
        purity_groups, alpha=alpha, n_resamples=n_resamples, rng=generator
    )
    max_siblings = max((s - 1 for s in corpus.cluster_sizes().values()), default=0)
    return CoveragePoint(
        arm=arm_name,
        k=int(k),
        coverage=coverage_ci,
        purity=purity_ci,
        max_cluster_siblings=int(max_siblings),
        verified_deterministic=verified_deterministic,
    )
