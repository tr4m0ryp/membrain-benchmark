"""Orchestrates the full grid: every free arm x every pre-registered k.

Loads the frozen corpus (verifying its hash on the way in, per the loader's
default), runs `point.evaluate_point` and `coverage.coverage_point` for every
cell, and assembles everything -- pre-registration block first -- into one
JSON-able payload. `run()` is the single entry point the CLI and the tests use.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from brain.config import CORPUS_DIR
from brain.corpus import CorpusHashMismatch, load_corpus

from brain.experiments.gate_k_sweep import registration as reg
from brain.experiments.gate_k_sweep.coverage import CoveragePoint, coverage_point
from brain.experiments.gate_k_sweep.headroom import headroom_table
from brain.experiments.gate_k_sweep.point import KPoint, evaluate_point

__all__ = ["SweepResult", "run"]


class SweepResult:
    """Every grid cell plus the derived answers, ready for `report.write_sweep_report`."""

    def __init__(
        self,
        *,
        points: dict[tuple[str, int], KPoint],
        coverage: dict[tuple[str, int], CoveragePoint],
        corpus_sha_before: str,
        corpus_sha_after: str,
        wall_clock: float,
    ) -> None:
        self.points = points
        self.coverage = coverage
        self.corpus_sha_before = corpus_sha_before
        self.corpus_sha_after = corpus_sha_after
        self.wall_clock = wall_clock

    # -- derived answers -----------------------------------------------

    def monotonicity_check(self, arm: str) -> dict[str, Any]:
        """Q2 formal claim: disambiguation(k) is non-increasing in k for an arm
        whose `retrieve(query, k)` is a length-k prefix of one fixed per-query
        score order (true of both plain_vector and bm25, `baselines/base.py
        rank_hits` truncates a full sort). Checked against the measured points,
        not assumed.
        """
        ks = sorted(k for (a, k) in self.points if a == arm)
        values = [self.points[(arm, k)].disambiguation.mean for k in ks]
        violations = [
            (ks[i], values[i], ks[i + 1], values[i + 1])
            for i in range(len(ks) - 1)
            if values[i + 1] > values[i] + 1e-9
        ]
        return {
            "arm": arm,
            "k_grid": ks,
            "disambiguation_by_k": values,
            "non_increasing": not violations,
            "violations": violations,
            "total_drop": (values[0] - values[-1]) if values else math.nan,
        }

    def recall_saturation(self, arm: str) -> dict[str, Any]:
        """Q1: where recall@k stops moving. `plateau_k` is the smallest k after
        which every later grid point is within one percentage point of the
        largest k measured."""
        ks = sorted(k for (a, k) in self.points if a == arm)
        values = [self.points[(arm, k)].recall_at_k.mean for k in ks]
        ceiling = values[-1] if values else math.nan
        plateau_k = None
        for k, v in zip(ks, values):
            if math.isfinite(v) and math.isfinite(ceiling) and abs(ceiling - v) <= 0.01:
                plateau_k = k
                break
        return {
            "arm": arm,
            "k_grid": ks,
            "recall_by_k": values,
            "ceiling_at_max_k": ceiling,
            "plateau_k_within_1pt": plateau_k,
        }

    def to_dict(self) -> dict[str, Any]:
        by_arm_k: dict[str, dict[str, Any]] = {}
        for (arm, k), point in sorted(self.points.items()):
            by_arm_k.setdefault(arm, {})[str(k)] = point.to_dict()
        coverage_by_arm_k: dict[str, dict[str, Any]] = {}
        for (arm, k), point in sorted(self.coverage.items()):
            coverage_by_arm_k.setdefault(arm, {})[str(k)] = point.to_dict()

        ctrl_points = {k: self.points[("plain_vector", k)] for (a, k) in self.points if a == "plain_vector"}
        bm25_points = {k: self.points[("bm25", k)] for (a, k) in self.points if a == "bm25"}
        cov_ctrl = {k: v for (a, k), v in self.coverage.items() if a == "plain_vector"}

        return {
            "pre_registration": reg.pre_registration(),
            "results": {
                "by_arm": by_arm_k,
                "precondition_coverage_by_arm": coverage_by_arm_k,
                "headroom": headroom_table(ctrl_points, bm25_points, cov_ctrl),
            },
            "answers": {
                "q1_recall_saturation": {
                    arm: self.recall_saturation(arm) for arm in reg.FREE_ARMS
                },
                "q2_disambiguation_monotonicity": {
                    arm: self.monotonicity_check(arm) for arm in reg.FREE_ARMS
                },
                "q3_precondition_and_headroom": (
                    "see results.headroom and results.precondition_coverage_by_arm; "
                    "H is NOT computable for llm_pick/verify_veto at any k here (out of scope)"
                ),
            },
            "corpus_integrity": {
                "expected_content_sha256": reg.EXPECTED_CONTENT_SHA256,
                "content_sha256_before_run": self.corpus_sha_before,
                "content_sha256_after_run": self.corpus_sha_after,
                "unchanged": self.corpus_sha_before == self.corpus_sha_after == reg.EXPECTED_CONTENT_SHA256,
            },
            "wall_clock_seconds": round(float(self.wall_clock), 3),
        }


def run(*, progress: Callable[[str], None] | None = None) -> SweepResult:
    """Run the full pre-registered grid on the frozen corpus."""
    started = time.perf_counter()
    corpus = load_corpus(CORPUS_DIR / "synthetic")  # verify=True by default: refuses a drifted corpus
    sha_before = corpus.content_hash
    if sha_before != reg.EXPECTED_CONTENT_SHA256:
        raise CorpusHashMismatch(
            f"loaded corpus sha {sha_before} does not match the pre-registered "
            f"{reg.EXPECTED_CONTENT_SHA256}; refusing to run the sweep on a different corpus"
        )

    points: dict[tuple[str, int], KPoint] = {}
    coverage: dict[tuple[str, int], CoveragePoint] = {}
    for arm in reg.FREE_ARMS:
        for k in reg.K_GRID:
            if progress:
                progress(f"{arm} k={k}: metrics")
            points[(arm, k)] = evaluate_point(arm, corpus, k)
            if progress:
                progress(f"{arm} k={k}: precondition coverage")
            coverage[(arm, k)] = coverage_point(arm, corpus, k)

    # Re-verify on the way out: the sweep must not have touched the on-disk
    # corpus (it only reads), and this proves it rather than assuming it.
    corpus_after = load_corpus(CORPUS_DIR / "synthetic")
    sha_after = corpus_after.content_hash

    return SweepResult(
        points=points,
        coverage=coverage,
        corpus_sha_before=sha_before,
        corpus_sha_after=sha_after,
        wall_clock=time.perf_counter() - started,
    )
