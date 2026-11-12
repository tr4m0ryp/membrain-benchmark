"""The two coordinator follow-ups, run and reported without re-running the
original 14-cell grid.

1. **Close the precondition window by measurement.** The first pass bracketed
   the F15/I4 coverage/purity crossover between k=5 (coverage incomplete) and
   k=10 (coverage complete, purity collapsed) without ever measuring k in
   {7, 8} -- the largest cluster's own sibling count, and therefore the only
   k values where "bounded, enumerable, one-to-one" could plausibly hold at
   once. This module measures those two k values directly with
   `coverage.coverage_point`, the same function the original grid used.

2. **Fix the headroom basis.** The first pass's chat summary (not the result
   JSON's own `headroom_table`, which was already single-basis) divided F24's
   published llm_pick/verify_veto numerators -- measured with a 0.6000 control
   on a 180-query stratified subsample -- by a denominator built from this
   sweep's full-360-query plain_vector control (0.6194). `headroom.py:
   corrected_headroom` fixes that: control and both numerators stay pinned to
   F24's own subsample values, and only the ceiling varies, with every ceiling
   that was not itself measured on the subsample labeled as an assumption.

No new bootstrap over recall/precision/MRR is run: the ceiling values needed
for (2) are either the paper's own quoted R@5 or already-measured full-corpus
recall@k figures from the first run (k=10/20/50 all saturate at 1.0000, per
`results/k-sweep/ksweep-20260801T184604Z.md`), so reusing them costs nothing.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

from brain.config import CORPUS_DIR, ensure_dir, results_dir
from brain.corpus import CorpusHashMismatch, load_corpus
from brain.eval.results import environment, jsonable

from brain.experiments.gate_k_sweep import registration as reg
from brain.experiments.gate_k_sweep.coverage import CoveragePoint, coverage_point
from brain.experiments.gate_k_sweep.headroom import corrected_headroom

__all__ = ["AddendumResult", "run", "render_markdown", "write_addendum_report", "main"]

#: Full-corpus plain_vector recall@k already measured in the first pass
#: (`results/k-sweep/ksweep-20260801T184604Z.md`), reused rather than re-bootstrapped.
_MEASURED_FULL_CORPUS_RECALL: dict[int, float] = {10: 1.0000, 20: 1.0000, 50: 1.0000}


class AddendumResult:
    def __init__(
        self,
        *,
        coverage: dict[tuple[str, int], CoveragePoint],
        headroom: dict[str, Any],
        corpus_sha_before: str,
        corpus_sha_after: str,
        wall_clock: float,
        source_run: str,
    ) -> None:
        self.coverage = coverage
        self.headroom = headroom
        self.corpus_sha_before = corpus_sha_before
        self.corpus_sha_after = corpus_sha_after
        self.wall_clock = wall_clock
        self.source_run = source_run

    def to_dict(self) -> dict[str, Any]:
        by_arm_k: dict[str, dict[str, Any]] = {}
        for (arm, k), point in sorted(self.coverage.items()):
            by_arm_k.setdefault(arm, {})[str(k)] = point.to_dict()
        return {
            "addendum_of": self.source_run,
            "reason": reg.ADDENDUM_REASON,
            "precondition_at_k": {
                "k": list(reg.ADDENDUM_K),
                "by_arm": by_arm_k,
            },
            "corrected_headroom": self.headroom,
            "corpus_integrity": {
                "expected_content_sha256": reg.EXPECTED_CONTENT_SHA256,
                "content_sha256_before_run": self.corpus_sha_before,
                "content_sha256_after_run": self.corpus_sha_after,
                "unchanged": (
                    self.corpus_sha_before == self.corpus_sha_after == reg.EXPECTED_CONTENT_SHA256
                ),
            },
            "wall_clock_seconds": round(float(self.wall_clock), 3),
        }


def run(*, source_run: str, progress: Any = None) -> AddendumResult:
    started = time.perf_counter()
    corpus = load_corpus(CORPUS_DIR / "synthetic")
    sha_before = corpus.content_hash
    if sha_before != reg.EXPECTED_CONTENT_SHA256:
        raise CorpusHashMismatch(
            f"loaded corpus sha {sha_before} does not match the pre-registered "
            f"{reg.EXPECTED_CONTENT_SHA256}; refusing to run the addendum on a different corpus"
        )

    coverage: dict[tuple[str, int], CoveragePoint] = {}
    for arm in reg.FREE_ARMS:
        for k in reg.ADDENDUM_K:
            if progress:
                progress(f"{arm} k={k}: precondition coverage (addendum)")
            coverage[(arm, k)] = coverage_point(arm, corpus, k)

    ceiling_by_k = {5: reg.PUBLISHED_F24["r_at_5_paper"], **_MEASURED_FULL_CORPUS_RECALL}
    headroom = corrected_headroom(ceiling_by_k)

    corpus_after = load_corpus(CORPUS_DIR / "synthetic")
    return AddendumResult(
        coverage=coverage,
        headroom=headroom,
        corpus_sha_before=sha_before,
        corpus_sha_after=corpus_after.content_hash,
        wall_clock=time.perf_counter() - started,
        source_run=source_run,
    )


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _ci(block: dict[str, Any]) -> str:
    return f"{_fmt(block['mean'])} [{_fmt(block['lower'])}, {_fmt(block['upper'])}]"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Gate k-sweep addendum: precondition at k in {7, 8}, corrected headroom basis",
        "",
        f"Addendum of: `{payload['addendum_of']}`",
        f"Reason: {payload['reason']}",
        "",
        f"Corpus SHA unchanged: {payload['corpus_integrity']['unchanged']}",
        "",
        "## 1. F15/I4 precondition, measured at k in {7, 8}",
        "",
        "| arm | k | sibling coverage | competitor purity | coverage complete |",
        "|---|---|---|---|---|",
    ]
    cov = payload["precondition_at_k"]["by_arm"]
    for arm in sorted(cov):
        for k in sorted((int(k) for k in cov[arm]), key=int):
            cell = cov[arm][str(k)]
            lines.append(
                f"| {arm} | {k} | {_ci(cell['sibling_coverage'])} "
                f"| {_ci(cell['competitor_purity'])} | {cell['precondition_satisfiable']} |"
            )
    lines += [
        "",
        "## 2. Corrected headroom -- single basis, ceiling varied only",
        "",
        f"Basis: {payload['corrected_headroom']['basis']}",
        "",
        "| k | ceiling source | R (ceiling) | A_ctrl (0.6000, subsample) | gap | "
        "H(llm_pick) | H(verify_veto) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["corrected_headroom"]["rows"]:
        lines.append(
            f"| {row['k']} | {row['ceiling_source']} | {_fmt(row['R_ceiling'])} "
            f"| {_fmt(row['A_ctrl_subsample_180q'])} | {_fmt(row['gap'])} "
            f"| {_fmt(row['H_llm_pick'])} | {_fmt(row['H_verify_veto'])} |"
        )
    lines += ["", f"Wall clock: {_fmt(payload.get('wall_clock_seconds'), 1)}s"]
    return "\n".join(lines)


def write_addendum_report(
    result: AddendumResult, *, command: str, directory: Path | str
) -> tuple[Path, Path, dict[str, Any]]:
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"ksweep-addendum-{stamp}.json"
    md_path = root / f"ksweep-addendum-{stamp}.md"
    payload = result.to_dict()
    payload["command"] = command
    payload["environment"] = environment()
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    return json_path, md_path, payload


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    source = args[0] if args else "results/ksweep-20260801T184604Z.json"
    command = "python -m brain.experiments.gate_k_sweep.addendum " + " ".join(args)
    result = run(source_run=source, progress=lambda m: print(f"  {m}", flush=True))
    json_path, md_path, _ = write_addendum_report(result, command=command, directory=results_dir("k-sweep"))
    print(f"\n{json_path}\n{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
