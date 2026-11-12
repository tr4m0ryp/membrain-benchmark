"""K-I4 evaluated mechanically, the headroom stated, and the two artifacts.

Pre-registered (proposal-v3, restated verbatim, not loosely): *I4 dies if
verify-and-veto fails to beat LLM-picks-from-top-k on disambiguation, OR if it
beats it only by discarding correct items at a higher rate (the tradeoff must be
favourable on both axes).*

Every clause is a function of measured numbers; nothing takes a threshold from
the result it judges, and a fired criterion is returned as the result rather than
softened. Both axes print side by side in every view: a headline gain with the
rejection rate omitted is the reporting failure the Mnemonic Similarity Task's
Lure Discrimination Index exists to prevent.

    python -m brain.experiments.gate_veto.report --seeds 5 --k 5 [--smoke 10]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from brain.config import CORPUS_DIR, ensure_dir, results_dir
from brain.corpus import load_corpus
from brain.corpus.types import Corpus
from brain.eval.bootstrap import DEFAULT_RESAMPLES
from brain.eval.results import corpus_provenance, environment, jsonable
from brain.eval.runner import DEFAULT_K
from brain.experiments.gate_veto.runner import (
    ARM_A, ARM_B, AXES, CONTROL, VetoRun, cost_estimate, run_veto_gate, safe_paired_delta)

__all__ = ["K_I4", "FROZEN_SHA", "RECALL_AT_5_CEILING", "AXIS_DEFINITIONS", "directory_sha",
           "evaluate_k_i4", "headroom", "build_payload", "render_markdown", "write_report", "main"]

K_I4 = ("I4 dies if verify-and-veto fails to beat llm-pick on disambiguation, OR if it beats it "
        "only by discarding correct items at a higher rate. The tradeoff must be favourable on "
        "BOTH axes.")
#: The frozen corpus every result to date is stated against (rules.md).
FROZEN_SHA = "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb"
#: The diagnostic that redirected the project (retrieved, then out-ranked). It
#: is the ceiling that bounds both arms.
RECALL_AT_5_CEILING = 0.958
SCHEMA = "veto-result-v1"
AXIS_DEFINITIONS = {
    "disambiguation": "eligible queries whose correct trace outranks every sibling",
    "rejection_rate": (
        "CORRECT-ITEM REJECTION. Over queries whose correct trace was a candidate, the fraction "
        "where the arm's own reject act fired on it. verify_veto: the boolean veto fired. "
        "llm_pick and plain_vector: a sibling was placed ahead of the correct trace, which is how "
        "a selector discards. Near-dual to disambiguation for a selector, independent for a veto"),
    "demotion_rate": ("one formula for every arm: correct trace retrieved but out-ranked by a "
                      "sibling. The apples-to-apples harm number"),
    "harm_rate": ("regression against the control: of the queries plain_vector already "
                  "disambiguated, the fraction this arm broke")}


def directory_sha(root: Path) -> str:
    """Digest over every file in `root`, a tripwire beside the manifest hash."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        digest.update(path.name.encode("utf-8") + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _point(run: VetoRun, arm: str, axis: str) -> float:  # cluster-bootstrap point estimate
    return float(run.intervals[arm][axis].mean)


def evaluate_k_i4(run: VetoRun, *, n_resamples: int = DEFAULT_RESAMPLES) -> dict[str, Any]:
    """The verdict, computed from the numbers. A fired criterion IS the result."""
    if missing := [n for n in (ARM_A, ARM_B) if n not in run.intervals]:
        return {"criterion": K_I4, "status": "undecided", "blocked": dict(run.blocked),
                "reasons": [f"missing required arm(s): {missing}"]}
    gain_b, gain_a = _point(run, ARM_B, "disambiguation"), _point(run, ARM_A, "disambiguation")
    rej_b, rej_a = _point(run, ARM_B, "rejection_rate"), _point(run, ARM_A, "rejection_rate")
    delta = {axis: safe_paired_delta(run.groups[ARM_B][axis], run.groups[ARM_A][axis],
                                     n_resamples=n_resamples)
             for axis in ("disambiguation", "rejection_rate")}
    delta_gain, delta_rej = delta["disambiguation"], delta["rejection_rate"]
    beats, no_more = bool(gain_b > gain_a), bool(rej_b <= rej_a)
    ok, reasons = bool(beats and no_more), []
    if not beats:
        reasons.append(f"verify_veto does not beat llm_pick on disambiguation: {gain_b:.4f} vs "
                       f"{gain_a:.4f} (delta {(gain_b - gain_a) * 100:+.2f} points)")
    if beats and not no_more:
        reasons.append(f"verify_veto beats llm_pick only by discarding correct items at a "
                       f"higher rate: {rej_b:.4f} vs {rej_a:.4f} "
                       f"(delta {(rej_b - rej_a) * 100:+.2f} points)")
    return {
        "criterion": K_I4, "status": "survives" if ok else "killed",
        "killed": not ok, "survives": ok,
        "beats_on_disambiguation": beats, "rejects_no_more_than_arm_a": no_more,
        "verify_veto_disambiguation": gain_b, "llm_pick_disambiguation": gain_a,
        "verify_veto_rejection_rate": rej_b, "llm_pick_rejection_rate": rej_a,
        "paired_delta_disambiguation_b_minus_a": delta_gain.to_dict(),
        "paired_delta_rejection_b_minus_a": delta_rej.to_dict(),
        # Strength qualifier, reported beside the verdict and never in place of
        # it: a point estimate whose paired interval spans zero is not a win.
        "interval_conclusive": (bool(delta_gain.lower > 0.0) if beats
                                else bool(delta_gain.upper < 0.0)),
        "reasons": reasons}


def headroom(run: VetoRun) -> dict[str, Any]:
    """What fraction of the gap to the recall@5 ceiling each arm captured."""
    control, arms = _point(run, CONTROL, "disambiguation"), {}
    gap = RECALL_AT_5_CEILING - control
    for name in run.intervals:
        value = _point(run, name, "disambiguation")
        arms[name] = {
            "disambiguation": value,
            "gain_over_control_points": round((value - control) * 100.0, 4),
            "headroom_captured": (None if abs(gap) < 1e-12
                                  else round((value - control) / gap, 4))}
    recall = round(float(run.arms[CONTROL].metrics[0]["recall_at_k"]), 6)
    return {"ceiling_recall_at_5": RECALL_AT_5_CEILING, "control_disambiguation": control,
            "measured_recall_at_k_this_run": recall, "arms": arms,
            "available_headroom_points": round(gap * 100.0, 4)}


def build_payload(
    run: VetoRun, *, command: str, sha_before: str, sha_after: str, smoke: int | None = None,
    estimate: dict[str, Any] | None = None, n_resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Every number this run produced, in one JSON-safe record."""
    verdict = evaluate_k_i4(run, n_resamples=n_resamples)
    provenance = corpus_provenance(run.corpus)
    def intervals(src: dict[str, Any]) -> dict[str, Any]:
        return {n: {a: i.to_dict() for a, i in sorted(v.items())} for n, v in sorted(src.items())}

    return {
        "schema": SCHEMA, "command": command, "smoke_n_queries": smoke, "k_i4": verdict,
        "experiment": "Task 004 -- the veto experiment (I4, kill criterion K-I4)",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "verdict_line": f"VERDICT: K-I4 {verdict['status'].upper()}", "headroom": headroom(run),
        "axis_definitions": AXIS_DEFINITIONS,
        "axes": {name: {axis: run.intervals[name][axis].to_dict() for axis in AXES}
                 for name in sorted(run.intervals)},
        "delta_vs_control": intervals(run.deltas), "seed_intervals": intervals(run.seed_intervals),
        "per_seed_metrics": {name: arm.metrics for name, arm in sorted(run.arms.items())},
        "arm_params": {name: (arm.describe or {}).get("params", {})
                       for name, arm in sorted(run.arms.items())},
        "pairing_check": run.pairing,
        "blocked_arms": dict(sorted(run.blocked.items())), "config": {
            "k": run.k, "seeds": list(run.seeds), "bootstrap_resamples": int(n_resamples),
            "resampling_unit": "interference cluster",
            "seed_pooling": "mean over seeds per query, then resample clusters",
            "note": ("the cluster bootstrap carries the verdict; the across-seed t interval is "
                     "legitimately zero-width because every arm here is deterministic given the "
                     "frozen corpus and the on-disk response cache")},
        "corpus": {**provenance, "frozen_sha_expected": FROZEN_SHA,
                   "frozen_sha_matches": provenance["content_sha256"] == FROZEN_SHA,
                   "directory_sha_before": sha_before, "directory_sha_after": sha_after,
                   "directory_unchanged": sha_before == sha_after},
        "llm_accounting": {
            name: {key: arm.stats.get(key) for key in sorted(arm.stats)
                   if key not in {"n_indexed", "queries"}}
            for name, arm in sorted(run.arms.items()) if "api_calls" in arm.stats},
        "cost_usd_reported": round(run.cost_usd(), 4), "cost_estimate": estimate,
        "wall_clock_seconds": round(float(run.wall_clock), 2), "environment": environment(),
    }


def _row(payload: dict[str, Any], name: str) -> str:
    axes = payload["axes"].get(name, {})
    cells = ["n/a" if axes.get(a) is None else
             f"{axes[a]['mean']:.4f} [{axes[a]['lower']:.4f}, {axes[a]['upper']:.4f}]"
             for a in AXES]
    captured = payload["headroom"]["arms"].get(name, {}).get("headroom_captured")
    cells.append("n/a" if captured is None else f"{captured:.3f}")
    return f"| {name} | " + " | ".join(cells) + " |"


def render_markdown(payload: dict[str, Any]) -> str:
    verdict, head = payload["k_i4"], payload["headroom"]
    rows = [_row(payload, n) for n in (CONTROL, ARM_A, ARM_B) if n in payload["axes"]]
    lines = [
        "# Task 004 -- the veto experiment (I4)", "", f"**{payload['verdict_line']}**", "",
        f"> {K_I4}", "", "## Both axes, together", "",
        "| arm | disambiguation | correct-item rejection | demotion | harm vs control | headroom |",
        "|---|---|---|---|---|---|",
        *rows, "",
        f"Ceiling recall@5 = {head['ceiling_recall_at_5']} (measured this run: "
        f"{head['measured_recall_at_k_this_run']}). Control disambiguation "
        f"{head['control_disambiguation']:.4f}, so {head['available_headroom_points']:.2f} "
        "points of headroom were available.",
        "", "## K-I4", "", f"- status: **{verdict['status']}**"]
    if "beats_on_disambiguation" in verdict:
        delta = verdict["paired_delta_disambiguation_b_minus_a"]
        lines += [
            f"- beats llm_pick on disambiguation: {verdict['beats_on_disambiguation']} "
            f"({verdict['verify_veto_disambiguation']:.4f} vs "
            f"{verdict['llm_pick_disambiguation']:.4f})",
            f"- rejects correct items no more often: {verdict['rejects_no_more_than_arm_a']} "
            f"({verdict['verify_veto_rejection_rate']:.4f} vs "
            f"{verdict['llm_pick_rejection_rate']:.4f})",
            f"- paired B-A disambiguation delta: {delta['mean'] * 100:+.2f} points "
            f"[{delta['lower'] * 100:+.2f}, {delta['upper'] * 100:+.2f}]",
            f"- interval conclusive: {verdict['interval_conclusive']}"]
    lines += [f"- reason: {r}" for r in verdict.get("reasons", [])]
    corpus, config = payload["corpus"], payload["config"]
    lines += ["", "## Provenance", "",
        f"- corpus content SHA: `{corpus['content_sha256']}` (frozen: {corpus['frozen_sha_matches']})",
        f"- corpus directory unchanged across the run: {corpus['directory_unchanged']}",
        f"- candidate sets byte-identical across arms: {payload['pairing_check'].get('identical')}",
        f"- reported LLM spend this run: ${payload['cost_usd_reported']:.4f}",
        f"- seeds: {config['seeds']}, k={config['k']}, "
        f"resamples={config['bootstrap_resamples']}", ""]
    return "\n".join(lines)


def write_report(payload: dict[str, Any], directory: Path | str) -> tuple[Path, Path]:
    """Write `results/veto/veto-<stamp>.{json,md}` and return both paths."""
    root = ensure_dir(Path(directory))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "veto-smoke" if payload.get("smoke_n_queries") else "veto"
    stem, suffix = f"{prefix}-{stamp}", 1
    while (root / f"{stem}.json").exists() or (root / f"{stem}.md").exists():
        stem, suffix = f"{prefix}-{stamp}-{suffix}", suffix + 1
    json_path, md_path = root / f"{stem}.json", root / f"{stem}.md"
    body = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    json_path.write_text(body + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _factories() -> dict[str, Any]:
    from brain.baselines.plain_vector import PlainVectorRetriever  # lazy: torch is heavy
    from brain.baselines.veto import LlmPickRetriever, VerifyVetoRetriever
    return {CONTROL: PlainVectorRetriever, ARM_A: LlmPickRetriever, ARM_B: VerifyVetoRetriever}


def _subsample(corpus: Corpus, n: int) -> Corpus:
    """`n` evenly spaced queries over the id-sorted list. Evenly spaced, not the
    first n and not every `len//n`-th: an integer step truncates to the head of
    the list whenever n does not divide the total, silently dropping whole
    clusters and unbalancing the cluster sizes. Spacing by `i * len / n` keeps
    every cluster represented, which the cluster bootstrap needs."""
    queries: list[Any] = sorted(corpus.queries, key=lambda q: q.query_id)
    want = max(1, min(int(n), len(queries)))
    picked = sorted({int(i * len(queries) / want) for i in range(want)})
    return Corpus(corpus.episodes, [queries[i] for i in picked], corpus.manifest)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m brain.experiments.gate_veto.report")
    for flag, kw in (("--seeds", {"type": int, "default": 5}),
                     ("--k", {"type": int, "default": DEFAULT_K}),
                     ("--smoke", {"type": int, "default": 0, "help": "N queries only"}),
                     ("--resamples", {"type": int, "default": DEFAULT_RESAMPLES}),
                     ("--workers", {"type": int, "default": 8}),
                     ("--no-prefetch", {"action": "store_true"}),
                     ("--results-dir", {"default": str(results_dir("veto"))})):
        parser.add_argument(flag, **kw)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args, directory = _parse(raw), CORPUS_DIR / "synthetic"
    command = "python -m brain.experiments.gate_veto.report " + " ".join(map(shlex.quote, raw))
    sha_before, corpus = directory_sha(directory), load_corpus(directory)  # verifies the hash
    if corpus.content_hash != FROZEN_SHA:
        raise RuntimeError(f"frozen corpus SHA changed: {corpus.content_hash} != {FROZEN_SHA}")
    full_queries = len(corpus.queries)
    corpus = _subsample(corpus, args.smoke) if args.smoke else corpus
    print(f"corpus sha={corpus.content_hash[:12]} eps={len(corpus.episodes)} "
          f"queries={len(corpus.queries)}/{full_queries} k={args.k} "
          f"seeds={args.seeds} smoke={bool(args.smoke)}", flush=True)
    run = run_veto_gate(
        corpus, factories=_factories(), seeds=tuple(range(max(1, int(args.seeds)))),
        k=int(args.k), n_resamples=int(args.resamples), prefetch=not args.no_prefetch,
        prefetch_workers=int(args.workers), progress=lambda m: print(f"  {m}", flush=True))
    estimate = cost_estimate(run, n_smoke=len(corpus.queries),
                             n_full=full_queries) if args.smoke else None
    if (sha_after := directory_sha(directory)) != sha_before:  # immutability, enforced
        raise RuntimeError("the frozen corpus directory changed during the run")
    payload = build_payload(run, command=command, sha_before=sha_before, sha_after=sha_after,
                            smoke=int(args.smoke) or None, estimate=estimate,
                            n_resamples=int(args.resamples))
    print("\n" + render_markdown(payload))
    json_path, md_path = write_report(payload, args.results_dir)
    if estimate:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        if estimate["over_budget"]:
            print("STOP: projected full-pass cost exceeds the 25 USD budget.", flush=True)
    for name, reason in sorted(run.blocked.items()):
        print(f"BLOCKED {name}: {reason}", flush=True)
    print(f"{json_path}\n{md_path}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
