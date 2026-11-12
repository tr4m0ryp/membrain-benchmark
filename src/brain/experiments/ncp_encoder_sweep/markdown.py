"""Render the encoder-sweep payload to markdown."""

from __future__ import annotations

from typing import Any

__all__ = ["render_markdown"]


def _fmt_delta(delta: dict[str, Any] | None) -> str:
    if delta is None:
        return "n/a"
    mean, lower, upper = delta["mean"] * 100.0, delta["lower"] * 100.0, delta["upper"] * 100.0
    sign = "conclusive" if (lower > 0.0 or upper < 0.0) else "not conclusive"
    return f"{mean:+.2f} [{lower:+.2f}, {upper:+.2f}] ({sign})"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    p = lines.append
    p(f"# NCP -- sweep the winner: encoder x neighbourhood size ({payload['created_at']})")
    p("")
    p(f"Command: `{payload['command']}`")
    p("")
    p(">F26 measured NCP's local-df rule against SIF and global-IDF with the encoder frozen at "
      "all-MiniLM-L6-v2 and k_neighbours frozen at 8, and the pre-registered cell FAILED the bar "
      "(+0.83 over SIF, +1.94 over global-IDF, neither conclusive). Per this project's improvement "
      "discipline that is LOST A COMPARISON, not proven closed. This run sweeps both frozen knobs.")
    p("")
    p("## Pre-registration (fixed before any number below was read)")
    p("")
    p("```json")
    import json as _json
    p(_json.dumps(payload["preregistration"], indent=2, sort_keys=True))
    p("```")
    p("")
    p("## Per-encoder headline")
    p("")
    p("| encoder | dim | plain_vector | best local | best global | best SIF | "
      "primary-cell delta (local - global) | sibling cos floor |")
    p("|---|---|---|---|---|---|---|---|")
    for s in payload["encoder_summaries"]:
        p(f"| {s['encoder']} | {s['dim']} | {_fmt_pct(s['plain_vector_disambiguation'])} | "
          f"{_fmt_pct(s['best_local_disambiguation'])} | {_fmt_pct(s['best_global_disambiguation'])} | "
          f"{_fmt_pct(s['best_sif_disambiguation'])} | "
          f"{_fmt_delta(s['primary_cell']['delta'])} | {_fmt_pct(s['sibling_cosine_floor'])} |")
    p("")
    p("**Max-over-cells reading (selection-optimistic, reported alongside the primary cell, not "
      "instead of it):**")
    p("")
    p("| encoder | best local - best global | best local - best SIF |")
    p("|---|---|---|")
    for s in payload["encoder_summaries"]:
        p(f"| {s['encoder']} | {_fmt_delta(s['best_local_minus_best_global']['delta'])} | "
          f"{_fmt_delta(s['best_local_minus_best_sif']['delta'])} |")
    p("")
    p("## Strict one-variable paired ablation: local vs global, both win directions counted")
    p("")
    p("Same gamma, same alpha, same query side, same k: only WHERE df is measured changes.")
    p("")
    p("| encoder | cells compared (all k) | local wins conclusively | global wins conclusively | "
      "positive point estimate | mean delta (pts) |")
    p("|---|---|---|---|---|---|")
    for s in payload["encoder_summaries"]:
        c = s["strict_matched_win_counts_all_k"]
        p(f"| {s['encoder']} | {c['cells_compared']} | {c['local_beats_global_conclusively']} | "
          f"{c['global_beats_local_conclusively']} | {c['positive_point_estimate']} | "
          f"{c['mean_delta_points']} |")
    p("")
    p("### Broken out by k_neighbours")
    p("")
    p("| encoder | k | cells | local wins | global wins | positive | mean delta (pts) |")
    p("|---|---|---|---|---|---|---|")
    for s in payload["encoder_summaries"]:
        for k, c in sorted(s["strict_matched_win_counts_by_k"].items(), key=lambda kv: int(kv[0])):
            p(f"| {s['encoder']} | {k} | {c['cells_compared']} | "
              f"{c['local_beats_global_conclusively']} | {c['global_beats_local_conclusively']} | "
              f"{c['positive_point_estimate']} | {c['mean_delta_points']} |")
    p("")
    p("## Sibling-cosine floor per encoder")
    p("")
    p("F26 predicted the sibling cosine would fall to 0.155 under gamma and bottomed at 0.5153 "
      "(MiniLM, k=8). If this floor moves materially across encoders, the diagnosis (contextual "
      "embeddings of distinctive tokens still carry surrounding boilerplate) is a property of "
      "MiniLM's embedding space, not of pooling in general.")
    p("")
    p("| encoder | sibling cosine floor | cell |")
    p("|---|---|---|")
    for s in payload["encoder_summaries"]:
        p(f"| {s['encoder']} | {_fmt_pct(s['sibling_cosine_floor'])} | {s['sibling_cosine_floor_cell']} |")
    p("")
    p("## Correctness assertions")
    p("")
    for s in payload["encoder_summaries"]:
        p(f"- `{s['encoder']}`: gamma=0 identity to mean pooling holds: **{s['gamma_zero_identity_ok']}**")
    p(f"- frozen corpus unchanged across the whole run: **{payload['corpus']['unchanged']}**")
    p(f"- corpus content SHA-256: `{payload['corpus']['content_sha256']}`")
    p("")
    p("## Verdict")
    p("")
    v = payload["verdict"]
    p(f"- any (encoder, k) where local beats global/SIF CONCLUSIVELY: **{v['any_local_conclusive_win']}**"
      f" ({v['local_conclusive_win_count']} of {v['total_matched_cells']} matched cells across the "
      "whole sweep)")
    p(f"- any (encoder, k) where global/SIF beats local CONCLUSIVELY: **{v['any_global_conclusive_win']}**"
      f" ({v['global_conclusive_win_count']} of {v['total_matched_cells']})")
    p(f"- **{v['headline']}**")
    p("")
    p("## Runtime")
    p("")
    p(f"- encoders run: {payload['encoders_run']}")
    p(f"- encoders skipped (budget): {payload['encoders_skipped']}")
    p(f"- wall clock: {payload['wall_clock_seconds']} s (budget {payload['preregistration']['runtime_budget_seconds']} s)")
    p("")
    return "\n".join(lines) + "\n"
