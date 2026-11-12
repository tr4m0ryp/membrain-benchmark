"""The human-readable half of the artifact: the whole surface, never only the best cell.

Layout order is deliberate. The verdict, the paired deltas and the C-A reordering
test come first, because if NCP is flat in gamma or its margin over weighted
pooling does not clear zero, nothing in the surface tables below matters. The
tables themselves live in `tables`; this module is the order they appear in.
"""

from __future__ import annotations

from typing import Any

from brain.experiments.gate_ncp.expectations import expectation_table, headroom
from brain.experiments.gate_ncp.tables import (
    baseline_table,
    cost_table,
    delta_table,
    incumbent_table,
    reordering_table,
    surface,
)

__all__ = ["render_markdown"]


def render_markdown(payload: dict[str, Any]) -> str:
    verdict, config = payload["verdict"], payload["config"]
    pre = verdict.get("preregistered_cell") or {}
    captured = headroom(payload)
    lines = [
        "# Task 008 -- neighbourhood-conditional pooling (NCP)", "",
        f"**{payload['verdict_line']}**", "",
        f"Run {payload['created_at']} | command `{payload['command']}`", "",
        "> The target was never to beat `llm_pick` (0.9444). It was to approach it at ZERO "
        "per-query cost, because `llm_pick` bills about $0.014 on every query and that "
        "violates this project's founding efficiency premise.", "",
        "## Headline", "",
        f"- plain_vector (control): **{float(verdict['plain_vector_disambiguation']):.4f}**",
        f"- best NCP cell `{verdict['best_cell']}`: "
        f"**{float(verdict['best_disambiguation']):.4f}** "
        f"({float(verdict['best_delta_points']):+.2f} points)",
        f"- PRE-REGISTERED cell `{pre.get('arm', 'n/a')}`: "
        f"**{float(pre.get('disambiguation', float('nan'))):.4f}** "
        f"({float(pre.get('delta_points', float('nan'))):+.2f} points), named before the run so "
        "the best cell can be read against a cell chosen without hindsight",
        f"- beats plain_vector by >= 5 points: **{bool(verdict['beats_plain_vector_by_5_points'])}**",
        f"- best weighted-pooling incumbent `{verdict.get('best_incumbent_cell')}`: "
        f"**{float(verdict.get('best_incumbent_disambiguation') or float('nan')):.4f}**, so the "
        f"local rule is "
        f"{float(verdict.get('local_minus_best_incumbent_points') or float('nan')):+.2f} points "
        "ahead of it at each family's own best cell. See the PAIRED interval below before "
        "reading that as an improvement",
        f"- **headroom to `llm_pick` captured: {captured}**. That is the number this task "
        "exists to move, and it is what a free method has to earn against a paid one",
        "- **selection caution, stated up front:** the best cell is a maximum over "
        f"{len([c for c in payload['cells'] if c['weights'] == 'local'])} local configurations "
        f"against {len([c for c in payload['cells'] if c['weights'] != 'local'])} incumbent "
        "ones, so the max-over-cells comparison is optimistic in NCP's own favour. The "
        "pre-registered cell and the matched paired deltas are the honest readings",
    ]
    lines += baseline_table(payload)
    lines += delta_table(payload)
    lines += reordering_table(payload)
    lines += expectation_table(payload)
    lines += ["", "## The pre-registered (gamma, alpha) surface", "",
              "Every cell is reported, not only the best one. `gamma = 0` is mean pooling "
              "exactly, so its row must equal plain_vector at every alpha."]
    lines += surface(payload, f"PRIMARY: local df, k = {config.get('primary_k', 8)}, "
                      "query side = mean",
                      "The neighbourhood-conditional rule at the pre-registered primary setting.",
                      weights="local", k_neighbours=payload["preregistration"]["primary_k"],
                      query_side="mean")
    lines += surface(payload, "Query-side variant: global-IDF weighted query",
                      "Same document side, query pooled with global IDF weights instead of "
                      "mean pooling.",
                      weights="local", k_neighbours=payload["preregistration"]["primary_k"],
                      query_side="idf")
    for k in payload["preregistration"]["k_grid"]:
        if int(k) == int(payload["preregistration"]["primary_k"]):
            continue
        lines += surface(payload, f"Neighbourhood size sensitivity: k = {k}, query side = mean",
                          "Pre-registered secondary, reported in full.",
                          weights="local", k_neighbours=int(k), query_side="mean")
    lines += incumbent_table(payload)
    lines += cost_table(payload)
    corpus = payload["corpus"]
    lines += ["", "## Provenance", "",
              f"- corpus content SHA-256: `{corpus.get('content_sha256')}`",
              f"- every frozen split unchanged across the run: **{corpus.get('unchanged')}**"]
    for name, state in sorted((corpus.get("shas_after") or {}).items()):
        if state.get("present"):
            lines.append(f"  - `{name}`: content `{state['content_sha256']}` "
                         f"(expected prefix `{state['expected_prefix']}`, "
                         f"matches {state['matches']})")
    lines += [f"- episodes / queries / clusters: {corpus.get('n_episodes')} / "
              f"{corpus.get('n_queries')} / {corpus.get('n_clusters')}",
              f"- k = {config['k']}, seeds = {config['seeds']}, "
              f"resamples = {config['bootstrap_resamples']}, "
              f"resampling unit = {config['resampling_unit']}",
              f"- cells evaluated: {payload['diagnostics'].get('n_cells')}, "
              f"wall clock {payload['wall_clock_seconds']} s",
              f"- environment: {payload['environment']}", ""]
    return "\n".join(lines)
