"""Every table the NCP report prints, and the arithmetic each one needs.

Split out of `markdown` so that module is only the assembly order. Two rules are
enforced by shape here rather than by discipline: sibling cosine sits in the same
row as disambiguation in every surface table, so a cosine drop can never be read
as a win on its own; and the paired-delta table exists because two overlapping
marginal intervals settle nothing about whether local df beats global df.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["interval_cell", "points_cell", "surface", "baseline_table", "delta_table",
           "reordering_table", "incumbent_table", "cost_table", "select"]

#: Sibling cosine and disambiguation are adjacent columns on purpose: C-A dropped
#: sibling cosine 0.824 to 0.704 and moved disambiguation by nothing, so the two
#: numbers must never be readable apart.
_SURFACE_HEAD = ("| gamma | alpha | disambiguation | 95% cluster BCa | sibling cos | "
                 "non-sib cos | contrast | pairs reordered | recall@k | degenerate |")
_SURFACE_RULE = "|---|---|---|---|---|---|---|---|---|---|"


def _interval_parts(value: Any) -> tuple[float, float, float]:
    if isinstance(value, dict):
        return float(value["mean"]), float(value["lower"]), float(value["upper"])
    return float(value.mean), float(value.lower), float(value.upper)


def interval_cell(value: Any) -> str:
    mean, lower, upper = _interval_parts(value)
    if mean != mean:  # NaN
        return "n/a"
    if lower != lower or upper != upper:
        return f"{mean:.4f} [n/a, n/a]"
    return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"


def _surface_rows(cells: Sequence[dict[str, Any]]) -> list[str]:
    out = []
    for cell in cells:
        out.append(
            f"| {float(cell['gamma']):g} | {float(cell['alpha']):g} | "
            f"{float(cell['metrics']['disambiguation']):.4f} | "
            f"{interval_cell(cell['bootstrap'])} | {float(cell['sibling_cosine']):.4f} | "
            f"{float(cell['non_sibling_cosine']):.4f} | {float(cell['contrast']):.4f} | "
            f"{int(cell['sibling_pairs_reordered'])}/{int(cell['sibling_pairs_compared'])} | "
            f"{float(cell['metrics']['recall_at_k']):.4f} | "
            f"{int(cell['degenerate_documents'])} |")
    return out


def select(payload: dict[str, Any], **match: Any) -> list[dict[str, Any]]:
    def keep(cell: dict[str, Any]) -> bool:
        return all(cell.get(key) == value for key, value in match.items())

    return [c for c in payload["cells"] if keep(c)]


def surface(payload: dict[str, Any], title: str, note: str, **match: Any) -> list[str]:
    cells = sorted(select(payload, **match),
                   key=lambda c: (float(c["gamma"]), float(c["alpha"])))
    if not cells:
        return []
    return ["", f"### {title}", "", note, "", _SURFACE_HEAD, _SURFACE_RULE, *_surface_rows(cells)]


def baseline_table(payload: dict[str, Any]) -> list[str]:
    lines = ["", "## Baseline arms (all five required)", "",
             "| arm | query set | disambiguation | 95% cluster BCa | on the 180 subsample | "
             "recall@k | LLM calls | cost |", "|---|---|---|---|---|---|---|---|"]
    for name in ("plain_vector", "bm25", "llm_pick", "full_context"):
        arm = payload["baselines"].get(name)
        if arm is None:
            lines.append(f"| {name} | n/a | BLOCKED | "
                         f"{payload['blocked_arms'].get(name, 'not run')} | n/a | n/a | n/a | n/a |")
            continue
        stats = arm.get("stats", {}) or {}
        calls = int(stats.get("api_calls", 0) or 0)
        cost = float(stats.get("cost_usd", 0.0) or 0.0)
        lines.append(
            f"| {name} | {arm.get('query_set', 'full')} | "
            f"{float(arm['metrics']['disambiguation']):.4f} | {interval_cell(arm['bootstrap'])} | "
            f"{interval_cell(arm['subsample_bootstrap'])} | "
            f"{float(arm['metrics']['recall_at_k']):.4f} | {calls} | ${cost:.4f} |")
    best = _best_cell(payload)
    if best is not None:
        lines.append(
            f"| **ncp (best cell: {best['arm']})** | full | "
            f"{float(best['metrics']['disambiguation']):.4f} | {interval_cell(best['bootstrap'])} | "
            f"{interval_cell(best['subsample_bootstrap'])} | "
            f"{float(best['metrics']['recall_at_k']):.4f} | 0 | $0.0000 |")
    return lines


def _best_cell(payload: dict[str, Any]) -> dict[str, Any] | None:
    cells = [c for c in payload["cells"] if c["weights"] == "local"]
    return max(cells, key=lambda c: float(c["metrics"]["disambiguation"])) if cells else None


def incumbent_table(payload: dict[str, Any]) -> list[str]:
    lines = ["", "## The named incumbent: weighted pooling on GLOBAL statistics",
             "", "Arora et al (SIF / uSIF) already weight tokens before pooling. The claimed "
             "delta of NCP is that the weights are conditioned on the LOCAL neighbourhood. "
             "`global` is the one-variable ablation (same weight function, corpus-wide df); "
             "`sif` is the published `a / (a + p(t))` form, with and without common-component "
             "removal.", "",
             "| arm | weights | gamma | a | remove PC | alpha | query side | disambiguation | "
             "95% cluster BCa | sibling cos |", "|---|---|---|---|---|---|---|---|---|---|"]
    cells = [c for c in payload["cells"] if c["weights"] in ("global", "sif")]
    for cell in sorted(cells, key=lambda c: (str(c["weights"]), float(c["sif_a"]),
                                             bool(c["remove_pc"]), float(c["gamma"]),
                                             float(c["alpha"]), str(c["query_side"]))):
        lines.append(
            f"| {cell['arm']} | {cell['weights']} | {float(cell['gamma']):g} | "
            f"{float(cell['sif_a']):g} | {bool(cell['remove_pc'])} | {float(cell['alpha']):g} | "
            f"{cell['query_side']} | {float(cell['metrics']['disambiguation']):.4f} | "
            f"{interval_cell(cell['bootstrap'])} | {float(cell['sibling_cosine']):.4f} |")
    return lines


def reordering_table(payload: dict[str, Any]) -> list[str]:
    verdict = payload["verdict"]
    lines = ["", "## The C-A test: does varying gamma actually REORDER sibling pairs?", "",
             "> A1 (C-A) was a symmetric transform on finished vectors: a colliding pair shared "
             "a normalizer, so the sign of the score difference was independent of the knob. It "
             "measured 0.8056 at every strength from 0 to 0.8, identical to four decimals. NCP "
             "acts before pooling with a different weight vector per episode, so the premise "
             "fails; the point of this table is that the CONCLUSION is measured, not assumed.",
             "",
             f"- sibling pairs reordered, maximum over all cells: "
             f"**{int(verdict['max_sibling_pairs_reordered'])}**",
             f"- pair reordering observed: **{bool(verdict['pair_reordering_observed'])}**",
             f"- disambiguation flat in gamma (spread <= 1e-4 at every alpha): "
             f"**{bool(verdict['flat_in_gamma'])}**",
             f"- C-A signature (flat AND zero reorderings): **{bool(verdict['ca_signature'])}**",
             "", "Spread of disambiguation across gamma, at the primary k and query side:", "",
             "| alpha | max - min over gamma |", "|---|---|"]
    for key, value in sorted(verdict["gamma_spread_by_alpha"].items()):
        lines.append(f"| {key} | {float(value):.6f} |")
    dropped = verdict.get("cells_with_cosine_drop_and_no_gain") or []
    lines += ["", f"Cells at or below C-A's sibling cosine (0.704) that did NOT beat "
                  f"plain_vector: **{len(dropped)}**. A cosine drop with a flat or negative "
                  f"accuracy delta is a failure, not a partial win."]
    return lines


def points_cell(value: Any) -> str:
    mean, lower, upper = _interval_parts(value)
    if mean != mean:
        return "n/a"
    return f"{mean * 100:+.2f} [{lower * 100:+.2f}, {upper * 100:+.2f}]"


def delta_table(payload: dict[str, Any]) -> list[str]:
    """Paired cluster-BCa deltas. Two overlapping marginals settle nothing."""
    deltas = payload.get("paired_deltas") or {}
    if not deltas:
        return []
    lines = ["", "## Paired deltas (cluster-BCa, same resampled clusters on both arms)", "",
             "The claim NCP has to earn is not that weighted pooling helps, which Arora et al "
             "published in 2017. It is that conditioning the weights on the LOCAL neighbourhood "
             "beats conditioning them on global corpus statistics. Marginal intervals cannot "
             "settle that; these are paired.", "",
             "| comparison | delta (points) | interval excludes zero |", "|---|---|---|"]
    for key in ("best_local_minus_plain_vector", "best_local_minus_bm25",
                "best_local_minus_best_global", "best_local_minus_best_sif",
                "best_local_minus_llm_pick_on_subsample"):
        if key not in deltas:
            continue
        entry = deltas[key]
        _, lower, upper = _interval_parts(entry)
        conclusive = (lower > 0.0) or (upper < 0.0)
        label = key.replace("best_local_minus_", "best NCP cell minus ").replace("_", " ")
        lines.append(f"| {label} | {points_cell(entry)} | {conclusive} |")
    matched = deltas.get("local_minus_global_matched") or []
    if matched:
        total = deltas.get("matched_cells_compared", len(matched))
        lines += ["", f"**The strict one-variable ablation.** Same gamma, same alpha, same query "
                      f"side, k = {payload['preregistration']['primary_k']}: only WHERE df is "
                      f"measured changes. Both directions are counted, because reporting only "
                      f"the cells that favour the local rule would be the exact failure this "
                      f"project exists to avoid.",
                  "",
                  f"- local beats global conclusively (paired interval excluding zero): "
                  f"**{deltas.get('matched_cells_where_local_beats_global_conclusively', 0)} "
                  f"of {total}**",
                  f"- global beats local conclusively: "
                  f"**{deltas.get('matched_cells_where_global_beats_local_conclusively', 0)} "
                  f"of {total}**",
                  f"- positive point estimate, conclusive or not: "
                  f"**{deltas.get('matched_cells_with_positive_point_estimate', 0)} of {total}**",
                  f"- mean delta across every matched cell: "
                  f"**{float(deltas.get('mean_matched_delta_points') or float('nan')):+.2f} "
                  f"points**",
                  "",
                  "| gamma | alpha | query side | local | global | paired delta (points) |",
                  "|---|---|---|---|---|---|"]
        for row in matched:
            lines.append(
                f"| {float(row['gamma']):g} | {float(row['alpha']):g} | {row['query_side']} | "
                f"{float(row['local']):.4f} | {float(row['global']):.4f} | {points_cell(row['delta'])} |")
    return lines



def cost_table(payload: dict[str, Any]) -> list[str]:
    lines = ["", "## Measured cost", "",
             "| corpus | episodes | token encode (s) | index (s) | query (ms/query) | "
             "extra bytes/episode | LLM calls | cost |", "|---|---|---|---|---|---|---|---|"]
    for name, probe in sorted((payload.get("cost") or {}).items()):
        lines.append(
            f"| {name} | {probe['n_episodes']} | {probe['token_encode_seconds']} | "
            f"{probe['index_seconds']} | {probe['query_ms_per_query']} | "
            f"{probe['extra_bytes_per_episode']} | {probe['llm_calls']} | "
            f"${probe['cost_usd']:.4f} |")
    accounting = payload["llm_accounting"]
    lines += ["", f"- LLM calls made by this whole run: **{accounting['api_calls']}**, "
                  f"spend **${float(accounting['cost_usd']):.4f}** "
                  f"(backend forced to `{accounting['backend']}`, so a cold call was "
                  f"unreachable)",
              f"- per-query LLM calls on the NCP path: "
              f"**{accounting['per_query_llm_calls_on_the_ncp_path']}**"]
    return lines


