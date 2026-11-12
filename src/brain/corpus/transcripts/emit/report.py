"""Render the run report. Counts, categories and statistics. Never content.

The report is written into `results/`, which IS tracked, so this module is the
last line of the safety argument: it may format only numbers that came out of the
scanner and the statistics modules, and it has no access to episode text. The
banner at the top is not decoration. A reader who takes one line from this file
has to take the line that says the corpus is unvalidated.
"""

from __future__ import annotations

from typing import Any

from brain.corpus.transcripts.protocol import PROTOCOL

__all__ = ["render_markdown"]

_BANNER = """\
> **UNVALIDATED. This corpus carries NO ground truth.**
> Every pair below is a CANDIDATE selected by similarity and stratified sampling.
> No pair has a class. Nothing here supports an external-validity claim, and no
> number computed from these labels exists, because there are no labels. The
> statistics section is the only part that is a measurement, and it is a
> measurement of the transcripts, not of any mechanism.
"""


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _provenance_caveat(inv: dict[str, Any]) -> str:
    """Say plainly what genre the store actually holds.

    Emitted from the data rather than written by hand, because the dominant-genre
    fraction is exactly the kind of fact that a report is tempted to leave out.
    """
    genres = inv["genres"]
    if not genres:
        return "No sessions found.\n"
    top, count = max(genres.items(), key=lambda kv: kv[1])
    share = count / max(inv["n_sessions"], 1)
    if share < 0.5:
        return f"Mixed store: no single genre exceeds half ({top} at {share:.1%}).\n"
    return (
        f"**Provenance caveat.** {share:.1%} of this store is a single genre, `{top}`, and "
        f"{inv['n_multi_turn']} of {inv['n_sessions']} sessions have more than one user turn. "
        "This is not a sample of interactive development work; it is one automated task "
        "repeated. Whatever a labelled corpus built from it supports, it does not support a "
        "claim about agent memory over general sessions, and any external-validity statement "
        "has to be scoped to this genre or wait for a store that contains more than one.\n"
    )


def _inventory_section(inv: dict[str, Any]) -> str:
    genres = _table(
        [(g, str(n)) for g, n in sorted(inv["genres"].items(), key=lambda kv: -kv[1])],
        ("genre", "sessions"),
    )
    return (
        f"Root `{inv['root']}`, **{inv['n_sessions']} session files**, "
        f"{inv['n_bytes'] / 1e6:.1f} MB, {inv['date_first'][:10]} to {inv['date_last'][:10]}.\n\n"
        f"{genres}\n\n"
        f"Episode-bearing sessions: **{inv['n_episode_bearing']}**. "
        f"Sessions with more than one user turn: {inv['n_multi_turn']}. "
        f"Sessions with any tool call: {inv['n_with_tool_use']} "
        f"({inv['total_tool_uses']} calls in total).\n"
    )


def _scan_section(scan: dict[str, Any]) -> str:
    findings = _table(
        [(f["category"], f["severity"], str(f["count"])) for f in scan["findings"]],
        ("category", "severity", "hits"),
    )
    return (
        f"Scanned **{scan['n_scanned']}** episode texts. "
        f"blocked **{scan['n_blocked']}**, flagged **{scan['n_flagged']}**, "
        f"identifier-only {scan['n_redacted']}, clean {scan['n_clean']}.\n\n"
        f"{findings}\n\n"
        "Counts only. No detected value is recorded in this file, in the result JSON, "
        "in the candidate corpus or in any log line: the scanner has no return path "
        "that carries matched text.\n"
    )


def _comparison_section(comparison: dict[str, dict[str, Any]]) -> str:
    blocks: list[str] = []
    for regime, rows in sorted(comparison.items()):
        table = _table(
            [
                (name, _fmt(row["real"]), _fmt(row["synthetic"]),
                 "" if row["delta"] is None else _fmt(row["delta"]))
                for name, row in rows.items()
            ],
            ("quantity", "real", "synthetic", "delta"),
        )
        blocks.append(f"### Regime `{regime}`\n\n{table}\n")
    return "\n".join(blocks)


def _audit_section(audit: dict[str, Any]) -> str:
    """Population size against sample size, per stratum.

    Printed because the sampling fraction is the difference between "we reviewed
    the pairs" and "we reviewed 68 of 1.7 million", and only one of those two
    readings supports anything.
    """
    if not audit:
        return "No mining audit: the encoder was unavailable, so no pair was selected.\n"
    blocks: list[str] = []
    for regime, entry in sorted(audit.items()):
        rows = [
            (stratum, str(entry["available"].get(stratum, 0)),
             str(entry["selected"].get(stratum, 0)))
            for stratum in sorted(entry["available"])
        ]
        thresholds = entry.get("thresholds", {})
        blocks.append(
            f"Regime `{regime}`, {entry['n_episodes']} episodes. Stratum cut points: "
            f"near_miss at cos >= {thresholds.get('near_miss', float('nan')):.4f}, "
            f"mid_band at cos >= {thresholds.get('mid_band', float('nan')):.4f} "
            "(quantiles of the observed cross-referent distribution).\n\n"
            + _table(rows, ("stratum", "pairs available", "pairs sampled"))
            + "\n"
        )
    return "\n".join(blocks)


def render_markdown(payload: dict[str, Any]) -> str:
    """The full report, from the result payload alone."""
    effort = payload["labelling_effort"]
    mining = payload["mining"]
    parts = [
        f"# Transcript-derived candidate corpus, {payload['run_id']}",
        "",
        _BANNER,
        "",
        "## 1. Inventory",
        "",
        _inventory_section(payload["inventory"]),
        _provenance_caveat(payload["inventory"]),
        "## 2. Extraction",
        "",
        f"Candidate episodes cut: **{payload['extraction']['n_extracted']}** "
        f"({payload['extraction']['n_events']} work events x "
        f"{len(payload['extraction']['by_regime'])} regimes). "
        f"Referents (distinct files): {payload['extraction']['n_entities']}. "
        f"Surviving the scan: **{payload['extraction']['n_survivors']}**.\n",
        "## 3. Secret scan",
        "",
        _scan_section(payload["scan"]["pre_pseudonymisation"]),
        f"Re-scan after pseudonymisation: "
        f"{payload['scan']['post_pseudonymisation']['n_blocked']} blocked, "
        f"{payload['scan']['post_pseudonymisation']['n_flagged']} flagged. "
        f"The pipeline refuses to write when either is non-zero.\n",
        "## 4. Pseudonymisation",
        "",
        f"{payload['pseudonymisation']['n_mapped']} distinct identifiers replaced across "
        f"{len(payload['pseudonymisation']['categories'])} categories. The inverse map and its "
        "salt are 0600 under `data/transcripts/`, gitignored, and are the only route back.\n",
        "## 5. Candidate pairs",
        "",
        f"**{mining['n_pairs']} pairs** for review, none labelled. "
        f"By stratum: {mining['by_stratum']}. By regime: {mining['by_regime']}. "
        f"Mean cosine {mining['cos']['mean']:.4f}, range "
        f"[{mining['cos']['min']:.4f}, {mining['cos']['max']:.4f}].\n\n"
        + _audit_section(payload.get("mining_audit", {})),
        "## 6. Real versus synthetic",
        "",
        _comparison_section(payload["comparison"]),
        "## 7. Labelling effort",
        "",
        f"**{effort['n_judgements_total']} judgements** "
        f"({effort['n_pairs_total']} pairs plus {effort['double_label_fraction']:.0%} "
        f"double-labelled for agreement), **{effort['hours_total']} hours**, "
        f"about {effort['days_at_5h']} working days at five focused hours. "
        f"Basis: {effort['basis']}.\n",
        "## 8. Protocol",
        "",
        PROTOCOL,
    ]
    return "\n".join(parts) + "\n"
