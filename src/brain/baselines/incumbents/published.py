"""Published numbers for the incumbents. NOT MEASURED BY US.

Every entry here was read out of a paper and is carried so the comparison table
can name what each system claims for itself. None of it was produced on this
machine, none of it is on this project's corpus, and none of it is comparable to
the measured rows: the benchmarks differ, the answer models differ, and in two
cases the reported quantity is not accuracy at all.

Each row therefore carries `measured_by_us: False` and a source down to the
table. The `caveat` field exists because several of these numbers are widely
miscited, and a comparison table that repeats a miscitation is worse than one
that omits the row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["PublishedNumber", "PUBLISHED", "table"]


@dataclass(frozen=True, slots=True)
class PublishedNumber:
    system: str
    benchmark: str
    metric: str
    value: str
    model: str
    source: str
    caveat: str = ""
    measured_by_us: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PUBLISHED: tuple[PublishedNumber, ...] = (
    PublishedNumber(
        system="Mem0",
        benchmark="LOCOMO",
        metric="Overall LLM-as-a-judge score J",
        value="66.88 +/- 0.15%",
        model="GPT-4o-mini as answer generator; judge model not named in the paper",
        source="arXiv:2504.19413v1, Table 2",
        caveat=(
            "Full-context in the same table scores 72.90 +/- 0.19%, i.e. the naive baseline "
            "beats Mem0 on the paper's own numbers. The mem0ai README advertises LoCoMo 92.5 "
            "for an April 2026 algorithm; that is the MANAGED platform, not the OSS package, "
            "and is not in the paper."
        ),
    ),
    PublishedNumber(
        system="Mem0-graph",
        benchmark="LOCOMO",
        metric="Overall LLM-as-a-judge score J",
        value="68.44 +/- 0.17%",
        model="GPT-4o-mini, Neo4j graph store",
        source="arXiv:2504.19413v1, Table 2",
        caveat=(
            "Not reproducible from current PyPI: graph memory was removed from mem0ai in 2.0.0 "
            "(2026-04-16); 1.0.11 is the last version that contains it."
        ),
    ),
    PublishedNumber(
        system="Mem0",
        benchmark="LOCOMO",
        metric="p95 total latency",
        value="1.440 s (vs 17.117 s full-context)",
        model="GPT-4o-mini",
        source="arXiv:2504.19413v1, Table 2",
    ),
    PublishedNumber(
        system="Zep",
        benchmark="Deep Memory Retrieval (DMR)",
        metric="accuracy",
        value="94.8% (gpt-4-turbo), 98.2% (gpt-4o-mini)",
        model="gpt-4-turbo / gpt-4o-mini",
        source="arXiv:2501.13956v1, Table 1",
        caveat=(
            "Full-conversation in the same table scores 94.4% and 98.0%, so the margin over "
            "stuffing the whole conversation is 0.4 and 0.2 points."
        ),
    ),
    PublishedNumber(
        system="Zep",
        benchmark="LongMemEval_s",
        metric="accuracy and latency",
        value="63.8% at 3.20 s (gpt-4o-mini), 71.2% at 2.58 s (gpt-4o)",
        model="gpt-4o-mini / gpt-4o",
        source="arXiv:2501.13956v1, Table 2",
        caveat=(
            "Full-context scores 55.4% at 31.3 s and 60.2% at 28.9 s on the same rows. The "
            "single-session-user row of Table 3 is byte-identical across the two models, which "
            "is a defect in the published table."
        ),
    ),
    PublishedNumber(
        system="Zep",
        benchmark="LOCOMO",
        metric="Overall J, as measured by the Mem0 authors",
        value="65.99 +/- 0.16%",
        model="GPT-4o-mini",
        source="arXiv:2504.19413v1, Table 2",
        caveat="A competitor's re-run of Zep, not Zep's own reported number.",
    ),
    PublishedNumber(
        system="TOKI",
        benchmark="LoCoMo answerable-factual pool, n=1444",
        metric="paired memory-layer ablation delta",
        value="+0.492, 95% CI [+0.465, +0.519], McNemar p < 1e-4",
        model="pinned synthesiser and judge; the structural results need no key",
        source="arXiv:2606.06240v1, section 4.7",
        caveat=(
            "TOKI makes NO utility-superiority claim. Its headline figures are mechanism "
            "deltas on constructed slices, not leaderboard scores, and its cross-system "
            "transparency ledger reports mem0 v3 -0.04, Graphiti -0.08, Zep +0.02, all "
            "confidence intervals covering zero."
        ),
    ),
    PublishedNumber(
        system="TOKI",
        benchmark="verdict matrix over eight systems",
        metric="anomalies admitted (replay inconsistency, belief-drift skew, audit erasure)",
        value="every baseline admits at least one; TOKI excludes all three",
        model="n/a, structural",
        source="arXiv:2606.06240v1, Table 4",
        caveat=(
            "This is prior art for the audit half of our claim: it compares the verdict a "
            "system's design implies against the verdict observed from its running code, over "
            "mem0 v2, mem0 v3, Graphiti, Letta, Zep, MIRIX and WorldDB."
        ),
    ),
    PublishedNumber(
        system="Nous",
        benchmark="LoCoMo, 10 conversations, 1540 questions",
        metric="strict token-F1 by category (SH / MH / T / OD)",
        value="44.66 / 23.48 / 51.10 / 10.54, macro 32.45",
        model="gemini-2.5-flash for extraction and answering, gpt-4o-mini as judge",
        source="arXiv:2606.22030v2, Tables 1 and 2",
        caveat=(
            "v2 carries an erratum: v1's headline 'F1' column was an LLM judge, not F1, and "
            "those figures are retracted. The baseline rows in its Table 1 transpose the "
            "multi-hop and temporal columns relative to Mem0's Table 1."
        ),
    ),
    PublishedNumber(
        system="Nous",
        benchmark="component ablation, 385 questions",
        metric="LLM-judge score with and without the Bayesian update",
        value="69.62 full engine, 72.46 with the Bayesian update DISABLED",
        model="as above",
        source="arXiv:2606.22030v2, Table 4",
        caveat=(
            "The paper's own central negative result: removing the belief update raises the "
            "score. Worth citing next to any claim that probabilistic belief maintenance is "
            "what makes such a system work."
        ),
    ),
)


def table() -> dict[str, Any]:
    return {
        "label": "PUBLISHED, NOT RUN BY US",
        "warning": (
            "These numbers were read from papers on 2026-08-01. They are on other corpora with "
            "other answer models and are not comparable to any measured row in this file."
        ),
        "rows": [row.to_dict() for row in PUBLISHED],
    }
