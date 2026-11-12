"""F21: memory-precision draws, with a LIVE entailment judge.

Every fresh corpus seed produces different episode text, so this is one of
the two statistics in the sweep that cannot fully replay from
`data/cache/entailment/`: an uncached (transcript, assertion) pair always
pays a live call through `EntailmentJudge` (the `claude` CLI transport, since
no `ANTHROPIC_API_KEY` is set here). Seed 0 reproduces the frozen corpus
byte-for-byte, so it is the one draw that replays entirely free and must
reproduce F21's 0.9497716894977168 exactly.

Same pipeline shape as `brain.eval.precision.__main__`: one episode per
interference cluster (first `N_TRANSCRIPTS`, sorted by cluster id) as the
transcript, one bounded extraction call per transcript, then every stored
assertion judged against its own transcript.
"""

from __future__ import annotations

from typing import Any, Sequence

from brain.corpus.generate import GeneratorConfig, generate_corpus
from brain.eval.precision.entailment import EntailmentJudge
from brain.eval.precision.metric import DEFAULT_SEEDS, precision_report

__all__ = ["N_TRANSCRIPTS", "ASSERTIONS_PER_TRANSCRIPT", "draw_transcripts", "draw_precision"]

N_TRANSCRIPTS = 30
ASSERTIONS_PER_TRANSCRIPT = 8


def draw_transcripts(seed: int, n: int = N_TRANSCRIPTS) -> dict[str, str]:
    """One episode per interference cluster, ids sorted, from a fresh draw."""
    corpus = generate_corpus(GeneratorConfig(seed=int(seed)))
    clusters = corpus.clusters()
    chosen: dict[str, str] = {}
    for cluster_id in sorted(clusters)[: max(int(n), 0)]:
        episode = clusters[cluster_id][0]
        chosen[episode.trace_id] = episode.text
    return chosen


def draw_precision(
    seed: int,
    judge: EntailmentJudge,
    *,
    n_transcripts: int = N_TRANSCRIPTS,
    bootstrap_seeds: Sequence[int] = DEFAULT_SEEDS,
    workers: int = 6,
) -> dict[str, Any]:
    """One draw: fresh transcripts, live extraction, live judging, the score."""
    transcripts = draw_transcripts(seed, n_transcripts)
    stored = judge.extract_many(
        list(transcripts.values()), max_assertions=ASSERTIONS_PER_TRANSCRIPT, workers=workers
    )
    items = {
        trace_id: (text, assertions)
        for (trace_id, text), assertions in zip(transcripts.items(), stored)
    }
    report = precision_report(items, judge=judge, seeds=bootstrap_seeds, workers=workers)
    return {
        "seed": int(seed),
        "score": report.score,
        "n_assertions": report.n_assertions,
        "n_transcripts": report.n_transcripts,
        "counts": dict(report.counts),
        "invented_rate": report.invented_rate,
        "fabrication_rate": report.fabrication_rate,
    }
