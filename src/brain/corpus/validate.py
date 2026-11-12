"""Ground-truth invariants. A failure here is a generator bug, never data noise.

The gate experiment's headline metric is disambiguation accuracy against sibling
distractors, so every one of these properties has to hold exactly: one correct
trace per query, siblings that really are different events, and a cue that picks
out exactly one episode corpus-wide.
"""

from __future__ import annotations

from brain.corpus.types import Corpus, Episode

__all__ = ["CorpusValidationError", "MAX_EPISODE_WORDS", "validate_corpus"]

#: Hard cap on episode length, in whitespace words. The gate embedder
#: (all-MiniLM-L6-v2) truncates at 256 wordpieces and drops the tail silently.
#: Measured on this corpus: 1.79 wordpieces per word, so 130 words is about 233
#: wordpieces, roughly 10 percent under the window. An episode whose tail was cut
#: would lose part of what distinguishes it from its siblings, which is exactly
#: the information the gate is testing. Re-verified against the real tokenizer at
#: freeze time (`freeze._assert_fits_encoder`).
MAX_EPISODE_WORDS = 130

_AXIS_OF_CUE = {"trigger": "trigger", "outcome": "outcome", "cause": "diagnosis"}


class CorpusValidationError(RuntimeError):
    """Raised when data would give ambiguous or unusable ground truth."""


def validate_corpus(corpus: Corpus) -> None:
    """Assert the ground truth is unambiguous."""
    by_id = corpus.episode_by_id()
    if len(by_id) != len(corpus.episodes):
        raise CorpusValidationError("duplicate trace_id")
    if len({q.query_id for q in corpus.queries}) != len(corpus.queries):
        raise CorpusValidationError("duplicate query_id")
    if len({e.text for e in corpus.episodes}) != len(corpus.episodes):
        raise CorpusValidationError("two episodes share identical text")
    _validate_lengths(corpus)
    clusters = corpus.clusters()
    for cluster_id, members in clusters.items():
        if len(members) < 2:
            raise CorpusValidationError(f"cluster {cluster_id} has no distractors")
        for axis in ("trigger", "diagnosis", "outcome"):
            values = [m.facts[axis] for m in members]
            if len(set(values)) != len(values):
                raise CorpusValidationError(f"cluster {cluster_id} repeats axis {axis}")
        if len({m.topic["path"] for m in members}) != 1:
            raise CorpusValidationError(f"cluster {cluster_id} spans more than one file")
    paths = [members[0].topic["path"] for members in clusters.values()]
    if len(set(paths)) != len(paths):
        raise CorpusValidationError("two clusters share a file path")
    _validate_queries(corpus, by_id, clusters)


def _validate_lengths(corpus: Corpus) -> None:
    longest_words, longest_id = max(
        (len(episode.text.split()), episode.trace_id) for episode in corpus.episodes
    )
    if longest_words > MAX_EPISODE_WORDS:
        raise CorpusValidationError(
            f"{longest_id} is {longest_words} words, over the {MAX_EPISODE_WORDS} word cap; "
            "the embedder would truncate it and drop the distinguishing fact"
        )


def _validate_queries(
    corpus: Corpus, by_id: dict[str, Episode], clusters: dict[str, list[Episode]]
) -> None:
    for query in corpus.queries:
        target = by_id.get(query.correct_trace_id)
        if target is None:
            raise CorpusValidationError(f"{query.query_id} points at a missing trace")
        if not query.distractor_ids:
            raise CorpusValidationError(f"{query.query_id} has no distractor")
        if query.correct_trace_id in query.distractor_ids:
            raise CorpusValidationError(f"{query.query_id} lists the answer as a distractor")
        siblings = sorted(
            e.trace_id for e in clusters[query.cluster_id] if e.trace_id != query.correct_trace_id
        )
        if query.distractor_ids != siblings:
            raise CorpusValidationError(f"{query.query_id} distractors are not the siblings")
        # The cue must select exactly one episode corpus-wide: the file pins the
        # cluster and the axis value pins the member.
        axis = _AXIS_OF_CUE[query.cue_axis]
        matches = [
            e.trace_id
            for e in corpus.episodes
            if e.topic["path"] == target.topic["path"] and e.facts[axis] == query.cue_value_id
        ]
        if matches != [query.correct_trace_id]:
            raise CorpusValidationError(f"{query.query_id} cue matches {matches}")
