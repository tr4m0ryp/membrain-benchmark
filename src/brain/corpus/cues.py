"""Cue lookups and the verbatim-cue rendering used to calibrate difficulty."""

from __future__ import annotations

from brain.corpus.content import DIAGNOSES, TOPICS, TRIGGERS, render_query
from brain.corpus.types import Corpus

__all__ = ["verbatim_cue_query_texts"]

_TOPIC_BY_ID = {topic.topic_id: topic for topic in TOPICS}
_TRIGGER_BY_ID = {value.value_id: value for value in TRIGGERS}
_DIAGNOSIS_BY_ID = {value.value_id: value for value in DIAGNOSES}
# The outcome cue reads the episode's own rendered text: the numbers were drawn
# per episode, so the bank entry alone would not reproduce the verbatim wording.


def verbatim_cue_query_texts(corpus: Corpus) -> list[str]:
    """The same queries with each cue restored to the episode's own wording.

    Diagnostic only, returned in corpus order. Scoring these measures the ceiling
    a retriever would reach if the cue were NOT paraphrased, which separates two
    very different reasons a query can be missed: an unanswerable question versus
    interference between siblings. A corpus whose verbatim ceiling is low is not
    a valid instrument, because then even a perfect cue fails to identify the
    episode and the label is effectively unreachable.
    """
    by_id = corpus.episode_by_id()
    texts: list[str] = []
    for query in corpus.queries:
        episode = by_id[query.correct_trace_id]
        topic = _TOPIC_BY_ID[episode.topic["topic_id"]]
        if query.cue_axis == "trigger":
            cue = _TRIGGER_BY_ID[query.cue_value_id].episode
        elif query.cue_axis == "outcome":
            cue = episode.facts["outcome_text"]
        else:
            cue = _DIAGNOSIS_BY_ID[query.cue_value_id].cause
        texts.append(render_query(topic=topic, cue_axis=query.cue_axis, cue=cue))
    return texts
