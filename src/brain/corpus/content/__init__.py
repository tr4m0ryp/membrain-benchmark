"""Content banks for the synthetic corpus: topics, discriminative axes, narrative."""

from brain.corpus.content.axes import DIAGNOSES, OUTCOMES, TRIGGERS, AxisValue, Diagnosis
from brain.corpus.content.narrative import (
    CUE_AXES,
    N_SHARED_MAX,
    SHARED_SENTENCES,
    VARIANTS,
    render_answer,
    render_episode,
    render_query,
)
from brain.corpus.content.topics import TOPICS, Topic

__all__ = [
    "AxisValue",
    "Diagnosis",
    "Topic",
    "TRIGGERS",
    "DIAGNOSES",
    "OUTCOMES",
    "TOPICS",
    "CUE_AXES",
    "N_SHARED_MAX",
    "SHARED_SENTENCES",
    "VARIANTS",
    "render_answer",
    "render_episode",
    "render_query",
]
