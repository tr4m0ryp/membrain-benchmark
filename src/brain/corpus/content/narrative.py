"""Sentence templates that turn a topic plus its axis values into text.

The shared/discriminative split is the whole difficulty knob. A rendered episode
is `n_shared` topic sentences (identical across siblings) interleaved with the
four discriminative sentences (unique within the cluster). Raising `n_shared`
raises the fraction of text every sibling has in common, which is exactly what
the `similarity` parameter buys.
"""

from __future__ import annotations

from .axes import AxisValue, Diagnosis
from .topics import Topic

__all__ = [
    "SHARED_SENTENCES",
    "VARIANTS",
    "N_SHARED_MAX",
    "CUE_AXES",
    "render_episode",
    "render_query",
    "render_answer",
]

SHARED_SENTENCES: dict[str, str] = {
    "s1": "Worked in {path} today, {subsystem} side.",
    "s2": "The problem is {symptom} in {component}.",
    "s3": "Repro: {repro}.",
    "s4": "Signal: {signal}.",
    "s5": "Stack is {language} and the covering check is {check}.",
    "s6": "Same corner of {path} as the earlier {symptom} work, so context carried over.",
}

_DISCRIMINATIVE_SENTENCES: dict[str, str] = {
    "g1": "This time it turned up {trigger}.",
    "g2": "Root cause: {cause}.",
    "g3": "Fix: {fix} ({artifact}).",
    "g4": "Result: {outcome}.",
}

#: Sentence orders. Siblings share a variant with probability `similarity`, so a
#: high-similarity cluster shares its narrative scaffold as well as its topic.
VARIANTS: tuple[tuple[str, ...], ...] = (
    ("s1", "s2", "s3", "s4", "g1", "g2", "g3", "g4", "s5", "s6"),
    ("s2", "s1", "g1", "s3", "g2", "s4", "g4", "g3", "s6", "s5"),
    ("s1", "g1", "s2", "s4", "g2", "g3", "s3", "s5", "g4", "s6"),
    ("s6", "s2", "s1", "g1", "s4", "g2", "g3", "s3", "g4", "s5"),
)

N_SHARED_MAX: int = len(SHARED_SENTENCES)

#: Query kinds: which axis is the cue, and which axes the answer must supply.
CUE_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("trigger", ("cause", "fix")),
    ("outcome", ("trigger", "cause")),
    ("cause", ("trigger", "outcome")),
)

_QUERY_TEMPLATES: dict[str, str] = {
    "trigger": (
        "In {path}, the {symptom} in {component} that showed up {cue}: "
        "what was the underlying cause, and which change fixed it?"
    ),
    "outcome": (
        "Which {symptom} in {path} ended with {cue}? "
        "What condition set it off, and what was the root cause?"
    ),
    "cause": (
        "In {path}, the {symptom} that traced back to {cue}: "
        "under what condition did it appear, and what did the fix measurably achieve?"
    ),
}


def _topic_fields(topic: Topic) -> dict[str, str]:
    return {
        "path": topic.path,
        "subsystem": topic.subsystem,
        "component": topic.component,
        "symptom": topic.symptom,
        "language": topic.language,
        "repro": topic.repro,
        "signal": topic.signal,
        "check": topic.check,
    }


def render_episode(
    *,
    topic: Topic,
    date: str,
    trigger: AxisValue,
    diagnosis: Diagnosis,
    outcome: str,
    artifact: str,
    variant: int,
    n_shared: int,
) -> str:
    """Render one episode. `n_shared` of the shared sentences survive, in order."""
    fields = _topic_fields(topic)
    fields.update(
        trigger=trigger.episode,
        cause=diagnosis.cause,
        fix=diagnosis.fix,
        outcome=outcome,
        artifact=artifact,
    )
    order = VARIANTS[variant % len(VARIANTS)]
    budget = max(0, min(n_shared, N_SHARED_MAX))
    sentences: list[str] = []
    for token in order:
        if token in SHARED_SENTENCES:
            if budget == 0:
                continue
            budget -= 1
            sentences.append(SHARED_SENTENCES[token].format(**fields))
        else:
            sentences.append(_DISCRIMINATIVE_SENTENCES[token].format(**fields))
    return f"[{date}] {topic.path}\n" + " ".join(sentences)


def render_query(*, topic: Topic, cue_axis: str, cue: str) -> str:
    fields = _topic_fields(topic)
    fields["cue"] = cue
    return _QUERY_TEMPLATES[cue_axis].format(**fields)


def render_answer(
    *,
    cue_axis: str,
    trigger: AxisValue,
    diagnosis: Diagnosis,
    outcome: str,
) -> str:
    """The ground-truth answer, composed from the axes the query asks for."""
    if cue_axis == "trigger":
        return f"cause: {diagnosis.cause}; fix: {diagnosis.fix}"
    if cue_axis == "outcome":
        return f"trigger: {trigger.episode}; cause: {diagnosis.cause}"
    if cue_axis == "cause":
        return f"trigger: {trigger.episode}; result: {outcome}"
    raise ValueError(f"unknown cue axis: {cue_axis!r}")
