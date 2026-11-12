"""LoCoMo (Maharana et al., ACL 2024) mapped onto `brain.corpus.types`.

LoCoMo is ten very-long-term two-speaker conversations with turn-level evidence
annotations for every question. That annotation is what makes it usable here:
the crediting convention only has anything to say when a merged record absorbs
the queried event AND another event that some question distinguishes it from,
and only an evidence annotation says which turn is which event.

**The mapping, stated so a reader can disagree with it before seeing a number.**

* One EPISODE per dialogue turn, rendered exactly as the benchmark's own
  evaluation renders it (`task_eval/claude_utils.py`): `Speaker said, "text"`,
  plus ` and shared <caption>.` for an image turn. The turn is the unit the
  evidence annotation points at, so it is the unit an event-level metric can be
  scored on.
* One STORE per conversation. The ten conversations have different speakers and
  disjoint topics, so a cross-conversation merge would be an artifact of pooling
  rather than a decision any deployed system faces.
* CLUSTER is the session (`sample_id/session_k`). Sessions are LoCoMo's own
  temporal grouping and the granularity a memory system consolidates at.
* One QUERY per answerable annotated question. `correct_trace_id` is the
  EARLIEST evidence turn (session index, then turn index), which is deterministic
  and independent of anything measured.
* DISTRACTORS come from the annotation and never from similarity. The pool is
  every evidence turn of every OTHER included question in the same conversation,
  i.e. turns some annotator marked as the answer to a different question, which
  is exactly "a distinct event of the same family". Two scopes are carried and
  both are pre-registered: `session` restricts the pool to the session holding
  the queried turn, `conversation` uses the whole conversation.

Excluded, with reasons: category 5 (adversarial, carries `adversarial_answer`
and no answerable event), questions whose evidence list is empty, and questions
with an evidence id that does not resolve to a turn (9 of 2815 references).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from brain.config import DATA_DIR
from brain.corpus.types import Episode, Query

__all__ = [
    "DISTRACTOR_SCOPES",
    "EXTERNAL_DIR",
    "LOCOMO_COMMIT",
    "LOCOMO_JSON",
    "LOCOMO_JSON_SHA256",
    "LOCOMO_REPO_URL",
    "LocomoConversation",
    "ensure_locomo",
    "load_locomo",
    "locomo_provenance",
    "render_turn",
    "scoped_queries",
]

EXTERNAL_DIR: Path = DATA_DIR / "external"
LOCOMO_REPO_URL = "https://github.com/snap-research/locomo.git"
#: The commit the numbers in `results/convention-external/*` were produced from.
LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_JSON: Path = EXTERNAL_DIR / "locomo-repo" / "data" / "locomo10.json"
#: Asserted on load. A benchmark that drifted is not the benchmark we report on.
LOCOMO_JSON_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

#: Both are pre-registered and both are reported. `session` is primary.
DISTRACTOR_SCOPES: tuple[str, ...] = ("session", "conversation")

#: LoCoMo category 5 is adversarial: unanswerable by design, no `answer` key.
ADVERSARIAL_CATEGORY = 5
_SESSION = re.compile(r"session_(\d+)$")


@dataclass(frozen=True, slots=True)
class LocomoConversation:
    """One conversation: its turns as episodes, its questions as queries.

    `distractors` holds one map per scope, so a scope can be swapped without
    re-reading the benchmark or re-deriving anything.
    """

    sample_id: str
    episodes: list[Episode]
    queries: list[Query]
    distractors: dict[str, dict[str, tuple[str, ...]]]
    stats: dict[str, Any]

    def clusters(self) -> dict[str, list[Episode]]:
        out: dict[str, list[Episode]] = {}
        for episode in sorted(self.episodes, key=lambda e: e.trace_id):
            out.setdefault(episode.cluster_id, []).append(episode)
        return {cid: out[cid] for cid in sorted(out)}


def ensure_locomo(*, root: Path | str = EXTERNAL_DIR, timeout_s: int = 600) -> Path:
    """Return the path to `locomo10.json`, cloning the release repo if needed."""
    target = Path(root) / "locomo-repo"
    data = target / "data" / "locomo10.json"
    if not data.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", LOCOMO_REPO_URL, str(target)],
            check=True, capture_output=True, text=True, timeout=timeout_s,
        )
    if not data.is_file():
        raise FileNotFoundError(f"LoCoMo clone did not produce {data}")
    return data


def render_turn(turn: Mapping[str, Any]) -> str:
    """The benchmark's own rendering, so the episode text is not our invention."""
    text = f'{turn.get("speaker", "")} said, "{turn.get("text", "")}"'
    caption = turn.get("blip_caption")
    if caption:
        text += f" and shared {caption}."
    return text


def _sessions(conversation: Mapping[str, Any]) -> list[tuple[int, str]]:
    """(chronological index, key) for every session, in chronological order."""
    found = [(int(m.group(1)), key) for key in conversation if (m := _SESSION.fullmatch(key))]
    return sorted(found)


def _turn_records(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten to per-turn records carrying the ordering keys a metric needs."""
    conversation = sample["conversation"]
    sample_id = str(sample["sample_id"])
    out: list[dict[str, Any]] = []
    for order, (index, key) in enumerate(_sessions(conversation)):
        date = str(conversation.get(f"{key}_date_time", ""))
        for position, turn in enumerate(conversation[key]):
            out.append({
                "trace_id": f"{sample_id}/{turn['dia_id']}",
                "dia_id": str(turn["dia_id"]),
                "cluster_id": f"{sample_id}/{key}",
                "session_index": index,
                "session_order": order,
                "position": position,
                "speaker": str(turn.get("speaker", "")),
                "date": date,
                "text": render_turn(turn),
            })
    return out


def _dedupe(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Collapse byte-identical turns, keeping the earliest, and map the rest to it.

    `content_sha` is UNIQUE in the store, so identical texts would collapse into
    one row silently and shrink the corpus underneath every metric. Collapsing
    them here instead makes the count reportable (6 turns in 5882) and lets the
    evidence annotation follow the surviving id.
    """
    kept: list[dict[str, Any]] = []
    first_by_text: dict[str, str] = {}
    alias: dict[str, str] = {}
    for record in sorted(records, key=lambda r: (r["session_order"], r["position"])):
        text = str(record["text"])
        if text in first_by_text:
            alias[str(record["trace_id"])] = first_by_text[text]
            continue
        first_by_text[text] = str(record["trace_id"])
        kept.append(dict(record))
    return kept, alias


def _evidence(question: Mapping[str, Any], sample_id: str,
              by_id: Mapping[str, Mapping[str, Any]],
              alias: Mapping[str, str]) -> list[str] | None:
    """Resolved evidence trace ids in chronological order, or None if unusable."""
    raw = question.get("evidence") or []
    if not raw or question.get("category") == ADVERSARIAL_CATEGORY:
        return None
    if question.get("answer") is None:
        return None
    resolved: list[str] = []
    for item in raw:
        trace_id = f"{sample_id}/{item}"
        trace_id = alias.get(trace_id, trace_id)
        if trace_id not in by_id:
            return None  # an unresolved reference makes the whole item unusable
        if trace_id not in resolved:
            resolved.append(trace_id)
    order = {tid: (by_id[tid]["session_order"], by_id[tid]["position"]) for tid in resolved}
    return sorted(resolved, key=lambda t: order[t])


def _conversation(sample: Mapping[str, Any]) -> LocomoConversation:
    sample_id = str(sample["sample_id"])
    records, alias = _dedupe(_turn_records(sample))
    by_id = {str(r["trace_id"]): r for r in records}
    sizes: dict[str, int] = {}
    for record in records:
        sizes[str(record["cluster_id"])] = sizes.get(str(record["cluster_id"]), 0) + 1
    episodes = [
        Episode(
            trace_id=str(r["trace_id"]), cluster_id=str(r["cluster_id"]),
            cluster_size=sizes[str(r["cluster_id"])], date=str(r["date"]), text=str(r["text"]),
            topic={"sample_id": sample_id, "session": str(r["cluster_id"]).split("/")[-1]},
            facts={"speaker": str(r["speaker"]), "dia_id": str(r["dia_id"])},
        )
        for r in sorted(records, key=lambda r: str(r["trace_id"]))
    ]

    usable: list[tuple[str, Mapping[str, Any], list[str]]] = []
    dropped = {"adversarial": 0, "no_evidence": 0, "unresolved_evidence": 0}
    for index, question in enumerate(sample.get("qa") or []):
        evidence = _evidence(question, sample_id, by_id, alias)
        if evidence is None:
            if question.get("category") == ADVERSARIAL_CATEGORY:
                dropped["adversarial"] += 1
            elif not (question.get("evidence") or []):
                dropped["no_evidence"] += 1
            else:
                dropped["unresolved_evidence"] += 1
            continue
        usable.append((f"{sample_id}/q{index:04d}", question, evidence))

    pool = sorted({tid for _, _, evidence in usable for tid in evidence})
    queries: list[Query] = []
    distractors: dict[str, dict[str, tuple[str, ...]]] = {s: {} for s in DISTRACTOR_SCOPES}
    for query_id, question, evidence in usable:
        correct = evidence[0]
        own = set(evidence)
        session = str(by_id[correct]["cluster_id"])
        for scope in DISTRACTOR_SCOPES:
            distractors[scope][query_id] = tuple(
                tid for tid in pool
                if tid not in own
                and (scope == "conversation" or str(by_id[tid]["cluster_id"]) == session)
            )
        queries.append(Query(
            query_id=query_id, cluster_id=session, cluster_size=sizes[session],
            text=str(question["question"]), correct_trace_id=correct, distractor_ids=[],
            cue_axis=f"category:{question.get('category')}",
            cue_value_id=str(by_id[correct]["dia_id"]),
            answer_axes=["answer"], answer=str(question["answer"]),
        ))
    stats = {
        "n_turns_raw": sum(len(sample["conversation"][k]) for _, k in _sessions(sample["conversation"])),
        "n_episodes": len(episodes), "n_sessions": len(sizes), "n_collapsed_turns": len(alias),
        "n_qa_raw": len(sample.get("qa") or []), "n_queries": len(queries),
        "n_answer_events": len(pool), "dropped": dropped,
        "distractors_median": {
            scope: sorted(len(v) for v in distractors[scope].values())[len(queries) // 2] if queries else 0
            for scope in DISTRACTOR_SCOPES},
        "queries_with_distractors": {
            scope: sum(1 for v in distractors[scope].values() if v) for scope in DISTRACTOR_SCOPES},
    }
    return LocomoConversation(sample_id, episodes, queries, distractors, stats)


def load_locomo(path: Path | str | None = None, *, verify: bool = True) -> list[LocomoConversation]:
    """Every conversation, in `sample_id` order, hash-verified by default."""
    source = Path(path) if path is not None else ensure_locomo()
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if verify and digest != LOCOMO_JSON_SHA256:
        raise RuntimeError(f"locomo10.json sha256 {digest}, expected {LOCOMO_JSON_SHA256}")
    samples = json.loads(payload.decode("utf-8"))
    return sorted((_conversation(s) for s in samples), key=lambda c: c.sample_id)


def scoped_queries(conversation: LocomoConversation, scope: str) -> list[Query]:
    """The conversation's queries with `scope`'s distractor set attached."""
    if scope not in DISTRACTOR_SCOPES:
        raise ValueError(f"scope must be one of {DISTRACTOR_SCOPES}, got {scope!r}")
    table = conversation.distractors[scope]
    return [replace(q, distractor_ids=list(table[q.query_id])) for q in conversation.queries]


def locomo_provenance(conversations: Iterable[LocomoConversation]) -> dict[str, Any]:
    """The shape `brain.eval.results.corpus_provenance` fills for our own corpus."""
    items = list(conversations)
    return {
        "benchmark": "LoCoMo-10 (Maharana et al., ACL 2024)",
        "source": LOCOMO_REPO_URL, "commit": LOCOMO_COMMIT,
        "content_sha256": LOCOMO_JSON_SHA256, "file": str(LOCOMO_JSON),
        "n_conversations": len(items),
        "n_episodes": sum(len(c.episodes) for c in items),
        "n_queries": sum(len(c.queries) for c in items),
        "n_clusters": sum(c.stats["n_sessions"] for c in items),
        "per_conversation": {c.sample_id: dict(c.stats) for c in items},
    }
