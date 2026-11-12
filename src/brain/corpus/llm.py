"""Optional LLM rewrite pass over the template corpus (claude-haiku-4-5).

Why a REWRITE and not LLM authoring: ground truth must stay programmatic. The
template generator owns the facts, the labels, and the sibling sets; the model
only restates an episode in more natural prose. A rewrite that drops or invents
a discriminative fact is rejected and the template text is kept, so the corpus
cannot silently lose its ground truth to a sampling accident.

Reproducibility: temperature 0 and an on-disk response cache keyed by the exact
prompt. A cached corpus regenerates offline, byte for byte, from the cache that
is committed next to the manifest. Queries are never rewritten: their cues are
deliberately paraphrased away from the episode wording, and a model asked to
"make it natural" tends to restore the episode's vocabulary, which would hand
the answer to a lexical baseline.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from brain.corpus.types import Corpus, Episode

__all__ = ["LlmConfig", "LlmUnavailable", "rewrite_corpus"]

DEFAULT_MODEL = "claude-haiku-4-5"
_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    "a an and the it its this that was were is are of to in on for from with only "
    "after before at by as had has have not no but so then than".split()
)


class LlmUnavailable(RuntimeError):
    """No anthropic client and no cached response for a prompt."""


class _Client(Protocol):  # pragma: no cover - structural typing only
    def complete(self, *, model: str, temperature: float, max_tokens: int, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class LlmConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 600
    min_fact_recall: float = 0.6


_INSTRUCTION = (
    "Rewrite this engineering session note as natural prose, 60 to 110 words.\n"
    "Hard rules: keep every fact, add none; keep the file path, every number and "
    "every identifier exactly as written; keep it one paragraph; no lists, no "
    "headings, no commentary. Return only the rewritten note.\n\nNOTE:\n"
)


def _prompt_for(episode: Episode) -> str:
    return _INSTRUCTION + episode.text


def _cache_key(prompt: str, config: LlmConfig) -> str:
    digest = hashlib.sha256()
    digest.update(config.model.encode("utf-8"))
    digest.update(f"|{config.temperature}|{config.max_tokens}|".encode("utf-8"))
    digest.update(prompt.encode("utf-8"))
    return digest.hexdigest()


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP and len(word) > 2}


def preserves_facts(original: str, rewritten: str, *, min_recall: float) -> bool:
    """Reject a rewrite that lost a number or most of the distinguishing wording."""
    if not rewritten.strip():
        return False
    if set(_NUMBER.findall(original)) - set(_NUMBER.findall(rewritten)):
        return False
    source = _content_words(original)
    if not source:
        return False
    recall = len(source & _content_words(rewritten)) / len(source)
    return recall >= min_recall


class _AnthropicClient:  # pragma: no cover - requires network and a key
    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LlmUnavailable("anthropic SDK not installed; pip install -e '.[llm]'") from exc
        self._client = anthropic.Anthropic()

    def complete(self, *, model: str, temperature: float, max_tokens: int, prompt: str) -> str:
        message = self._client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()


def _load_cached(path: Path) -> str | None:
    if not path.is_file():
        return None
    return str(json.loads(path.read_text(encoding="utf-8"))["text"])


def _store_cached(path: Path, *, prompt: str, text: str, config: LlmConfig) -> None:
    path.write_text(
        json.dumps(
            {
                "model": config.model,
                "temperature": config.temperature,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "text": text,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def rewrite_corpus(
    corpus: Corpus,
    *,
    cache_dir: Path | str,
    config: LlmConfig | None = None,
    client: _Client | None = None,
    allow_api: bool = True,
) -> tuple[Corpus, dict[str, Any]]:
    """Rewrite episode texts through the model, cache first. Returns corpus and stats."""
    config = config or LlmConfig()
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    stats = {"model": config.model, "cache_hits": 0, "api_calls": 0, "rejected": 0}
    rewritten: list[Episode] = []
    for episode in sorted(corpus.episodes, key=lambda e: e.trace_id):
        prompt = _prompt_for(episode)
        cache_path = root / f"{_cache_key(prompt, config)}.json"
        text = _load_cached(cache_path)
        if text is None:
            if client is None:
                if not allow_api:
                    raise LlmUnavailable(f"no cached rewrite for {episode.trace_id}")
                client = _AnthropicClient()
            text = client.complete(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                prompt=prompt,
            )
            _store_cached(cache_path, prompt=prompt, text=text, config=config)
            stats["api_calls"] += 1
        else:
            stats["cache_hits"] += 1
        if preserves_facts(episode.text, text, min_recall=config.min_fact_recall):
            rewritten.append(replace(episode, text=text.strip()))
        else:
            stats["rejected"] += 1
            rewritten.append(episode)
    return Corpus(episodes=rewritten, queries=list(corpus.queries), manifest={}), stats
