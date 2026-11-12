"""Read a Claude Code session file into a `SessionRecord`.

Handles both the plain `sessions/<id>.jsonl` layout and the gzipped
`.sessions-full/<id>.jsonl.gz` mirror, which were verified to hold byte-identical
line content, so either root produces the same inventory.

Only the USER prompt and the assistant's TEXT blocks are kept. Everything else is
dropped on purpose:

- `attachment` lines carry the SessionStart hook payload, which is the user's
  global memory file injected verbatim into every session. It is identical
  across sessions, it is personal, and it is not a work event. Including it would
  manufacture a shared-boilerplate fraction near 1.0 and destroy the very
  statistic this pipeline exists to measure.
- `tool_result` lines carry file contents at full length, which is the largest
  single source of unreviewed private code.
- `system` lines are harness messages.

The genre classifier is not cosmetic. The store turned out to be one genre almost
end to end, and that fact is the headline finding of the inventory.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from brain.corpus.transcripts.types import SessionRecord

__all__ = ["GENRE_PREFIXES", "read_session", "iter_session_paths", "session_id_of"]

#: Prompt prefix -> genre. Checked longest-first so a more specific prefix wins.
GENRE_PREFIXES: dict[str, str] = {
    "Write ONE git commit subject line": "commit_subject",
    "You are the autonomous GLOBAL-MEMORY curator": "memory_curator",
    "[Request interrupted by user]": "interrupted",
}


def session_id_of(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def iter_session_paths(root: Path) -> Iterator[Path]:
    """Every session file directly under `root`, in sorted id order."""
    if not root.is_dir():
        return
    paths = sorted(
        (p for p in root.iterdir() if p.name.endswith((".jsonl", ".jsonl.gz"))),
        key=session_id_of,
    )
    yield from paths


def _open(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _is_tool_result(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )


def _classify(prompt: str) -> str:
    stripped = prompt.strip()
    for prefix in sorted(GENRE_PREFIXES, key=len, reverse=True):
        if stripped.startswith(prefix):
            return GENRE_PREFIXES[prefix]
    return "interactive" if stripped else "unknown"


def read_session(path: Path) -> SessionRecord:
    """Reduce one session file to the fields the pipeline uses."""
    prompts: list[str] = []
    answers: list[str] = []
    stamps: list[str] = []
    n_lines = n_user = n_tools = 0
    cwd = branch = ""
    with _open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1
            stamp = record.get("timestamp")
            if isinstance(stamp, str):
                stamps.append(stamp)
            cwd = str(record.get("cwd") or cwd)
            branch = str(record.get("gitBranch") or branch)
            kind = record.get("type")
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if kind == "user" and not _is_tool_result(content):
                text = _text_of(content)
                if text.strip():
                    n_user += 1
                    prompts.append(text)
            elif kind == "assistant" and isinstance(content, list):
                n_tools += sum(
                    1
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                )
                text = _text_of(content)
                if text.strip():
                    answers.append(text)
    prompt = "\n\n".join(prompts)
    return SessionRecord(
        session_id=session_id_of(path),
        genre=_classify(prompt),
        started_at=min(stamps) if stamps else "",
        ended_at=max(stamps) if stamps else "",
        n_lines=n_lines,
        n_bytes=path.stat().st_size,
        n_user_turns=n_user,
        n_tool_uses=n_tools,
        cwd=cwd,
        git_branch=branch,
        prompt=prompt,
        answer="\n\n".join(answers).strip(),
    )
