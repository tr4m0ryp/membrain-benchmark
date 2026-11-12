"""Write the candidate artifacts. Everything here lands under `data/transcripts/`.

That directory is gitignored, and `assert_ignored` re-checks it at write time
rather than trusting that it still is. A pipeline that silently started writing
real diffs into a tracked path would be the worst possible failure of this task,
so the check runs on every invocation and raises rather than warns.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from brain.config import REPO_ROOT

__all__ = [
    "TRANSCRIPTS_DIR", "EPISODES_FILE", "PAIRS_FILE", "MANIFEST_FILE", "QUARANTINE_DIR",
    "NotIgnored", "assert_ignored", "write_jsonl", "write_json", "write_artifacts",
    "git_identity_literals",
]

TRANSCRIPTS_DIR: Path = REPO_ROOT / "data" / "transcripts"
EPISODES_FILE = "episodes.jsonl"
PAIRS_FILE = "pairs.jsonl"
MANIFEST_FILE = "manifest.json"
QUARANTINE_DIR = "quarantine"


class NotIgnored(RuntimeError):
    """The output directory is not gitignored. Refuse to write anything."""


def assert_ignored(directory: Path) -> None:
    """Refuse to write transcript-derived data into a tracked path."""
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".ignore-probe"
    probe.write_text("", encoding="utf-8")
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(probe)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # no git available: fail closed
        probe.unlink(missing_ok=True)
        raise NotIgnored(
            f"cannot verify that {directory} is gitignored; refusing to write derived transcripts"
        ) from None
    probe.unlink(missing_ok=True)
    if result.returncode != 0:
        raise NotIgnored(
            f"{directory} is NOT gitignored. Add it to .gitignore before running this pipeline; "
            "the artifacts carry real diffs and pseudonymised but real identifiers."
        )


def git_identity_literals() -> dict[str, str]:
    """Author names and emails from `git log`, for literal pseudonymisation.

    A regex finds an email; nothing finds a person's name. Reading the names the
    repository already records is the one reliable source, and it is read here
    rather than hardcoded so this works on any checkout.
    """
    literals: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--format=%an%x00%ae%x00%cn"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return literals
    for line in result.stdout.splitlines():
        for field in line.split("\x00"):
            value = field.strip()
            if len(value) >= 3 and not value.lower().startswith("noreply"):
                literals[value] = value
    return literals


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_artifacts(
    directory: Path,
    *,
    episodes: Sequence[Any],
    pairs: Sequence[Any],
    quarantined: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Write episodes, pairs, quarantine and manifest. Returns the paths written."""
    assert_ignored(directory)
    written = {
        "episodes": str(write_jsonl(directory / EPISODES_FILE, (e.to_dict() for e in episodes))),
        "pairs": str(write_jsonl(directory / PAIRS_FILE, (p.to_dict() for p in pairs))),
        "quarantine": str(
            write_jsonl(directory / QUARANTINE_DIR / EPISODES_FILE, quarantined)
        ),
        "manifest": str(write_json(directory / MANIFEST_FILE, manifest)),
    }
    return written
