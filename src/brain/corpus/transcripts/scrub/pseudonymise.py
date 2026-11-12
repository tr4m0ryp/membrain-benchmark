"""Stable pseudonymisation. Reversible only with the local map.

A pseudonym is `HMAC-SHA256(salt, category || "\\x00" || normalized_value)`
truncated to 8 base32 characters. Three properties follow, and all three are
required:

- **Stable.** The same path in two sessions gets the same pseudonym, so a
  labeller can still tell that two episodes touch the same file. Destroying that
  would destroy the referent structure the corpus exists to capture.
- **Not invertible from the artifact.** Without the salt, recovering the value
  means guessing it and testing, and the salt never leaves `data/transcripts/`.
- **Reversible with the map.** `pseudonym-map.json` holds pseudonym to original.
  It is written 0600 into a gitignored directory and is the ONLY copy.

The salt is generated once and reused, because regenerating it would renumber
every pseudonym and silently invalidate every label a human had already applied.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
from base64 import b32encode
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from brain.corpus.transcripts.scrub.patterns import DETECTORS

__all__ = ["SALT_FILE", "MAP_FILE", "Pseudonymiser", "load_or_create_salt"]

SALT_FILE = ".salt"
MAP_FILE = "pseudonym-map.json"

#: category -> the shape its pseudonym is rendered in. The rendering keeps the
#: text parseable (a path still looks like a path) so downstream tokenisation
#: statistics are not distorted by the substitution itself.
_TEMPLATES = {
    "email": "user_{h}@example.invalid",
    "home_path": "/home/user_{h}",
    "private_ipv4": "10.0.0.{d}",
    "internal_hostname": "host-{h}.invalid",
    "github_repo_url": "github.com/org-{h}/repo-{h2}",
    "phone_e164": "+10000000000",
}

_PSEUDONYMISED = tuple(d for d in DETECTORS if d.category in _TEMPLATES)

_EXTRA_LITERAL_CATEGORY = "person_name"


def load_or_create_salt(directory: Path) -> bytes:
    """Read the salt, creating it 0600 on first use. Never regenerate it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SALT_FILE
    if path.is_file():
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    salt = secrets.token_bytes(32)
    path.write_text(salt.hex(), encoding="utf-8")
    os.chmod(path, 0o600)
    return salt


@dataclass
class Pseudonymiser:
    """Applies the substitution and accumulates the inverse map.

    `literals` are values a regex cannot find: author names, an organisation
    name, a machine name. They are supplied by the caller (from `git log` and
    from the store's own metadata) and matched case-insensitively on word
    boundaries.
    """

    salt: bytes
    literals: dict[str, str] = field(default_factory=dict)
    mapping: dict[str, str] = field(default_factory=dict)
    #: category -> how many DISTINCT identifiers of that category were replaced.
    #: Reportable: it is a count, and it never carries a value.
    categories: dict[str, int] = field(default_factory=dict)
    _literal_re: re.Pattern[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._compile_literals()

    def _compile_literals(self) -> None:
        names = sorted((n for n in self.literals if n.strip()), key=len, reverse=True)
        if not names:
            self._literal_re = None
            return
        joined = "|".join(re.escape(n) for n in names)
        self._literal_re = re.compile(rf"(?<![\w.\-])(?:{joined})(?![\w.\-])", re.I)

    def _digest(self, category: str, value: str) -> bytes:
        return hmac.new(
            self.salt, f"{category}\x00{value.strip().lower()}".encode("utf-8"), sha256
        ).digest()

    def token(self, category: str, value: str, *, width: int = 8) -> str:
        raw = b32encode(self._digest(category, value)).decode("ascii")
        return raw.rstrip("=").lower()[:width]

    def render(self, category: str, value: str) -> str:
        """Pseudonymise one value of a known category, recording the inverse."""
        template = _TEMPLATES.get(category, "{category}_{h}")
        digest = self._digest(category, value)
        rendered = template.format(
            h=self.token(category, value),
            h2=self.token(category + ":2", value),
            d=str(digest[0] % 254 + 1),
            category=category,
        )
        if rendered not in self.mapping:
            self.mapping[rendered] = value
            self.categories[category] = self.categories.get(category, 0) + 1
        return rendered

    def apply(self, text: str) -> str:
        """Substitute every identifier this instance knows how to find."""
        out = text
        for detector in _PSEUDONYMISED:
            out = detector.pattern.sub(
                lambda m, c=detector.category: self.render(c, m.group(0)), out
            )
        if self._literal_re is not None:
            out = self._literal_re.sub(
                lambda m: self.render(
                    _EXTRA_LITERAL_CATEGORY, self.literals.get(m.group(0), m.group(0))
                ),
                out,
            )
        return out

    def apply_all(self, texts: Iterable[str]) -> list[str]:
        return [self.apply(text) for text in texts]

    def write_map(self, directory: Path) -> Path:
        """Write the inverse map 0600. The only route back to the originals."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MAP_FILE
        payload = {
            "note": (
                "LOCAL ONLY. Maps pseudonym -> original identifier. This file is the "
                "sole route back to the real transcripts and is gitignored. Do not "
                "copy it out of data/transcripts/."
            ),
            "n_entries": len(self.mapping),
            "map": dict(sorted(self.mapping.items())),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        os.chmod(path, 0o600)
        return path
