"""Detector catalogue. High recall first, precision second.

Ordering matters and is deliberate. `_BENIGN` shapes are consumed BEFORE the
credential detectors run, because a unified diff is dense in structural hex
(`index a5b1a90..896e838`, blob ids, uuids) and a 40-hex detector that does not
know about them reports every diff in the store as a secret. Consuming them is
not suppression: each benign shape is a documented, non-secret token format, it
is counted under its own category, and the count is reported.

Severity decides what happens to the episode, and there are only three answers:

    block   credential-shaped with strong structure. Quarantine, no appeal.
    flag    could be a credential, could be source code talking about one.
            Quarantine pending human review. This tier is where the recall is.
    redact  an identifier, not a secret. Pseudonymised, never quarantined,
            because quarantining every episode containing a home path would
            quarantine the corpus.

No pattern in this file has a capture group whose contents are ever returned.
The scanner API is counts and spans; spans are used to overwrite, never to read.
"""

from __future__ import annotations

import re
from typing import Iterator, NamedTuple

__all__ = ["Detector", "BENIGN", "DETECTORS", "detectors_by_severity", "iter_spans"]


class Detector(NamedTuple):
    category: str
    severity: str
    pattern: re.Pattern[str]


def _d(category: str, severity: str, regex: str, flags: int = 0) -> Detector:
    return Detector(category, severity, re.compile(regex, flags))


#: Structural, non-secret token shapes present in every unified diff. Consumed
#: first so they cannot inflate the entropy detectors. Counted, and reported.
BENIGN: tuple[Detector, ...] = (
    _d("git_index_line", "redact", r"^index [0-9a-f]{7,40}\.\.[0-9a-f]{7,40}(?: \d{6})?$", re.M),
    _d("git_object_id", "redact", r"\b(?:blob|commit|tree) [0-9a-f]{7,40}\b"),
    _d("uuid", "redact", r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    _d("sha256_literal", "redact", r'"(?:content_)?sha256"\s*:\s*"[0-9a-f]{64}"'),
    # A content-addressed cache file: a hex basename with a data extension,
    # optionally with its directory prefix. Verified against this store: every
    # single high-entropy hit in the first pass was `data/cache/<arm>/<sha256>.json`,
    # a filename that is already committed to the repository. It is entropy without
    # secrecy, and left unrecognised it quarantined 59 percent of the corpus.
    # Note the directory prefix must be consumed too, because `/` is inside the
    # base64 character class and the whole path would otherwise match as one token.
    _d("content_addressed_path", "redact",
       r"(?:[A-Za-z0-9_.\-]+/)*[0-9a-f]{32,64}\.(?:json|jsonl|txt|npy|bin|pkl|parquet|csv)\b"),
)

#: The credential tier. Every one of these is a published vendor token shape or
#: an unambiguous credential envelope.
_BLOCK: tuple[Detector, ...] = (
    _d("aws_access_key_id", "block", r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
    _d("github_token", "block", r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
    _d("github_fine_grained", "block", r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b"),
    _d("anthropic_key", "block", r"\bsk-ant-[A-Za-z0-9_\-]{24,}"),
    _d("openai_key", "block", r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
    _d("google_api_key", "block", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    _d("slack_token", "block", r"\bxox[baprs]-[0-9A-Za-z\-]{10,}"),
    _d("stripe_key", "block", r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b"),
    _d("npm_token", "block", r"\bnpm_[A-Za-z0-9]{36}\b"),
    _d("pypi_token", "block", r"\bpypi-[A-Za-z0-9_\-]{50,}"),
    _d("hf_token", "block", r"\bhf_[A-Za-z0-9]{34,}\b"),
    _d("private_key_block", "block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    _d("ssh_public_key", "block", r"\bssh-(?:rsa|ed25519|dss) AAAA[0-9A-Za-z+/=]{20,}"),
    _d("jwt", "block", r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    _d("url_with_credentials", "block", r"\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
    _d("db_connection_string", "block",
       r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s'\"]+"),
    _d("authorization_header", "block",
       r"(?i)\bauthorization\s*[:=]\s*['\"]?(?:bearer|basic|token)\s+[A-Za-z0-9_\-.=+/]{12,}"),
)

#: The recall tier. Deliberately noisy. A hit here quarantines the episode and a
#: human decides; the report states how many episodes this tier alone costs, so
#: the price of the recall is visible rather than hidden.
_FLAG: tuple[Detector, ...] = (
    _d("credential_assignment", "flag",
       r"(?i)\b(?:api[_\-]?key|apikey|secret[_\-]?key|secret|token|password|passwd|pwd|"
       r"credential|access[_\-]?key|private[_\-]?key|auth[_\-]?token|client[_\-]?secret)\b"
       r"\s*[:=]\s*['\"][^'\"\n]{8,}['\"]"),
    _d("env_assignment", "flag",
       r"(?m)^\s*(?:export\s+)?[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)"
       r"\s*=\s*\S{8,}"),
    _d("high_entropy_hex", "flag", r"\b[0-9a-f]{40,}\b"),
    _d("high_entropy_base64", "flag", r"\b[A-Za-z0-9+/]{48,}={0,2}\b"),
    _d("bearer_literal", "flag", r"(?i)\bbearer\s+[A-Za-z0-9_\-.=+/]{20,}"),
)

#: Identifiers. Pseudonymised in place. Never a reason to drop an episode.
_REDACT: tuple[Detector, ...] = (
    _d("email", "redact", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    _d("home_path", "redact", r"/(?:home|Users)/[A-Za-z0-9._\-]+"),
    _d("private_ipv4", "redact",
       r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    _d("internal_hostname", "redact",
       r"\b[A-Za-z0-9\-]+\.(?:local|internal|lan|home|corp|intranet)\b"),
    _d("github_repo_url", "redact",
       r"\bgithub\.com[/:][A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+"),
    _d("phone_e164", "redact", r"(?<![\w.\-])\+[1-9]\d{9,14}(?![\w.\-])"),
)

DETECTORS: tuple[Detector, ...] = _BLOCK + _FLAG + _REDACT


def detectors_by_severity(severity: str) -> tuple[Detector, ...]:
    return tuple(d for d in DETECTORS if d.severity == severity)


def iter_spans(text: str, detector: Detector) -> Iterator[tuple[int, int]]:
    """Yield match spans. The caller gets offsets, never the matched substring."""
    for match in detector.pattern.finditer(text):
        yield match.span()
