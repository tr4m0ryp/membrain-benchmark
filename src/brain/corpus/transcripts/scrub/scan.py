"""The secret scanner. Returns counts and severities. Never returns a value.

The one design rule: `scan_text` has no return path that carries matched text.
It hands back a category-to-count mapping and the total masked length, and the
only thing it does with a span is overwrite it. A caller that wants to know WHAT
was found has to open the quarantine file on the local disk, which is gitignored
and never leaves the machine.

`mask_benign` runs first and replaces structural diff hex with a fixed-width
placeholder, so `high_entropy_hex` counts real anomalies rather than every git
index line in the store.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from brain.corpus.transcripts.scrub.patterns import BENIGN, DETECTORS, Detector
from brain.corpus.transcripts.types import ScanFinding, ScanReport

__all__ = ["MASK", "scan_text", "mask_benign", "redact_spans", "aggregate", "verdict"]

#: What a blocked span is replaced with. Fixed width so the replacement itself
#: carries no information about the length of what it replaced.
MASK = "[REDACTED]"

_BENIGN_MASK = "[GITOBJ]"


def mask_benign(text: str) -> tuple[str, dict[str, int]]:
    """Replace documented non-secret token shapes, returning per-shape counts."""
    counts: dict[str, int] = {}
    out = text
    for detector in BENIGN:
        out, n = detector.pattern.subn(_BENIGN_MASK, out)
        if n:
            counts[detector.category] = counts.get(detector.category, 0) + n
    return out, counts


def redact_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Overwrite spans with `MASK`. Spans are consumed, never read."""
    ordered = sorted(spans)
    if not ordered:
        return text
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append(MASK)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def scan_text(
    text: str,
    *,
    detectors: tuple[Detector, ...] = DETECTORS,
    redact: bool = True,
) -> tuple[str, dict[str, int]]:
    """Scan one text.

    Returns the text with every `block` and `flag` span overwritten (when
    `redact`), plus a `{category: count}` mapping. `redact` categories are left
    in place: they are handled by pseudonymisation, which needs the original
    string to compute a stable pseudonym for it.
    """
    masked, counts = mask_benign(text)
    spans: list[tuple[int, int]] = []
    for detector in detectors:
        hits = list(detector.pattern.finditer(masked))
        if not hits:
            continue
        counts[detector.category] = counts.get(detector.category, 0) + len(hits)
        if detector.severity in ("block", "flag"):
            spans.extend(match.span() for match in hits)
    if redact and spans:
        masked = redact_spans(masked, spans)
    return masked, counts


def verdict(counts: Mapping[str, int]) -> str:
    """`block`, `flag`, `redact` or `clean`, from the categories alone."""
    severities = {d.category: d.severity for d in DETECTORS}
    hit = {severities.get(c) for c, n in counts.items() if n > 0 and c in severities}
    for severity in ("block", "flag", "redact"):
        if severity in hit:
            return severity
    return "clean"


def aggregate(per_text: Iterable[Mapping[str, int]]) -> ScanReport:
    """Roll per-text counts into a report. Counts and categories only."""
    severities = {d.category: d.severity for d in DETECTORS}
    total: Counter[str] = Counter()
    n_scanned = n_clean = n_blocked = n_flagged = n_redacted = 0
    for counts in per_text:
        n_scanned += 1
        total.update({c: n for c, n in counts.items() if n})
        outcome = verdict(counts)
        if outcome == "block":
            n_blocked += 1
        elif outcome == "flag":
            n_flagged += 1
        elif outcome == "redact":
            n_redacted += 1
        else:
            n_clean += 1
    findings = tuple(
        ScanFinding(category=category, severity=severities.get(category, "redact"), count=count)
        for category, count in sorted(total.items())
    )
    return ScanReport(
        n_scanned=n_scanned,
        n_clean=n_clean,
        n_blocked=n_blocked,
        n_flagged=n_flagged,
        n_redacted=n_redacted,
        findings=findings,
    )
