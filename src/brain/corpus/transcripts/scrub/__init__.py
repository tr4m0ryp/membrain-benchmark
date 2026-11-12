"""Scrubbing: detect, quarantine, pseudonymise.

Order is fixed and is a safety property, not a preference:

    scan  ->  quarantine  ->  pseudonymise  ->  RE-SCAN  ->  write

The re-scan is the part that matters. Pseudonymisation rewrites text, and a
rewrite that reintroduced a credential shape would otherwise go unnoticed, so the
pipeline asserts the surviving set scans clean of every `block` and `flag`
category before a single byte is written.
"""

from __future__ import annotations

from brain.corpus.transcripts.scrub.patterns import (
    BENIGN,
    DETECTORS,
    Detector,
    detectors_by_severity,
)
from brain.corpus.transcripts.scrub.pseudonymise import (
    MAP_FILE,
    SALT_FILE,
    Pseudonymiser,
    load_or_create_salt,
)
from brain.corpus.transcripts.scrub.scan import (
    MASK,
    aggregate,
    mask_benign,
    redact_spans,
    scan_text,
    verdict,
)

__all__ = [
    "BENIGN", "DETECTORS", "Detector", "detectors_by_severity",
    "MAP_FILE", "SALT_FILE", "Pseudonymiser", "load_or_create_salt",
    "MASK", "aggregate", "mask_benign", "redact_spans", "scan_text", "verdict",
]
