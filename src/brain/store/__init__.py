"""Episodic trace store: one portable SQLite file plus sqlite-vec k-NN (T2)."""

from brain.store.schema import (
    NEVER_INVALID,
    SCHEMA_VERSION,
    Trace,
    sha256_text,
)
from brain.store.vectors import VECTOR_DTYPE
from brain.store.store import (
    DimensionMismatchError,
    DuplicateTraceIdError,
    EpisodicStore,
    InvalidVectorError,
    StoreError,
)

__all__ = [
    "EpisodicStore",
    "VECTOR_DTYPE",
    "Trace",
    "sha256_text",
    "SCHEMA_VERSION",
    "NEVER_INVALID",
    "StoreError",
    "DuplicateTraceIdError",
    "InvalidVectorError",
    "DimensionMismatchError",
]
