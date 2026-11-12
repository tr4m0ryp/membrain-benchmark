"""What each incumbent needs before it can be executed, and what is missing.

Recorded as data rather than prose so the result file carries it. Every field
was established by reading the installed distribution or by running it, on
2026-08-01; the versions inspected are named, because these projects move
quickly and a requirement is only true of a version.

The `blocker` field is the one that decides a verdict. It names the exact
dependency that must be paid for, or `None` when the system runs offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from brain.baselines.incumbents import bridge

__all__ = ["Requirement", "REQUIREMENTS", "survey"]


@dataclass(frozen=True, slots=True)
class Requirement:
    """One system's runtime requirements, as inspected."""

    system: str
    version_inspected: str
    package: str
    llm_required: str
    llm_default: str
    llm_local_option: str | None
    database: str
    database_serverless: str | None
    embedder: str
    embedder_local_option: str | None
    network_required: bool
    blocker: str | None
    notes: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        system="mem0",
        version_inspected="mem0ai 2.0.15 (PyPI, installed 2026-08-01)",
        package="mem0ai",
        llm_required="yes, on every inferring write; the client is constructed eagerly in Memory.__init__",
        llm_default="OpenAI, gpt-4o-mini class; Memory() raises openai.OpenAIError without OPENAI_API_KEY",
        llm_local_option=(
            "no in-process option. LlmFactory offers ollama, vllm and lmstudio, all of which are "
            "HTTP to a local daemon, and none is installed here. A provider slot can be claimed "
            "in process, which is what this probe does"
        ),
        database="vector store; default Qdrant in embedded local-path mode, no daemon",
        database_serverless="yes: faiss or qdrant on a local path, plus a stdlib sqlite3 history db",
        embedder="OpenAI text-embedding-3-small by default",
        embedder_local_option="yes: provider 'huggingface', sentence-transformers in process",
        network_required=False,
        blocker=None,
        notes=(
            "Runs fully offline once the LLM is substituted. Telemetry defaults to ON "
            "(MEM0_TELEMETRY) and posts to PostHog; it is disabled for every run here. "
            "Graph memory was REMOVED in mem0ai 2.0.0, so the paper's Mem0^g arm cannot be "
            "reproduced from current PyPI at all; 1.0.11 is the last version carrying it."
        ),
        sources=("mem0/memory/main.py", "mem0/utils/factory.py", "mem0/configs/prompts.py"),
    ),
    Requirement(
        system="mem0-legacy",
        version_inspected="mem0ai 0.1.118 (PyPI, installed 2026-08-01)",
        package="mem0ai",
        llm_required="yes, two calls per write: fact extraction, then the ADD/UPDATE/DELETE/NONE decision",
        llm_default="OpenAI",
        llm_local_option="same as above",
        database="vector store plus sqlite3 history",
        database_serverless="yes",
        embedder="OpenAI by default",
        embedder_local_option="yes: huggingface provider",
        network_required=False,
        blocker=None,
        notes=(
            "Included because it carries the two-branch write the paper's claim is about. "
            "In 2.0.15 that branch set is gone: the update-decision prompt and "
            "get_update_memory_messages are dead code, called from nowhere in the package."
        ),
        sources=("mem0/memory/main.py:424-473",),
    ),
    Requirement(
        system="graphiti",
        version_inspected="graphiti-core 0.29.2 (the version the paper's source audit recorded)",
        package="graphiti-core",
        llm_required="yes, unconditionally, on every add_episode; no skip flag exists",
        llm_default="OpenAI; Graphiti() hard-fails without OPENAI_API_KEY on the default path",
        llm_local_option="OpenAIGenericClient against a local OpenAI-compatible daemon (ollama, vLLM); none installed here",
        database="graph database; default Neo4j driver, a server",
        database_serverless="Kuzu only, and it is DEPRECATED in 0.29.x with a warning on every construction",
        embedder="OpenAI text-embedding-3-small",
        embedder_local_option=(
            "none. Every EmbedderClient subclass shipped calls a hosted API; the "
            "sentence-transformers extra provides a reranker, not an embedder"
        ),
        network_required=True,
        blocker=(
            "an OpenAI-compatible LLM endpoint AND an embedding endpoint. No local embedder "
            "class exists, and no local LLM daemon is installed on this machine. End-to-end "
            "Graphiti is BLOCKED; its write-decision function is not, and is executed instead"
        ),
        notes=(
            "resolve_extracted_edge takes its candidate edges as plain arguments and touches no "
            "driver, so the branch set is executable with no database and no key. Its response "
            "model EdgeDuplicate has exactly two fields, so the space is finite and is enumerated."
        ),
        sources=("graphiti_core/utils/maintenance/edge_operations.py:623", "graphiti_core/prompts/dedupe_edges.py:24"),
    ),
    Requirement(
        system="toki",
        version_inspected="github.com/ZhoeMrety/toki-bitemporal-memory (arXiv:2606.06240, v1 2026-06-04)",
        package="not on PyPI; distribution name 'bitemporal' 0.0.3, installed from a clone",
        llm_required="optional; the structural verdicts and ablations run with no key",
        llm_default="an OpenRouter-compatible endpoint for the utility slice only",
        llm_local_option="the LLM-free path is the documented default for everything except the G2 utility slice",
        database="DuckDB, embedded",
        database_serverless="yes",
        embedder="none; retrieval baseline is BM25",
        embedder_local_option="n/a",
        network_required=False,
        blocker=None,
        notes=(
            "The repository URL printed in the paper 404s; the live mirror is the one recorded "
            "above. The paper makes no utility-superiority claim, so its numbers are mechanism "
            "deltas rather than benchmark scores. Its core bitemporal suite was RUN here: 308 "
            "tests pass offline with no key and no server. Its cross-system anomaly-wire sweep "
            "did NOT run: it needs sqlalchemy for the Letta adapter and a pinned mem0 clone at "
            "a623cfaf76ae7379a58be1e837f8a88a9b15a184. TOKI ships a deterministic hash-based "
            "judge stub over the real write paths of mem0, Graphiti and Letta, which is prior "
            "art for the method used in this report and must be cited as such."
        ),
        sources=("arXiv:2606.06240", "repository README", "experiments/anomaly_wire/adapter_base.py"),
    ),
    Requirement(
        system="nous",
        version_inspected="nous-state 0.2.0 (PyPI, 2026-06-23; arXiv:2606.22030 v2 2026-07-16)",
        package="nous-state",
        llm_required="no; the shipped extractor is rule-based",
        llm_default="LLMExtractor is opt-in and defaults to an OpenRouter model",
        llm_local_option="not needed; the default path uses no model at all",
        database="stdlib sqlite3",
        database_serverless="yes",
        embedder="none required; retrieval is BM25 plus a two-hop entity walk",
        embedder_local_option="n/a",
        network_required=False,
        blocker=None,
        notes=(
            "Zero declared runtime dependencies. The only system here that runs end to end with "
            "its own components and nothing substituted, so it is the cleanest arm in the set. "
            "Reproducing the PAPER's benchmark does need a key; the write path does not."
        ),
        sources=("nous/engine.py:141", "nous/persistence.py"),
    ),
)


def survey() -> dict[str, Any]:
    """Requirements plus which interpreters this machine actually has."""
    availability: dict[str, Any] = {}
    for key in ("mem0", "mem0_legacy", "graphiti", "nous", "toki"):
        try:
            availability[key] = bridge.venv_for(key).to_dict()
        except Exception as error:
            availability[key] = {"available": False, "reason": str(error)}
    return {
        "inspected_on": "2026-08-01",
        "requirements": [item.to_dict() for item in REQUIREMENTS],
        "interpreters": availability,
        "credentials_present": {
            "ANTHROPIC_API_KEY": False,
            "OPENAI_API_KEY": False,
            "note": "no credential was obtained, purchased, or registered for during this study",
        },
    }
