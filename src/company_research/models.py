from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return the current UTC timestamp in compact ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Budget:
    """Runtime limits for search, crawling, downloads, and HTTP calls."""

    max_search_results: int = 100
    max_visited_pages: int = 50
    max_downloads: int = 16
    max_depth: int = 10
    max_runtime_seconds: int = 600
    crawl_delay_seconds: float = 1.0
    max_file_mb: int = 50
    http_timeout_seconds: int = 30


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for optional external LLM calls."""

    provider: str = ""
    endpoint: str = ""
    api_key: str = ""
    model_name: str = ""
    timeout_seconds: int = 60
    max_retries: int = 2
    mode: str = "provider"

    @property
    def should_call_provider(self) -> bool:
        """Return whether the configured LLM provider has enough settings to be called."""
        has_credentials = bool(self.endpoint and self.api_key)

        return self.mode != "mock" and has_credentials


@dataclass(frozen=True)
class QAConfig:
    """Configuration for how much stored evidence is sent to answer generation."""

    fact_evidence_limit: int = 25
    chunk_evidence_limit: int = 25


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration for search, budget, and answer generation."""

    search_provider: str = "brave"
    brave_search_api_key: str = ""
    search_fixture_path: str = ""
    budget: Budget = field(default_factory=Budget)
    llm: LLMConfig = field(default_factory=LLMConfig)
    qa: QAConfig = field(default_factory=QAConfig)

    def redacted(self) -> dict[str, Any]:
        """Return configuration suitable for logs and audit tables without secrets."""
        redacted_llm = {
            "provider": self.llm.provider,
            "endpoint": self.llm.endpoint,
            "api_key": "<redacted>" if self.llm.api_key else "",
            "model_name": self.llm.model_name,
            "timeout_seconds": self.llm.timeout_seconds,
            "max_retries": self.llm.max_retries,
            "mode": self.llm.mode,
        }

        return {
            "search_provider": self.search_provider,
            "brave_search_api_key": "<redacted>" if self.brave_search_api_key else "",
            "search_fixture_path": self.search_fixture_path,
            "budget": self.budget.__dict__,
            "llm": redacted_llm,
            "qa": self.qa.__dict__,
        }


@dataclass(frozen=True)
class SearchResult:
    """A single search engine result before ranking and fetching."""

    query: str
    url: str
    title: str = ""
    snippet: str = ""
    rank: int = 0
    publisher: str = ""
    age: str = ""


@dataclass(frozen=True)
class SourceDecision:
    """The ranking decision explaining whether a source should be visited."""

    accepted: bool
    reason: str
    source_type: str
    score: int


@dataclass(frozen=True)
class FetchResult:
    """Downloaded source content and metadata captured during retrieval."""

    url: str
    status_code: int
    content_type: str
    final_url: str
    body: bytes
    retrieved_at: str
    file_path: Path | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized text, chunks, links, and metadata extracted from a fetched source."""

    title: str
    publisher: str
    source_type: str
    publication_date: str | None
    text: str
    chunks: list[str]
    links: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExtractedFact:
    """A metric/value fact extracted from document text with supporting evidence."""

    metric: str
    value: str
    unit: str
    period: str
    entity_level: str
    evidence: str
    source_id: int
    document_id: int
    confidence: str = "medium"


@dataclass(frozen=True)
class Citation:
    """Evidence metadata included with generated answers."""

    url: str
    title: str
    publisher: str
    source_type: str
    publication_date: str | None
    retrieved_at: str
    evidence: str


@dataclass(frozen=True)
class Answer:
    """A cited answer produced from the persisted knowledge base."""

    answer: str
    confidence: str
    citations: list[Citation]
    warnings: list[str]
    limitations: list[str]
    audit_ref: str

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the answer and citations to JSON-compatible primitives."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": [citation.__dict__ for citation in self.citations],
            "warnings": self.warnings,
            "limitations": self.limitations,
            "audit_ref": self.audit_ref,
        }
