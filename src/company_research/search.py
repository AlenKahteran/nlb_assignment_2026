from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None

from .models import AppConfig, SearchResult

logger = logging.getLogger(__name__)


class SearchClient(ABC):
    """Interface implemented by online and fixture-backed search clients."""

    @abstractmethod
    def search(self, query: str, count: int) -> list[SearchResult]:
        """Return up to count web results for a query."""
        raise NotImplementedError


class BraveSearchClient(SearchClient):
    """Search client backed by the Brave Web Search API."""

    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout: int):
        """Store Brave API credentials and request timeout."""
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, count: int) -> list[SearchResult]:
        """Query Brave Search and normalize web results into SearchResult records."""
        if not self.api_key:
            logger.error("Brave search requested without BRAVE_SEARCH_API_KEY")
            raise RuntimeError("BRAVE_SEARCH_API_KEY is required when SEARCH_PROVIDER=brave")

        if httpx is None:
            logger.error("Brave search requested but httpx is unavailable")
            raise RuntimeError("Brave search requires httpx; install project dependencies first")

        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {"q": query, "count": min(count, 20), "text_decorations": False}

        logger.info("Searching Brave query=%r count=%s", query, count)

        response = httpx.get(self.endpoint, headers=headers, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        web_results = data.get("web", {}).get("results", [])
        results: list[SearchResult] = []

        for rank, item in enumerate(web_results[:count], start=1):
            results.append(
                SearchResult(
                    query=query,
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                    rank=rank,
                    publisher=item.get("profile", {}).get("name", ""),
                    age=item.get("age", ""),
                )
            )

        filtered = [result for result in results if result.url]
        logger.info("Brave returned %s usable results for query=%r", len(filtered), query)

        return filtered


class FixtureSearchClient(SearchClient):
    """Deterministic search adapter for tests and offline reviewer sanity checks."""

    def __init__(self, fixture_path: str | Path):
        """Load fixture search results from a JSON file."""
        self.fixture_path = Path(fixture_path)

        logger.info("Loading search fixture from %s", self.fixture_path)

        self.data = json.loads(self.fixture_path.read_text(encoding="utf-8"))

    def search(self, query: str, count: int) -> list[SearchResult]:
        """Return matching fixture results for a query."""
        normalized = query.lower()
        candidates = self.data.get(query) or []

        if not candidates:
            for key, value in self.data.items():
                if key.lower() in normalized or normalized in key.lower():
                    candidates = value
                    break

        logger.info("Fixture search query=%r matched %s candidates", query, len(candidates))

        results: list[SearchResult] = []

        for index, item in enumerate(candidates[:count], start=1):
            url = item.get("url", "")

            if not url and item.get("path"):
                url = (self.fixture_path.parent / item["path"]).resolve().as_uri()

            results.append(
                SearchResult(
                    query=query,
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    rank=index,
                    publisher=item.get("publisher", ""),
                    age=item.get("age", ""),
                )
            )

        filtered = [result for result in results if result.url]
        logger.debug("Fixture search returned %s usable results for query=%r", len(filtered), query)

        return filtered


def build_search_client(config: AppConfig) -> SearchClient:
    """Create the configured search client from application settings."""
    if config.search_fixture_path:
        logger.info("Using fixture search provider")
        return FixtureSearchClient(config.search_fixture_path)

    if config.search_provider.lower() == "brave":
        logger.info("Using Brave search provider")
        return BraveSearchClient(config.brave_search_api_key, config.budget.http_timeout_seconds)

    logger.error("Unsupported search provider: %s", config.search_provider)

    raise ValueError(f"Unsupported SEARCH_PROVIDER: {config.search_provider}")


def generate_queries(company_name: str) -> list[str]:
    """Build the fixed query set used to find public FY2024 company sources."""
    logger.debug("Generating search queries for company=%r", company_name)

    return [
        f'"{company_name}" 2024 annual report financial report',
        f'"{company_name}" FY2024 financial results annual report PDF',
        f'"{company_name}" investor relations annual report 2024',
        f'"{company_name}" public register financial statements 2024',
        f'"{company_name}" stock exchange annual report 2024',
    ]
