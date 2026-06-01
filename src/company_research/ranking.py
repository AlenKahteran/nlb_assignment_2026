from __future__ import annotations

import logging
from urllib.parse import urlparse

from .models import SearchResult, SourceDecision
from .safety import assess_url

OFFICIAL_HINTS = ("annual", "report", "investor", "financial", "results", "ir", "cdn")
AUTHORITY_HINTS = ("nlb", "petrol", "gen-i", "geni", "ljse", "seonet", "ajpes", "europa", "gov", "bsi")
SECONDARY_HINTS = ("news", "bloomberg", "reuters", "finance.yahoo", "marketscreener")

logger = logging.getLogger(__name__)


def classify_source(url: str, title: str = "") -> str:
    """Classify a source into a broad evidence category using URL and title hints."""
    lowered = f"{url} {title}".lower()

    if lowered.endswith(".pdf") or ".pdf" in lowered or "annual report" in lowered:
        source_type = "official_report"
    elif any(hint in lowered for hint in ("ljse", "seonet", "stock-exchange", "exchange")):
        source_type = "exchange_page"
    elif any(hint in lowered for hint in ("gov", "regulator", "bsi", "europa")):
        source_type = "regulator"
    elif any(hint in lowered for hint in ("register", "ajpes")):
        source_type = "public_register"
    elif any(hint in lowered for hint in SECONDARY_HINTS):
        source_type = "news"
    else:
        source_type = "company_page"

    logger.debug("Classified source url=%s title=%r as %s", url, title, source_type)

    return source_type


def score_source(result: SearchResult, company_name: str) -> SourceDecision:
    """Score a search result for relevance, authority, recency, and crawl safety."""
    safety = assess_url(result.url)

    if not safety.allowed:
        logger.info("Rejected source during scoring: url=%s reason=%s", result.url, safety.reason)
        return SourceDecision(False, safety.reason, "other", -100)

    lowered = f"{result.url} {result.title} {result.snippet}".lower()
    company_tokens = [token for token in company_name.lower().replace("-", " ").split() if len(token) > 2]
    score = 0

    if any(token in lowered for token in company_tokens):
        score += 25

    if any(hint in lowered for hint in OFFICIAL_HINTS):
        score += 20

    if "2024" in lowered or "fy2024" in lowered:
        score += 20

    if any(hint in lowered for hint in AUTHORITY_HINTS):
        score += 15

    if any(hint in lowered for hint in SECONDARY_HINTS):
        score -= 10

    host = urlparse(result.url).netloc.lower()

    if host and any(token.replace("group", "") in host for token in company_tokens):
        score += 15

    source_type = classify_source(result.url, result.title)
    accepted = score >= 20
    reason = "accepted: relevant authoritative public source" if accepted else "rejected: low relevance"

    logger.info(
        "Scored source accepted=%s score=%s type=%s url=%s",
        accepted,
        score,
        source_type,
        result.url,
    )

    return SourceDecision(accepted, reason, source_type, score)
