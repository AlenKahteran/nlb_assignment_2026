from __future__ import annotations

from dataclasses import dataclass
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None

FORBIDDEN_HOST_PARTS = (
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "reddit.com",
)

FORBIDDEN_URL_PARTS = (
    "login",
    "signin",
    "sign-in",
    "register",
    "account",
    "password",
    "paywall",
    "subscribe",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UrlSafety:
    """Result of checking whether a URL is suitable for this crawler."""

    allowed: bool
    reason: str


def assess_url(url: str) -> UrlSafety:
    """Check a URL for unsupported schemes, social sites, and account/paywall paths."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https", "file"}:
        reason = f"unsupported scheme: {parsed.scheme}"
        logger.warning("Rejected URL %s: %s", url, reason)
        return UrlSafety(False, reason)

    host = parsed.netloc.lower()
    lowered = url.lower()

    if any(part in host for part in FORBIDDEN_HOST_PARTS):
        reason = "social/private forum source is outside assignment scope"
        logger.info("Rejected URL %s: %s", url, reason)
        return UrlSafety(False, reason)

    if any(part in lowered for part in FORBIDDEN_URL_PARTS):
        reason = "login/paywall/account-like URL skipped"
        logger.info("Rejected URL %s: %s", url, reason)
        return UrlSafety(False, reason)

    logger.debug("Accepted URL candidate: %s", url)

    return UrlSafety(True, "public URL candidate")


def robots_allows(url: str, user_agent: str, timeout: int) -> tuple[bool, str]:
    """Fetch and evaluate robots.txt for a URL, failing open if robots cannot be read."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        logger.debug("Skipping robots.txt for non-HTTP URL: %s", url)
        return True, "robots.txt not applicable"

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)

    if httpx is None:
        logger.warning("Skipping robots.txt check for %s because httpx is unavailable", url)
        return True, "robots.txt check skipped: httpx not installed"

    try:
        logger.debug("Fetching robots.txt from %s", robots_url)
        response = httpx.get(robots_url, timeout=min(timeout, 10), follow_redirects=True)

        if response.status_code >= 400:
            logger.info("robots.txt unavailable for %s: status=%s", url, response.status_code)
            return True, f"robots.txt unavailable ({response.status_code})"

        parser.parse(response.text.splitlines())
        allowed = parser.can_fetch(user_agent, url)

        if allowed:
            logger.debug("robots.txt allows %s", url)
        else:
            logger.warning("robots.txt disallows %s", url)

        return allowed, "allowed by robots.txt" if allowed else "disallowed by robots.txt"

    except Exception as exc:  # noqa: BLE001
        logger.warning("robots.txt check failed open for %s: %s", url, type(exc).__name__)
        return True, f"robots.txt check failed open: {type(exc).__name__}"
