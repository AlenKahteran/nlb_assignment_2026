from __future__ import annotations

import hashlib
import logging
import mimetypes
import shutil
from pathlib import Path
from time import sleep
from urllib.parse import unquote, urlparse

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None

from .models import Budget, FetchResult, utc_now
from .safety import robots_allows

USER_AGENT = "company-research/0.1 (+public research; polite bounded crawler)"
logger = logging.getLogger(__name__)


def _failed_fetch_result(url: str, warnings: list[str] | tuple[str, ...]) -> FetchResult:
    """Return an empty fetch result for expected fetch/navigation failures."""
    return FetchResult(
        url=url,
        status_code=0,
        content_type="",
        final_url=url,
        body=b"",
        retrieved_at=utc_now(),
        warnings=tuple(warnings),
    )


def _short_error(exc: Exception) -> str:
    """Return a compact one-line description for logs and audit metadata."""
    message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__

    return f"{type(exc).__name__}: {message}"


def _safe_filename(url: str, content_type: str) -> str:
    """Create a deterministic safe local filename for fetched content."""
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name or "index.html"
    suffix = Path(name).suffix

    if not suffix:
        suffix = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".html"

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    stem = Path(name).stem[:80] or "source"
    filename = f"{stem}-{digest}{suffix}"

    logger.debug("Mapped url=%s content_type=%s to filename=%s", url, content_type, filename)

    return filename


def fetch_url(url: str, raw_dir: Path, budget: Budget, *, check_robots: bool = True) -> FetchResult:
    """Fetch a URL or local file into raw_dir while enforcing robots and file-size limits."""
    logger.info("Fetching URL: %s", url)
    parsed = urlparse(url)
    retrieved_at = utc_now()

    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        target = raw_dir / _safe_filename(url, content_type)

        logger.info("Copying local file %s to %s", path, target)
        shutil.copyfile(path, target)

        body = target.read_bytes()
        logger.debug("Copied local file bytes=%s content_type=%s", len(body), content_type)

        return FetchResult(
            url=url,
            status_code=200,
            content_type=content_type,
            final_url=url,
            body=body,
            retrieved_at=retrieved_at,
            file_path=target,
        )

    warnings: list[str] = []

    if check_robots:
        allowed, reason = robots_allows(url, USER_AGENT, budget.http_timeout_seconds)
        warnings.append(reason)

        if not allowed:
            logger.warning("Skipping fetch because robots.txt disallowed url=%s reason=%s", url, reason)
            return FetchResult(
                url=url,
                status_code=0,
                content_type="",
                final_url=url,
                body=b"",
                retrieved_at=retrieved_at,
                warnings=tuple(warnings),
            )

    if httpx is None:
        logger.error("HTTP fetch requested but httpx is unavailable")
        raise RuntimeError("HTTP fetching requires httpx; install project dependencies first")

    sleep(max(budget.crawl_delay_seconds, 0))

    with httpx.Client(
        timeout=budget.http_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        logger.debug("Sending HTTP GET url=%s timeout=%s", url, budget.http_timeout_seconds)
        response = client.get(url)

        content_length = int(response.headers.get("content-length") or 0)
        max_bytes = budget.max_file_mb * 1024 * 1024

        if content_length > max_bytes:
            logger.error("Refusing oversized response url=%s bytes=%s max=%s", url, content_length, max_bytes)
            raise RuntimeError(f"file too large: {content_length} bytes")

        body = response.content[:max_bytes]

        if len(response.content) > max_bytes:
            logger.error("Download exceeded max file size url=%s max_mb=%s", url, budget.max_file_mb)
            raise RuntimeError(f"download exceeded max file size: {budget.max_file_mb} MB")

        content_type = response.headers.get("content-type", "application/octet-stream")
        target = raw_dir / _safe_filename(str(response.url), content_type)
        target.write_bytes(body)

        logger.info(
            "Fetched URL status=%s bytes=%s final_url=%s saved=%s",
            response.status_code,
            len(body),
            response.url,
            target,
        )

        return FetchResult(
            url=url,
            status_code=response.status_code,
            content_type=content_type,
            final_url=str(response.url),
            body=body,
            retrieved_at=retrieved_at,
            file_path=target,
            warnings=tuple(warnings),
        )


def fetch_rendered_html(url: str, raw_dir: Path, budget: Budget) -> FetchResult:
    """Render a web page with Playwright and save its final HTML, falling back to fetch_url."""
    logger.info("Fetching rendered HTML: %s", url)

    try:
        from playwright.sync_api import sync_playwright
    except Exception:  # noqa: BLE001
        logger.warning("Playwright unavailable; falling back to plain fetch for %s", url)
        return fetch_url(url, raw_dir, budget)

    parsed = urlparse(url)

    if parsed.scheme == "file":
        logger.debug("Local file does not need rendering: %s", url)
        return fetch_url(url, raw_dir, budget, check_robots=False)

    allowed, reason = robots_allows(url, USER_AGENT, budget.http_timeout_seconds)

    if not allowed:
        logger.warning("Skipping rendered fetch because robots.txt disallowed url=%s", url)
        return _failed_fetch_result(url, (reason,))

    sleep(max(budget.crawl_delay_seconds, 0))

    try:
        with sync_playwright() as pw:
            logger.debug("Launching headless browser for %s", url)
            browser = pw.chromium.launch(headless=True)

            try:
                page = browser.new_page(user_agent=USER_AGENT)
                response = page.goto(url, wait_until="networkidle", timeout=budget.http_timeout_seconds * 1000)
                html = page.content()
                final_url = page.url
                status = response.status if response else 200

            finally:
                browser.close()

    except Exception as exc:  # noqa: BLE001
        warning = f"rendered fetch failed: {_short_error(exc)}"

        logger.warning("Rendered fetch failed url=%s reason=%s", url, warning)

        return _failed_fetch_result(url, (reason, warning))

    body = html.encode("utf-8")
    target = raw_dir / _safe_filename(final_url, "text/html")
    target.write_bytes(body)

    logger.info("Rendered HTML status=%s bytes=%s final_url=%s saved=%s", status, len(body), final_url, target)

    return FetchResult(
        url=url,
        status_code=status,
        content_type="text/html; charset=utf-8",
        final_url=final_url,
        body=body,
        retrieved_at=utc_now(),
        file_path=target,
        warnings=(reason,),
    )
