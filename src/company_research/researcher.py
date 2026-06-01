from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from .extractors import extract_document
from .facts import dedupe_facts, extract_facts_from_text
from .fetch import fetch_rendered_html, fetch_url
from .kb import KnowledgeBase
from .models import AppConfig, SearchResult
from .ranking import score_source
from .search import build_search_client, generate_queries
from .safety import assess_url

logger = logging.getLogger(__name__)


def _should_render(result: SearchResult) -> bool:
    """Return whether a search result should be fetched with browser rendering."""
    lowered = result.url.lower()

    return not any(
        lowered.endswith(ext)
        for ext in (".pdf", ".csv", ".xlsx", ".docx", ".json", ".xml", ".txt")
    )


def _linked_search_result(parent: SearchResult, url: str, title: str, publisher: str) -> SearchResult:
    """Build a search-result-like record for a URL discovered inside an accepted source."""
    return SearchResult(
        query=f"linked from {parent.url}",
        url=url,
        title=title or parent.title,
        snippet=f"Linked from {title or parent.title or parent.url}",
        rank=0,
        publisher=publisher or parent.publisher,
    )


def research_company(kb: KnowledgeBase, company_name: str, config: AppConfig) -> str:
    """Run bounded search, fetch, extraction, fact persistence, and audit logging for a company."""
    logger.info("Starting research for company=%r", company_name)

    run_id = kb.start_run(company_name, config.redacted())
    company_id = kb.upsert_company(company_name)
    search_client = build_search_client(config)
    started = time.monotonic()
    visited = 0
    downloads = 0

    kb.log_event(run_id, "research_started", f"researching {company_name}", {"company_id": company_id})
    try:
        candidates: list[tuple[SearchResult, int, int]] = []
        queued_urls: set[str] = set()

        for query in generate_queries(company_name):
            if time.monotonic() - started > config.budget.max_runtime_seconds:
                logger.warning("Runtime budget reached before search completed for run_id=%s", run_id)
                kb.log_event(run_id, "budget_stop", "runtime budget reached before search completed")
                break

            query_id = kb.log_query(run_id, company_id, query)
            results = search_client.search(query, config.budget.max_search_results)

            logger.info("Search query_id=%s returned %s results", query_id, len(results))

            kb.log_search_results(query_id, results)

            for result in results:
                decision = score_source(result, company_name)

                kb.upsert_source(run_id, company_id, result, decision)
                kb.log_event(
                    run_id,
                    "source_decision",
                    decision.reason,
                    {"url": result.url, "score": decision.score, "source_type": decision.source_type},
                )

                if decision.accepted:
                    candidates.append((result, decision.score, 0))
                    queued_urls.add(result.url)

        candidates.sort(key=lambda item: item[1], reverse=True)
        logger.info("Research run_id=%s has %s accepted candidate sources", run_id, len(candidates))

        seen_urls: set[str] = set()

        while candidates:
            candidates.sort(key=lambda item: item[1], reverse=True)
            result, _score, depth = candidates.pop(0)

            if visited >= config.budget.max_visited_pages:
                logger.warning("Visited page budget reached for run_id=%s", run_id)
                kb.log_event(run_id, "budget_stop", "visited page budget reached")
                break

            if time.monotonic() - started > config.budget.max_runtime_seconds:
                logger.warning("Runtime budget reached for run_id=%s", run_id)
                kb.log_event(run_id, "budget_stop", "runtime budget reached")
                break

            if result.url in seen_urls:
                logger.debug("Skipping duplicate candidate URL: %s", result.url)
                continue

            seen_urls.add(result.url)

            safety = assess_url(result.url)

            if not safety.allowed:
                logger.info("Skipping unsafe source url=%s reason=%s", result.url, safety.reason)
                kb.log_event(run_id, "source_skipped", safety.reason, {"url": result.url})
                continue

            decision = score_source(result, company_name)
            existing_source = kb.parsed_source_for_url(company_id, result.url)

            if existing_source:
                message = "already parsed source skipped"
                logger.info(
                    "Skipping already parsed source url=%s source_id=%s document_id=%s",
                    result.url,
                    existing_source["source_id"],
                    existing_source["document_id"],
                )
                kb.log_event(
                    run_id,
                    "source_skipped",
                    message,
                    {
                        "url": result.url,
                        "source_id": existing_source["source_id"],
                        "document_id": existing_source["document_id"],
                        "content_sha256": existing_source["content_sha256"],
                    },
                )
                continue

            try:
                logger.info("Processing source url=%s score=%s depth=%s", result.url, _score, depth)

                if _should_render(result):
                    fetch = fetch_rendered_html(result.url, kb.raw_dir, config.budget)
                else:
                    if downloads >= config.budget.max_downloads:
                        logger.warning("Download budget reached for run_id=%s", run_id)
                        kb.log_event(run_id, "budget_stop", "download budget reached")
                        continue

                    fetch = fetch_url(result.url, kb.raw_dir, config.budget)
                    downloads += 1

                visited += 1

                if fetch.status_code >= 400 or not fetch.body:
                    logger.warning(
                        "Fetch failed url=%s status=%s body_bytes=%s warnings=%s",
                        result.url,
                        fetch.status_code,
                        len(fetch.body),
                        fetch.warnings,
                    )
                    kb.log_event(
                        run_id,
                        "fetch_failed",
                        f"status {fetch.status_code}",
                        {"url": result.url, "warnings": fetch.warnings},
                    )
                    continue

                content_sha256 = hashlib.sha256(fetch.body).hexdigest()
                existing_document = kb.parsed_document_for_hash(company_id, content_sha256)

                if existing_document:
                    source_id = kb.upsert_source(run_id, company_id, result, decision, fetch)
                    message = "duplicate document content skipped"
                    logger.info(
                        "Skipping duplicate document content url=%s source_id=%s existing_document_id=%s",
                        result.url,
                        source_id,
                        existing_document["document_id"],
                    )
                    kb.log_event(
                        run_id,
                        "source_skipped",
                        message,
                        {
                            "url": result.url,
                            "source_id": source_id,
                            "existing_source_id": existing_document["source_id"],
                            "existing_document_id": existing_document["document_id"],
                            "content_sha256": content_sha256,
                        },
                    )
                    continue

                extractor, doc = extract_document(fetch)
                source_id = kb.upsert_source(run_id, company_id, result, decision, fetch, doc)
                document_id = kb.insert_document(source_id, fetch, doc, extractor, content_sha256)
                facts = dedupe_facts(extract_facts_from_text(doc.text, source_id, document_id))

                kb.insert_facts(facts)

                logger.info(
                    "Processed source_id=%s document_id=%s chunks=%s facts=%s",
                    source_id,
                    document_id,
                    len(doc.chunks),
                    len(facts),
                )

                kb.log_event(
                    run_id,
                    "source_extracted",
                    f"extracted {len(doc.chunks)} chunks and {len(facts)} facts",
                    {
                        "url": result.url,
                        "extractor": extractor,
                        "source_id": source_id,
                        "document_id": document_id,
                        "depth": depth,
                    },
                )

                if depth < config.budget.max_depth:
                    discovered = 0

                    for link in doc.links:
                        if link in seen_urls or link in queued_urls:
                            continue

                        link_safety = assess_url(link)

                        if not link_safety.allowed:
                            kb.log_event(
                                run_id,
                                "source_skipped",
                                link_safety.reason,
                                {"url": link, "depth": depth + 1},
                            )
                            continue

                        linked_result = _linked_search_result(result, link, doc.title, doc.publisher)
                        linked_decision = score_source(linked_result, company_name)

                        kb.upsert_source(run_id, company_id, linked_result, linked_decision)
                        kb.log_event(
                            run_id,
                            "source_decision",
                            linked_decision.reason,
                            {
                                "url": link,
                                "score": linked_decision.score,
                                "source_type": linked_decision.source_type,
                                "depth": depth + 1,
                                "parent_url": result.url,
                            },
                        )

                        queued_urls.add(link)

                        if linked_decision.accepted:
                            candidates.append((linked_result, linked_decision.score, depth + 1))
                            discovered += 1

                    if discovered:
                        logger.info(
                            "Queued %s linked sources from source_id=%s at next depth=%s",
                            discovered,
                            source_id,
                            depth + 1,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error while processing source url=%s", result.url)
                kb.log_event(
                    run_id,
                    "source_error",
                    f"{type(exc).__name__}: {exc}",
                    {"url": result.url},
                )

        kb.finish_run(run_id)
    except Exception:
        logger.exception("Research run failed run_id=%s company=%r", run_id, company_name)
        kb.finish_run(run_id, "failed")
        raise

    logger.info(
        "Completed research run_id=%s company=%r visited=%s downloads=%s",
        run_id,
        company_name,
        visited,
        downloads,
    )

    return run_id


def research_companies(kb: KnowledgeBase, companies: list[str], config: AppConfig) -> list[str]:
    """Research each non-empty company name and return their run ids."""
    run_ids: list[str] = []

    for company in companies:
        if company.strip():
            logger.info("Researching company from batch: %r", company.strip())
            run_ids.append(research_company(kb, company.strip(), config))

    logger.info("Completed batch research for %s companies", len(run_ids))

    return run_ids


def load_companies(path: str | Path) -> list[str]:
    """Load company names from a newline-delimited text file."""
    companies = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    logger.info("Loaded %s companies from %s", len(companies), path)

    return companies
