from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict

from .kb import KnowledgeBase
from .llm import LLMClient
from .models import Answer, AppConfig, Citation

logger = logging.getLogger(__name__)

QUALITATIVE_QUESTION_PREFIXES = (
    "does ",
    "do ",
    "is ",
    "are ",
    "was ",
    "were ",
    "has ",
    "have ",
    "can ",
)

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "unsupported": 0}

SOURCE_TYPE_RANK = {
    "official_report": 4,
    "exchange_page": 3,
    "regulator": 3,
    "public_register": 2,
    "company_page": 1,
    "news": 0,
}


def _string_list(value) -> list[str]:
    """Normalize an LLM field into a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value]

    return []


def _confidence(value, fallback: str) -> str:
    """Normalize an LLM confidence label, falling back to deterministic confidence."""
    if isinstance(value, str) and value.lower() in {"high", "medium", "low", "unsupported"}:
        return value.lower()

    return fallback


def _is_qualitative_question(question: str) -> bool:
    """Return whether a question asks for a yes/no or descriptive answer."""
    lowered = question.strip().lower()

    return lowered.startswith(QUALITATIVE_QUESTION_PREFIXES)


def _citation_from_row(row) -> Citation:
    """Build an answer citation from a fact row or text chunk row."""
    return Citation(
        url=row["url"] or "",
        title=row["title"] or "",
        publisher=row["publisher"] or "",
        source_type=row["source_type"] or "other",
        publication_date=row["publication_date"],
        retrieved_at=row["retrieved_at"] or "",
        evidence=(row["evidence"] if "evidence" in row.keys() else row["text"])[:4096],
    )


def _years_from_text(text: str) -> set[str]:
    """Return four-digit reporting years mentioned in text."""
    return set(re.findall(r"20\d{2}", text or ""))


def _row_text(row, keys: tuple[str, ...]) -> str:
    """Join available row values for simple ranking signals."""
    row_keys = set(row.keys())

    return " ".join(str(row[key] or "") for key in keys if key in row_keys)


def _year_alignment_score(row, requested_years: set[str]) -> int:
    """Score how well a row's source/evidence years match the question."""
    if not requested_years:
        return 0

    period_text = _row_text(row, ("period",))
    source_text = _row_text(row, ("title", "url", "publication_date"))
    evidence_text = _row_text(row, ("evidence", "text"))

    period_years = _years_from_text(period_text)
    source_years = _years_from_text(source_text)
    evidence_years = _years_from_text(evidence_text)

    score = 0

    if period_years & requested_years:
        score += 6

    if source_years & requested_years:
        score += 5

    if evidence_years & requested_years:
        score += 2

    if source_years and not (source_years & requested_years):
        score -= 5

    if period_years and not (period_years & requested_years):
        score -= 3

    return score


def _fact_rank(row, requested_years: set[str]) -> tuple[int, int, int, int, int, int]:
    """Rank extracted facts for deterministic fallback answer quality."""
    return (
        _year_alignment_score(row, requested_years),
        CONFIDENCE_RANK.get((row["confidence"] or "").lower(), 0),
        SOURCE_TYPE_RANK.get(row["source_type"] or "", 0),
        1 if row["unit"] else 0,
        1 if row["period"] else 0,
        int(row["id"]),
    )


def _best_facts_by_metric(rows, question: str) -> list:
    """Return one best supported fact per metric, preserving a useful output order."""
    requested_years = _years_from_text(question)
    best_by_metric = OrderedDict()

    for row in rows:
        if (row["confidence"] or "").lower() == "unsupported":
            continue

        metric = row["metric"]
        existing = best_by_metric.get(metric)

        if existing is None or _fact_rank(row, requested_years) > _fact_rank(existing, requested_years):
            best_by_metric[metric] = row

    return list(best_by_metric.values())


def _evidence_id(prefix: str, row) -> str:
    """Return a stable evidence identifier for one answer-generation payload."""
    return f"{prefix}:{row['id']}"


def _payload_with_evidence_id(row, prefix: str) -> dict:
    """Convert a database row to LLM payload with an explicit citation handle."""
    payload = dict(row)
    payload["evidence_id"] = _evidence_id(prefix, row)
    payload["evidence_type"] = "fact" if prefix == "fact" else "chunk"

    return payload


def _citations_for_ids(value, evidence_by_id: dict[str, Citation]) -> list[Citation]:
    """Resolve LLM-returned citation IDs to known evidence citations."""
    citations = []
    seen = set()

    for citation_id in _string_list(value):
        if citation_id in seen:
            continue

        citation = evidence_by_id.get(citation_id)

        if citation is None:
            continue

        seen.add(citation_id)
        citations.append(citation)

    return citations


def answer_question(kb: KnowledgeBase, question: str, config: AppConfig) -> Answer:
    """Answer a question using persisted facts, evidence chunks, and optional LLM output."""
    logger.info("Answering question: %s", question)
    run_id = kb.start_run(None, config.redacted())

    kb.log_event(run_id, "ask_started", question)

    company = kb.company_for_question(question)
    company_id = int(company["id"]) if company else None

    if company:
        kb.log_event(run_id, "ask_company_filter", company["name"], {"company_id": company_id})

    facts = kb.facts_for_question(question, company_id=company_id)
    evidence_rows = kb.search_evidence(question, company_id=company_id)
    logger.info("Question evidence counts facts=%s chunks=%s", len(facts), len(evidence_rows))

    evidence_by_id: OrderedDict[str, Citation] = OrderedDict()
    fact_payload = []
    chunk_payload = []

    for row in facts[: config.qa.fact_evidence_limit]:
        evidence_id = _evidence_id("fact", row)
        evidence_by_id[evidence_id] = _citation_from_row(row)
        fact_payload.append(_payload_with_evidence_id(row, "fact"))

    for row in evidence_rows[: config.qa.chunk_evidence_limit]:
        evidence_id = _evidence_id("chunk", row)
        evidence_by_id[evidence_id] = _citation_from_row(row)
        chunk_payload.append(_payload_with_evidence_id(row, "chunk"))

    evidence_payload = fact_payload + chunk_payload

    logger.debug(
        "Prepared LLM evidence payload facts=%s chunks=%s",
        len(fact_payload),
        len(chunk_payload),
    )

    llm_payload = LLMClient(config.llm).complete_json(question, evidence_payload)
    llm_warnings = []

    if llm_payload:
        logger.debug("LLM response payload: %s", json.dumps(llm_payload, ensure_ascii=False))
        llm_warnings.extend(_string_list(llm_payload.get("warnings")))

    if llm_warnings:
        logger.warning("LLM returned warnings: %s", llm_warnings)

    citations_map: OrderedDict[str, Citation] = OrderedDict()
    answer_parts: list[str] = []
    fallback_facts = _best_facts_by_metric(facts, question)

    for row in fallback_facts:
        fact_text = (
            f"{row['metric']}: {row['value']}"
            f"{(' ' + row['unit']) if row['unit'] else ''}"
            f"{(' for ' + row['period']) if row['period'] else ''}"
            f"{(' at ' + row['entity_level'] + ' level') if row['entity_level'] else ''}"
        )

        answer_parts.append(fact_text)

        citation = _citation_from_row(row)
        citations_map.setdefault(citation.url, citation)

    if not answer_parts and evidence_rows:
        answer_parts.append(
            "The knowledge base contains potentially relevant evidence, but no structured fact was extracted for the requested metrics."
        )

        for row in evidence_rows[:4]:
            citation = _citation_from_row(row)
            citations_map.setdefault(citation.url, citation)

    if answer_parts and not _is_qualitative_question(question):
        fallback_answer_text = "; ".join(answer_parts)
        fallback_confidence = "medium" if fallback_facts else "low"
        fallback_limitations = []

        if not fallback_facts:
            fallback_limitations.append(
                "No structured metric/value fact matched the question; answer is based on retrieved chunks."
            )
        elif len(fallback_facts) < len(facts):
            fallback_limitations.append(
                "Duplicate or lower-confidence facts were suppressed from the deterministic fallback answer."
            )
    elif evidence_rows:
        fallback_answer_text = (
            "The knowledge base contains potentially relevant evidence, but mock mode cannot reliably "
            "answer qualitative yes/no questions. Use provider mode for a grounded natural-language answer."
        )
        fallback_confidence = "low"
        fallback_limitations = ["Mock mode does not synthesize qualitative answers from evidence chunks."]
    else:
        fallback_answer_text = "unsupported"
        fallback_confidence = "unsupported"
        fallback_limitations = ["No matching evidence was found in the persisted knowledge base."]

    llm_answer = llm_payload.get("answer") if llm_payload else None
    llm_confidence = _confidence(llm_payload.get("confidence") if llm_payload else None, fallback_confidence)
    llm_citations = _citations_for_ids(llm_payload.get("citation_ids") if llm_payload else None, evidence_by_id)
    requires_citations = bool(evidence_payload) and llm_confidence != "unsupported"

    if isinstance(llm_answer, str) and llm_answer.strip() and (llm_citations or not requires_citations):
        answer_text = llm_answer.strip()
        confidence = llm_confidence
        limitations = _string_list(llm_payload.get("limitations")) or fallback_limitations
        citations = llm_citations[:8] if llm_citations else list(citations_map.values())[:8]
        logger.info("Using LLM answer text for run_id=%s", run_id)
    else:
        answer_text = fallback_answer_text
        confidence = fallback_confidence
        limitations = fallback_limitations
        citations = list(citations_map.values())[:8]

        if llm_payload:
            if isinstance(llm_answer, str) and llm_answer.strip() and requires_citations:
                llm_warnings.append(
                    "LLM returned an answer without valid citation_ids; used deterministic fallback."
                )
            else:
                logger.warning("LLM payload did not contain a usable answer; using deterministic fallback")

    warnings = llm_warnings
    warnings.append("Answers are limited to evidence previously collected into the local knowledge base.")

    answer = Answer(
        answer=answer_text,
        confidence=confidence,
        citations=citations,
        warnings=warnings,
        limitations=limitations,
        audit_ref=run_id,
    )
    kb.store_answer(run_id, question, answer)
    kb.log_event(run_id, "ask_completed", "answer generated", {"confidence": confidence})
    kb.finish_run(run_id)

    logger.info(
        "Answered question run_id=%s confidence=%s citations=%s",
        run_id,
        confidence,
        len(answer.citations),
    )

    return answer


def answer_to_json(answer: Answer) -> str:
    """Serialize an Answer object as pretty JSON for CLI output."""
    return json.dumps(answer.to_json_dict(), indent=2, ensure_ascii=False)
