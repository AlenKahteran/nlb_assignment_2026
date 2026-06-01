from __future__ import annotations

import logging
import re
from typing import Iterable, Literal

from .models import ExtractedFact

Confidence = Literal["high", "medium", "low", "unsupported"]

# extraction facts for the assignment

METRIC_PATTERNS = {
    "total net operating income": r"total net operating income",
    "profit after tax": r"profit after tax|net profit after tax",
    "total assets": r"total assets",
    "cet1 ratio": r"cet1 ratio|common equity tier 1",
    "sales revenue": r"sales revenue|revenue from sales|revenue",
    "adjusted gross profit": r"adjusted gross profit|gross profit",
    "ebitda": r"ebitda",
    "net profit or loss": r"net profit(?: or loss)?|net loss|profit or loss",
    "electricity/energy sales or trading volume": (
        r"electricity\s+(?:sales|trading|volume)"
        r"|energy\s+(?:sales|trading|volume)"
        r"|(?:electricity|energy)\s+sales\s+volume"
        r"|trading\s+volume"
        r"|sales\s+volume"
    ),
}

logger = logging.getLogger(__name__)

VALUE_WINDOW_CHARS = 4096
EVIDENCE_WINDOW_CHARS = 4096

VALUE_PATTERN = re.compile(
    r"(?<![\d.,])(?P<prefix_unit>eur|€)?\s*"
    r"(?P<value>-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?)\s*"
    r"(?P<suffix_unit>million|mio|bn|billion|eur|€|%|gwh|twh|mwh|kwh)?(?![\d.,])",
    re.IGNORECASE,
)

EXPECTED_UNITS = {
    "cet1 ratio": {"%"},
    "total net operating income": {"eur", "€", "million", "mio", "bn", "billion"},
    "profit after tax": {"eur", "€", "million", "mio", "bn", "billion"},
    "total assets": {"eur", "€", "million", "mio", "bn", "billion"},
    "sales revenue": {"eur", "€", "million", "mio", "bn", "billion"},
    "adjusted gross profit": {"eur", "€", "million", "mio", "bn", "billion"},
    "ebitda": {"eur", "€", "million", "mio", "bn", "billion"},
    "net profit or loss": {"eur", "€", "million", "mio", "bn", "billion"},
    "electricity/energy sales or trading volume": {"gwh", "twh", "mwh", "kwh"},
}

VALUE_CONNECTOR_PATTERN = re.compile(
    r"\b(reported|was|were|is|are|of|at|amounted|stood|reached|totaled|totalled)\b",
    re.IGNORECASE,
)
UNIT_AFTER_VALUE_PATTERN = re.compile(r"^(%|€|(?:million|mio|bn|billion|eur|gwh|twh|mwh|kwh)\b)", re.IGNORECASE)


def _normalize_value(value: str) -> str:
    """Normalize a numeric value for rough evidence matching."""
    return value.replace(",", "").replace(".", "").strip()


def _value_is_in_evidence(value: str, evidence: str) -> bool:
    """Return whether the extracted value can be found in the evidence text."""
    normalized_value = _normalize_value(value)
    normalized_evidence = _normalize_value(evidence)

    return bool(normalized_value) and normalized_value in normalized_evidence


def _metric_is_in_evidence(metric: str, evidence: str) -> bool:
    """Return whether the metric phrase appears in the evidence text."""
    metric_tokens = [token for token in re.split(r"\W+", metric.lower()) if len(token) > 2]
    evidence_lower = evidence.lower()

    if metric.lower() in evidence_lower:
        return True

    if not metric_tokens:
        return False

    matched_tokens = sum(1 for token in metric_tokens if token in evidence_lower)

    return matched_tokens >= max(1, len(metric_tokens) - 1)


def _metric_value_distance(metric: str, value: str, evidence: str) -> int | None:
    """Estimate character distance between a metric mention and a value in evidence."""
    metric_match = re.search(re.escape(metric), evidence, re.IGNORECASE)
    value_match = re.search(re.escape(value), evidence)

    if not metric_match or not value_match:
        return None

    return abs(metric_match.start() - value_match.start())


def evaluate_fact_confidence(
    *,
    metric: str,
    value: str,
    unit: str,
    period: str,
    entity_level: str,
    evidence: str,
) -> Confidence:
    """Evaluate confidence for an extracted fact using evidence quality signals."""
    if not metric or not value or not evidence:
        return "unsupported"

    score = 0
    normalized_unit = unit.lower()

    metric_found = _metric_is_in_evidence(metric, evidence)
    value_found = _value_is_in_evidence(value, evidence)

    if metric_found:
        score += 2

    if value_found:
        score += 2
    else:
        score -= 2

    if normalized_unit:
        score += 1

        expected_units = EXPECTED_UNITS.get(metric.lower(), set())
        if not expected_units or normalized_unit in expected_units:
            score += 1

    if period:
        score += 1

    if entity_level in {"group", "company"}:
        score += 1

    distance = _metric_value_distance(metric, value, evidence)

    if distance is not None and distance <= 180:
        score += 1

        between_start = min(
            re.search(re.escape(metric), evidence, re.IGNORECASE).end(),
            re.search(re.escape(value), evidence).end(),
        )
        between_end = max(
            re.search(re.escape(metric), evidence, re.IGNORECASE).start(),
            re.search(re.escape(value), evidence).start(),
        )
        nearby_text = evidence[between_start:between_end]

        if VALUE_CONNECTOR_PATTERN.search(nearby_text):
            score += 1

    if "?" in evidence:
        score -= 1

    if not metric_found and not value_found:
        return "unsupported"

    if score >= 8:
        return "high"

    if score >= 5:
        return "medium"

    if score >= 2:
        return "low"

    return "unsupported"


def _window(text: str, start: int, size: int) -> str:
    """Return a normalized text window around a character offset."""
    left = max(0, start - size // 2)
    right = min(len(text), start + size // 2)

    return re.sub(r"\s+", " ", text[left:right]).strip()


def _numeric_value(value: str) -> float | None:
    """Return a rough numeric value for filtering implausible candidates."""
    cleaned = value.replace(",", "").strip()

    if cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _looks_like_year(value: str, unit: str) -> bool:
    """Return whether a bare value is likely a reporting year rather than a metric value."""
    if unit:
        return False

    numeric = _numeric_value(value)

    return numeric is not None and numeric.is_integer() and 1900 <= numeric <= 2100


def _unit_is_plausible(metric: str, unit: str) -> bool:
    """Return whether a candidate unit is compatible with the metric being extracted."""
    if not unit:
        return metric.lower() not in {"cet1 ratio", "electricity/energy sales or trading volume"}

    normalized_unit = unit.lower()
    expected_units = EXPECTED_UNITS.get(metric.lower(), set())

    return not expected_units or normalized_unit in expected_units


def _extract_value(snippet: str, metric: str) -> tuple[str, str]:
    """Extract the first plausible numeric value and optional unit from a text snippet."""
    for match in VALUE_PATTERN.finditer(snippet):
        value = match.group("value")
        suffix_unit = match.group("suffix_unit")

        if not suffix_unit:
            tail = snippet[match.end() : match.end() + 20].lstrip()
            suffix_match = UNIT_AFTER_VALUE_PATTERN.search(tail)
            suffix_unit = suffix_match.group(1) if suffix_match else ""

        unit = suffix_unit or match.group("prefix_unit") or ""

        if _looks_like_year(value, unit):
            continue

        if not _unit_is_plausible(metric, unit):
            continue

        return value, unit

    return "", ""


def _period_for_text(text: str) -> str:
    """Find a FY2024-like reporting period in nearby evidence text."""
    period_match = re.search(r"(FY\s?2024|financial year 2024|year ended [^.]{0,40}2024|2024)", text, re.I)
    return period_match.group(1) if period_match else ""


def _entity_level_for_text(text: str) -> str:
    """Infer whether evidence appears to refer to group, company, or unspecified level."""
    lowered = text.lower()

    if "group" in lowered:
        return "group"

    if "company" in lowered:
        return "company"

    return "unspecified"


def extract_facts_from_text(
    text: str,
    source_id: int,
    document_id: int,
    *,
    max_per_metric: int = 10,
) -> list[ExtractedFact]:
    """Extract configured financial/operational metrics from normalized document text."""
    logger.info(
        "Extracting facts from document_id=%s source_id=%s text_chars=%s",
        document_id,
        source_id,
        len(text),
    )
    facts: list[ExtractedFact] = []
    lowered = text.lower()

    for metric, pattern in METRIC_PATTERNS.items():
        count = 0

        for match in re.finditer(pattern, lowered, flags=re.I):
            value_snippet = _window(text, match.start(), VALUE_WINDOW_CHARS)
            evidence = _window(text, match.start(), EVIDENCE_WINDOW_CHARS)

            post_metric_text = text[match.start() : match.start() + VALUE_WINDOW_CHARS]
            value, unit = _extract_value(post_metric_text, metric)

            if not value:
                value, unit = _extract_value(value_snippet, metric)

            if not value:
                continue

            period = _period_for_text(evidence)
            entity_level = _entity_level_for_text(evidence)
            confidence = evaluate_fact_confidence(
                metric=metric,
                value=value,
                unit=unit,
                period=period,
                entity_level=entity_level,
                evidence=evidence,
            )

            if confidence == "unsupported":
                logger.debug(
                    "Skipping unsupported fact metric=%s value=%s unit=%s source_id=%s document_id=%s",
                    metric,
                    value,
                    unit,
                    source_id,
                    document_id,
                )
                continue

            logger.debug(
                "Extracted fact metric=%s value=%s unit=%s confidence=%s source_id=%s document_id=%s",
                metric,
                value,
                unit,
                confidence,
                source_id,
                document_id,
            )
            facts.append(
                ExtractedFact(
                    metric=metric,
                    value=value,
                    unit=unit,
                    period=period,
                    entity_level=entity_level,
                    evidence=evidence,
                    source_id=source_id,
                    document_id=document_id,
                    confidence=confidence,
                )
            )
            count += 1

            if count >= max_per_metric:
                logger.debug("Reached max_per_metric=%s for metric=%s", max_per_metric, metric)
                break

    logger.info("Extracted %s facts from document_id=%s", len(facts), document_id)

    return facts


def dedupe_facts(facts: Iterable[ExtractedFact]) -> list[ExtractedFact]:
    """Remove duplicate facts by metric, value, period, and source."""
    seen: set[tuple[str, str, str, int]] = set()
    deduped: list[ExtractedFact] = []
    original_count = 0

    for fact in facts:
        original_count += 1
        key = (fact.metric.lower(), fact.value, fact.period, fact.source_id)

        if key in seen:
            logger.debug("Dropping duplicate fact metric=%s value=%s period=%s", *key[:3])
            continue

        seen.add(key)
        deduped.append(fact)

    logger.info("Deduped facts from %s to %s", original_count, len(deduped))

    return deduped
