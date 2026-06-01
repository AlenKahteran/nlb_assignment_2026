from pathlib import Path

from company_research.extractors import extract_csv, extract_html
from company_research.facts import evaluate_fact_confidence, extract_facts_from_text
from company_research.models import FetchResult


ALLOWED_CONFIDENCE = {"high", "medium", "low", "unsupported"}


def _fetch(path: Path, content_type: str) -> FetchResult:
    return FetchResult(
        url=path.as_uri(),
        status_code=200,
        content_type=content_type,
        final_url=path.as_uri(),
        body=path.read_bytes(),
        retrieved_at="2026-05-26T00:00:00+00:00",
        file_path=path,
    )


def test_html_extraction_and_fact_detection():
    fetch = _fetch(Path("tests/fixtures/company_report.html").resolve(), "text/html")
    doc = extract_html(fetch)
    facts = extract_facts_from_text(doc.text, source_id=1, document_id=1)

    assert doc.title == "Example Group FY2024 Annual Report"
    assert any("malicious page content" in chunk for chunk in doc.chunks)
    assert {fact.metric for fact in facts} >= {
        "total net operating income",
        "profit after tax",
        "total assets",
        "cet1 ratio",
    }
    assert {fact.confidence for fact in facts} <= ALLOWED_CONFIDENCE
    assert any(fact.confidence == "high" for fact in facts)


def test_csv_extraction():
    fetch = _fetch(Path("tests/fixtures/metrics.csv").resolve(), "text/csv")
    doc = extract_csv(fetch)

    assert "sales revenue" in doc.text
    assert doc.metadata["rows"] == 4


def test_fact_evidence_window_can_exceed_value_window():
    text = (
        "Introductory context. "
        "For FY2024, Example Group reported total net operating income of EUR 1,234 million. "
        + ("Additional supporting context. " * 40)
    )
    facts = extract_facts_from_text(text, source_id=1, document_id=1)

    fact = next(fact for fact in facts if fact.metric == "total net operating income")

    assert fact.value == "1,234"
    assert len(fact.evidence) > 500


def test_energy_efficiency_mentions_do_not_create_energy_sales_fact():
    text = (
        "NLB Group supports sustainable financing, particularly solar power plants and energy efficiency. "
        "The banking members reported net interest margin between 3.03% and 4.75%."
    )
    facts = extract_facts_from_text(text, source_id=1, document_id=1)

    assert "electricity/energy sales or trading volume" not in {fact.metric for fact in facts}


def test_fact_extraction_skips_years_and_incompatible_units():
    text = (
        "Example Group net profit or loss for 2024 was EUR 456 million. "
        "Example Group total assets ratio was 34.8%, while total assets were EUR 28,154 million. "
        "The Group CET1 ratio for 2024 was 16.2%."
    )
    facts = extract_facts_from_text(text, source_id=1, document_id=1)
    by_metric = {fact.metric: fact for fact in facts}

    assert by_metric["net profit or loss"].value == "456"
    assert by_metric["total assets"].value == "28,154"
    assert by_metric["total assets"].unit == "million"
    assert by_metric["cet1 ratio"].value == "16.2"
    assert by_metric["cet1 ratio"].unit == "%"


def test_fact_extraction_handles_billion_mio_and_negative_values():
    text = (
        "For FY2024, Example Group sales revenue reached EUR 1.2 billion. "
        "Adjusted gross profit amounted to 456 mio. "
        "Net loss was EUR -12.4 million for 2024."
    )
    facts = extract_facts_from_text(text, source_id=1, document_id=1)
    by_metric = {fact.metric: fact for fact in facts}

    assert by_metric["sales revenue"].value == "1.2"
    assert by_metric["sales revenue"].unit == "billion"
    assert by_metric["adjusted gross profit"].value == "456"
    assert by_metric["adjusted gross profit"].unit == "mio"
    assert by_metric["net profit or loss"].value == "-12.4"
    assert by_metric["net profit or loss"].unit == "million"


def test_fact_extraction_requires_units_for_energy_volume():
    text = (
        "Example Group reported electricity sales volume of 1,234 GWh for FY2024. "
        "The unrelated sales volume of retail products was 999 units."
    )
    facts = extract_facts_from_text(text, source_id=1, document_id=1)
    energy_facts = [fact for fact in facts if fact.metric == "electricity/energy sales or trading volume"]

    assert energy_facts
    assert {fact.value for fact in energy_facts} == {"1,234"}
    assert {fact.unit for fact in energy_facts} == {"GWh"}


def test_cet1_does_not_use_nearby_money_amount():
    text = (
        "The Group CET1 ratio was discussed next to capital of EUR 1,234 million. "
        "The CET1 ratio was 16.2% for the reporting period 2024."
    )
    facts = extract_facts_from_text(text, source_id=1, document_id=1)
    cet1_facts = [fact for fact in facts if fact.metric == "cet1 ratio"]

    assert cet1_facts
    assert cet1_facts[0].value == "16.2"
    assert cet1_facts[0].unit == "%"


def test_fact_confidence_evaluation_levels():
    assert (
        evaluate_fact_confidence(
            metric="total net operating income",
            value="1,234",
            unit="million",
            period="FY2024",
            entity_level="group",
            evidence="For FY2024, Example Group reported total net operating income of EUR 1,234 million.",
        )
        == "high"
    )

    assert (
        evaluate_fact_confidence(
            metric="profit after tax",
            value="456",
            unit="",
            period="2024",
            entity_level="unspecified",
            evidence="Profit after tax was 456 for the reporting period 2024.",
        )
        == "medium"
    )

    assert (
        evaluate_fact_confidence(
            metric="profit after tax",
            value="456",
            unit="",
            period="",
            entity_level="unspecified",
            evidence="Reported amount was 456.",
        )
        == "low"
    )

    assert (
        evaluate_fact_confidence(
            metric="profit after tax",
            value="",
            unit="",
            period="",
            entity_level="unspecified",
            evidence="Profit after tax is mentioned without a value.",
        )
        == "unsupported"
    )
