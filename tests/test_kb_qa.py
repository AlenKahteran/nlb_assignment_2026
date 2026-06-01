import sqlite3
from pathlib import Path

from company_research.config import load_config
from company_research.extractors import extract_html
from company_research.facts import extract_facts_from_text
from company_research.kb import KnowledgeBase, _company_match_score, _fts_query_from_question
from company_research.models import FetchResult, SearchResult
from company_research.qa import answer_question
from company_research.ranking import score_source


def test_fts_query_strips_question_punctuation():
    query = _fts_query_from_question("What were total net operating income for NLB group in 2024?")

    assert query == '"What" OR "were" OR "total" OR "net" OR "operating" OR "income" OR "for" OR "NLB" OR "group" OR "2024"'
    assert "?" not in query


def test_company_match_detects_company_without_group_suffix():
    assert _company_match_score("NLB Group", "What was net profit for NLB group?") > 0
    assert _company_match_score("Petrol Group", "What was net profit for NLB group?") == 0


def test_existing_kb_schema_is_migrated_for_content_hash(tmp_path):
    kb_path = tmp_path / "kb"
    kb_path.mkdir()
    db_path = kb_path / "knowledge_base.sqlite3"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          source_id INTEGER NOT NULL,
          local_path TEXT,
          title TEXT,
          content_type TEXT,
          extractor TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    kb = KnowledgeBase(kb_path)
    columns = {row["name"] for row in kb.conn.execute("PRAGMA table_info(documents)").fetchall()}
    indexes = {
        row["name"] for row in kb.conn.execute("PRAGMA index_list(documents)").fetchall()
    }
    kb.close()

    assert "content_sha256" in columns
    assert "idx_documents_content_sha256" in indexes


def test_persisted_kb_answers_after_restart(tmp_path):
    kb_path = tmp_path / "kb"
    fixture = Path("tests/fixtures/company_report.html").resolve()
    fetch = FetchResult(
        url=fixture.as_uri(),
        status_code=200,
        content_type="text/html",
        final_url=fixture.as_uri(),
        body=fixture.read_bytes(),
        retrieved_at="2026-05-26T00:00:00+00:00",
        file_path=fixture,
    )
    result = SearchResult(
        query="Example Group 2024 annual report",
        url=fixture.as_uri(),
        title="Example Group FY2024 Annual Report",
        snippet="",
    )

    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    company_id = kb.upsert_company("Example Group")
    doc = extract_html(fetch)
    source_id = kb.upsert_source(run_id, company_id, result, score_source(result, "Example Group"), fetch, doc)
    document_id = kb.insert_document(source_id, fetch, doc, "html")
    kb.insert_facts(extract_facts_from_text(doc.text, source_id, document_id))
    kb.finish_run(run_id)
    kb.close()

    restarted = KnowledgeBase(kb_path)
    answer = answer_question(
        restarted,
        "What were total net operating income, profit after tax, total assets, and CET1 ratio?",
        load_config(env_file=None),
    )
    restarted.close()

    payload = answer.to_json_dict()
    assert payload["confidence"] in {"medium", "high"}
    assert "total net operating income" in payload["answer"]
    assert payload["citations"]
    assert payload["audit_ref"]


def test_answer_filters_facts_by_detected_company(tmp_path):
    kb_path = tmp_path / "kb"

    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    example_company_id = kb.upsert_company("Example Group")
    petrol_company_id = kb.upsert_company("Petrol Group")

    example_fetch = FetchResult(
        url="https://example.test/example",
        status_code=200,
        content_type="text/html",
        final_url="https://example.test/example",
        body=b"Example Group net profit or loss was EUR 456 million for 2024.",
        retrieved_at="2026-05-26T00:00:00+00:00",
    )
    example_result = SearchResult(
        query="Example Group 2024 annual report",
        url="https://example.test/example",
        title="Example Group FY2024 Annual Report",
        snippet="",
    )
    example_doc = extract_html(example_fetch)
    example_source_id = kb.upsert_source(
        run_id,
        example_company_id,
        example_result,
        score_source(example_result, "Example Group"),
        example_fetch,
        example_doc,
    )
    example_document_id = kb.insert_document(example_source_id, example_fetch, example_doc, "html")
    kb.insert_facts(extract_facts_from_text(example_doc.text, example_source_id, example_document_id))

    petrol_fetch = FetchResult(
        url="https://example.test/petrol",
        status_code=200,
        content_type="text/html",
        final_url="https://example.test/petrol",
        body=b"Petrol Group net profit or loss was EUR 999 million for 2024.",
        retrieved_at="2026-05-26T00:00:00+00:00",
    )
    petrol_result = SearchResult(
        query="Petrol Group 2024 annual report",
        url="https://example.test/petrol",
        title="Petrol Group FY2024 Annual Report",
        snippet="",
    )
    petrol_doc = extract_html(petrol_fetch)
    petrol_source_id = kb.upsert_source(
        run_id,
        petrol_company_id,
        petrol_result,
        score_source(petrol_result, "Petrol Group"),
        petrol_fetch,
        petrol_doc,
    )
    petrol_document_id = kb.insert_document(petrol_source_id, petrol_fetch, petrol_doc, "html")
    kb.insert_facts(extract_facts_from_text(petrol_doc.text, petrol_source_id, petrol_document_id))
    kb.finish_run(run_id)

    answer = answer_question(kb, "What was net profit or loss for Example Group in 2024?", load_config(env_file=None))
    kb.close()

    payload = answer.to_json_dict()
    assert "456" in payload["answer"]
    assert "999" not in payload["answer"]
    assert payload["citations"]
    assert all("petrol" not in citation["url"] for citation in payload["citations"])


def test_mock_qualitative_question_does_not_dump_numeric_facts(tmp_path):
    kb_path = tmp_path / "kb"

    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    company_id = kb.upsert_company("Example Group")

    fetch = FetchResult(
        url="https://example.test/example",
        status_code=200,
        content_type="text/html",
        final_url="https://example.test/example",
        body=(
            b"Example Group provides banking services. "
            b"It finances energy efficiency projects for corporate clients. "
            b"Profit after tax was EUR 456 million for 2024."
        ),
        retrieved_at="2026-05-26T00:00:00+00:00",
    )
    result = SearchResult(
        query="Example Group 2024 annual report",
        url="https://example.test/example",
        title="Example Group FY2024 Annual Report",
        snippet="",
    )
    doc = extract_html(fetch)
    source_id = kb.upsert_source(
        run_id,
        company_id,
        result,
        score_source(result, "Example Group"),
        fetch,
        doc,
    )
    document_id = kb.insert_document(source_id, fetch, doc, "html")
    kb.insert_facts(extract_facts_from_text(doc.text, source_id, document_id))
    kb.finish_run(run_id)

    answer = answer_question(kb, "Does Example Group do business in energy sector?", load_config(env_file=None))
    kb.close()

    payload = answer.to_json_dict()
    assert "mock mode cannot reliably answer qualitative yes/no questions" in payload["answer"].lower()
    assert "profit after tax: 456" not in payload["answer"]


def test_deterministic_answer_suppresses_duplicate_metric_facts(tmp_path):
    kb_path = tmp_path / "kb"
    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    company_id = kb.upsert_company("Example Group")

    for index, value in enumerate(("456", "457"), start=1):
        fetch = FetchResult(
            url=f"https://example.test/report-{index}",
            status_code=200,
            content_type="text/html",
            final_url=f"https://example.test/report-{index}",
            body=f"Example Group profit after tax was EUR {value} million for 2024.".encode(),
            retrieved_at="2026-05-26T00:00:00+00:00",
        )
        result = SearchResult(
            query="Example Group 2024 annual report",
            url=f"https://example.test/report-{index}",
            title="Example Group Annual Report 2024",
            snippet="",
        )
        doc = extract_html(fetch)
        source_id = kb.upsert_source(run_id, company_id, result, score_source(result, "Example Group"), fetch, doc)
        document_id = kb.insert_document(source_id, fetch, doc, "html")
        kb.insert_facts(extract_facts_from_text(doc.text, source_id, document_id))

    kb.finish_run(run_id)
    answer = answer_question(kb, "What was profit after tax for Example Group in 2024?", load_config(env_file=None))
    kb.close()

    payload = answer.to_json_dict()
    assert payload["answer"].count("profit after tax:") == 1
    assert "lower-confidence facts were suppressed" in " ".join(payload["limitations"])


def test_deterministic_answer_prefers_official_report_over_news(tmp_path):
    kb_path = tmp_path / "kb"
    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    company_id = kb.upsert_company("Example Group")

    sources = [
        (
            "https://reuters.com/markets/example-group-update",
            "Reuters market update",
            b"<html><head><title>Reuters market update</title></head><body>"
            b"Example Group profit after tax was EUR 111 million for 2024."
            b"</body></html>",
        ),
        (
            "https://example.test/investors/annual-report-2024",
            "Example Group Annual Report 2024",
            b"<html><head><title>Example Group Annual Report 2024</title></head><body>"
            b"Example Group profit after tax was EUR 222 million for 2024."
            b"</body></html>",
        ),
    ]

    for url, title, body in sources:
        fetch = FetchResult(
            url=url,
            status_code=200,
            content_type="text/html",
            final_url=url,
            body=body,
            retrieved_at="2026-05-26T00:00:00+00:00",
        )
        result = SearchResult(
            query="Example Group 2024 annual report",
            url=url,
            title=title,
            snippet="",
        )
        doc = extract_html(fetch)
        source_id = kb.upsert_source(run_id, company_id, result, score_source(result, "Example Group"), fetch, doc)
        document_id = kb.insert_document(source_id, fetch, doc, "html")
        kb.insert_facts(extract_facts_from_text(doc.text, source_id, document_id))

    kb.finish_run(run_id)
    answer = answer_question(kb, "What was profit after tax for Example Group in 2024?", load_config(env_file=None))
    kb.close()

    payload = answer.to_json_dict()
    assert "222" in payload["answer"]
    assert "111" not in payload["answer"]
    assert payload["citations"][0]["source_type"] == "official_report"


def test_deterministic_answer_prefers_requested_year_source(tmp_path):
    kb_path = tmp_path / "kb"
    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    company_id = kb.upsert_company("Example Group")

    sources = [
        (
            "https://example.test/investors/annual-report-2024",
            "Example Group Annual Report 2024",
            b"<html><head><title>Example Group Annual Report 2024</title></head><body>"
            b"Example Group profit after tax was EUR 222 million for 2024."
            b"</body></html>",
        ),
        (
            "https://example.test/investors/annual-report-2021",
            "Example Group Annual Report 2021",
            b"<html><head><title>Example Group Annual Report 2021</title></head><body>"
            b"The audit committee reviewed the 2022-2024 Annual Report audit tender. "
            b"Example Group profit after tax was EUR 111 million for 2021."
            b"</body></html>",
        ),
    ]

    for url, title, body in sources:
        fetch = FetchResult(
            url=url,
            status_code=200,
            content_type="text/html",
            final_url=url,
            body=body,
            retrieved_at="2026-05-26T00:00:00+00:00",
        )
        result = SearchResult(
            query="Example Group 2024 annual report",
            url=url,
            title=title,
            snippet="",
        )
        doc = extract_html(fetch)
        source_id = kb.upsert_source(run_id, company_id, result, score_source(result, "Example Group"), fetch, doc)
        document_id = kb.insert_document(source_id, fetch, doc, "html")
        kb.insert_facts(extract_facts_from_text(doc.text, source_id, document_id))

    kb.finish_run(run_id)
    answer = answer_question(kb, "What was profit after tax for Example Group in 2024?", load_config(env_file=None))
    kb.close()

    payload = answer.to_json_dict()
    assert "222" in payload["answer"]
    assert "111" not in payload["answer"]
    assert "2024" in payload["citations"][0]["url"]


def test_answer_uses_llm_payload_when_available(tmp_path, monkeypatch):
    kb_path = tmp_path / "kb"
    fixture = Path("tests/fixtures/company_report.html").resolve()
    fetch = FetchResult(
        url=fixture.as_uri(),
        status_code=200,
        content_type="text/html",
        final_url=fixture.as_uri(),
        body=fixture.read_bytes(),
        retrieved_at="2026-05-26T00:00:00+00:00",
        file_path=fixture,
    )
    result = SearchResult(
        query="Example Group 2024 annual report",
        url=fixture.as_uri(),
        title="Example Group FY2024 Annual Report",
        snippet="",
    )

    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    company_id = kb.upsert_company("Example Group")
    doc = extract_html(fetch)
    source_id = kb.upsert_source(run_id, company_id, result, score_source(result, "Example Group"), fetch, doc)
    document_id = kb.insert_document(source_id, fetch, doc, "html")
    kb.insert_facts(extract_facts_from_text(doc.text, source_id, document_id))
    kb.finish_run(run_id)

    def fake_complete_json(_self, _question, evidence):
        assert evidence
        assert evidence[0]["evidence_id"].startswith("fact:")
        return {
            "answer": "LLM-composed answer from collected evidence.",
            "confidence": "high",
            "citation_ids": [evidence[0]["evidence_id"]],
            "warnings": [],
            "limitations": ["Synthetic test limitation."],
        }

    monkeypatch.setattr("company_research.qa.LLMClient.complete_json", fake_complete_json)
    answer = answer_question(kb, "What did the company report?", load_config(env_file=None))
    kb.close()

    payload = answer.to_json_dict()
    assert payload["answer"] == "LLM-composed answer from collected evidence."
    assert payload["confidence"] == "high"
    assert payload["limitations"] == ["Synthetic test limitation."]
    assert payload["citations"]
    assert payload["citations"][0]["url"] == fixture.as_uri()


def test_answer_rejects_llm_answer_without_valid_citation_ids(tmp_path, monkeypatch):
    kb_path = tmp_path / "kb"
    fixture = Path("tests/fixtures/company_report.html").resolve()
    fetch = FetchResult(
        url=fixture.as_uri(),
        status_code=200,
        content_type="text/html",
        final_url=fixture.as_uri(),
        body=fixture.read_bytes(),
        retrieved_at="2026-05-26T00:00:00+00:00",
        file_path=fixture,
    )
    result = SearchResult(
        query="Example Group 2024 annual report",
        url=fixture.as_uri(),
        title="Example Group FY2024 Annual Report",
        snippet="",
    )

    kb = KnowledgeBase(kb_path)
    run_id = kb.start_run("Example Group", {})
    company_id = kb.upsert_company("Example Group")
    doc = extract_html(fetch)
    source_id = kb.upsert_source(run_id, company_id, result, score_source(result, "Example Group"), fetch, doc)
    document_id = kb.insert_document(source_id, fetch, doc, "html")
    kb.insert_facts(extract_facts_from_text(doc.text, source_id, document_id))
    kb.finish_run(run_id)

    def fake_complete_json(_self, _question, evidence):
        assert evidence
        return {
            "answer": "Ungrounded LLM answer.",
            "confidence": "high",
            "citation_ids": ["fact:999999"],
            "warnings": [],
            "limitations": [],
        }

    monkeypatch.setattr("company_research.qa.LLMClient.complete_json", fake_complete_json)
    answer = answer_question(kb, "What did Example Group report in 2024?", load_config(env_file=None))
    kb.close()

    payload = answer.to_json_dict()
    assert payload["answer"] != "Ungrounded LLM answer."
    assert "without valid citation_ids" in " ".join(payload["warnings"])
    assert payload["citations"]
