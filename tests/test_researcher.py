import json

from company_research.kb import KnowledgeBase
from company_research.models import AppConfig, Budget, LLMConfig
from company_research.researcher import research_company


def _write_fixture_search(fixture_dir, root):
    search_fixture = fixture_dir / "search.json"
    search_fixture.write_text(
        json.dumps(
            {
                '"Example Group" 2024 annual report financial report': [
                    {
                        "path": root.name,
                        "title": "Example Group Annual Report 2024",
                        "snippet": "Official annual report.",
                        "publisher": "Example Group",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return search_fixture


def _fixture_config(search_fixture, *, max_depth, max_visited_pages=8, max_search_results=1):
    return AppConfig(
        search_fixture_path=str(search_fixture),
        budget=Budget(
            max_search_results=max_search_results,
            max_visited_pages=max_visited_pages,
            max_downloads=8,
            max_depth=max_depth,
            crawl_delay_seconds=0,
        ),
        llm=LLMConfig(mode="mock"),
    )


def _document_count(kb):
    return kb.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def test_research_follows_accepted_links_within_depth(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    root = fixture_dir / "root.html"
    detail = fixture_dir / "detail.html"

    root.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024</title></head>
          <body><a href="detail.html">Full annual report PDF</a></body>
        </html>
        """,
        encoding="utf-8",
    )
    detail.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024 Detail</title></head>
          <body>Example Group profit after tax was EUR 456 million for 2024.</body>
        </html>
        """,
        encoding="utf-8",
    )

    config = _fixture_config(_write_fixture_search(fixture_dir, root), max_depth=1)
    kb = KnowledgeBase(tmp_path / "kb")

    research_company(kb, "Example Group", config)

    facts = kb.conn.execute("SELECT metric, value FROM extracted_facts").fetchall()
    documents = _document_count(kb)
    kb.close()

    assert documents == 2
    assert ("profit after tax", "456") in [tuple(row) for row in facts]


def test_research_does_not_follow_links_when_depth_is_zero(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    root = fixture_dir / "root.html"
    detail = fixture_dir / "detail.html"

    root.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024</title></head>
          <body><a href="detail.html">Full annual report PDF</a></body>
        </html>
        """,
        encoding="utf-8",
    )
    detail.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024 Detail</title></head>
          <body>Example Group profit after tax was EUR 456 million for 2024.</body>
        </html>
        """,
        encoding="utf-8",
    )

    kb = KnowledgeBase(tmp_path / "kb")
    research_company(kb, "Example Group", _fixture_config(_write_fixture_search(fixture_dir, root), max_depth=0))

    documents = _document_count(kb)
    facts = kb.conn.execute("SELECT COUNT(*) FROM extracted_facts").fetchone()[0]
    kb.close()

    assert documents == 1
    assert facts == 0


def test_research_deduplicates_discovered_links(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    root = fixture_dir / "root.html"
    detail = fixture_dir / "detail.html"

    root.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024</title></head>
          <body>
            <a href="detail.html">Full annual report PDF</a>
            <a href="detail.html">Duplicate full annual report PDF</a>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    detail.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024 Detail</title></head>
          <body>Example Group profit after tax was EUR 456 million for 2024.</body>
        </html>
        """,
        encoding="utf-8",
    )

    kb = KnowledgeBase(tmp_path / "kb")
    research_company(kb, "Example Group", _fixture_config(_write_fixture_search(fixture_dir, root), max_depth=1))

    documents = _document_count(kb)
    sources = kb.conn.execute("SELECT COUNT(*) FROM sources WHERE url LIKE ?", ("%detail.html%",)).fetchone()[0]
    kb.close()

    assert documents == 2
    assert sources == 1


def test_research_respects_visited_page_budget_for_links(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    root = fixture_dir / "root.html"
    detail = fixture_dir / "detail.html"

    root.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024</title></head>
          <body><a href="detail.html">Full annual report PDF</a></body>
        </html>
        """,
        encoding="utf-8",
    )
    detail.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024 Detail</title></head>
          <body>Example Group profit after tax was EUR 456 million for 2024.</body>
        </html>
        """,
        encoding="utf-8",
    )

    kb = KnowledgeBase(tmp_path / "kb")
    research_company(
        kb,
        "Example Group",
        _fixture_config(_write_fixture_search(fixture_dir, root), max_depth=1, max_visited_pages=1),
    )

    documents = _document_count(kb)
    kb.close()

    assert documents == 1


def test_research_skips_url_already_parsed_in_kb(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    report = fixture_dir / "report.html"

    report.write_text(
        """
        <html>
          <head><title>Example Group Annual Report 2024</title></head>
          <body>Example Group profit after tax was EUR 456 million for 2024.</body>
        </html>
        """,
        encoding="utf-8",
    )

    kb = KnowledgeBase(tmp_path / "kb")
    config = _fixture_config(_write_fixture_search(fixture_dir, report), max_depth=0)

    research_company(kb, "Example Group", config)
    research_company(kb, "Example Group", config)

    documents = _document_count(kb)
    facts = kb.conn.execute("SELECT COUNT(*) FROM extracted_facts").fetchone()[0]
    skipped = kb.conn.execute(
        "SELECT COUNT(*) FROM audit_events WHERE event_type = ? AND message = ?",
        ("source_skipped", "already parsed source skipped"),
    ).fetchone()[0]
    kb.close()

    assert documents == 1
    assert facts == 1
    assert skipped >= 1


def test_research_skips_duplicate_document_content_in_kb(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    first = fixture_dir / "report-a.html"
    second = fixture_dir / "report-b.html"
    html = """
    <html>
      <head><title>Example Group Annual Report 2024</title></head>
      <body>Example Group profit after tax was EUR 456 million for 2024.</body>
    </html>
    """

    first.write_text(html, encoding="utf-8")
    second.write_text(html, encoding="utf-8")

    search_fixture = fixture_dir / "search.json"
    search_fixture.write_text(
        json.dumps(
            {
                '"Example Group" 2024 annual report financial report': [
                    {
                        "path": first.name,
                        "title": "Example Group Annual Report 2024",
                        "snippet": "Official annual report.",
                        "publisher": "Example Group",
                    },
                    {
                        "path": second.name,
                        "title": "Example Group Annual Report 2024 mirror",
                        "snippet": "Official annual report.",
                        "publisher": "Example Group",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    kb = KnowledgeBase(tmp_path / "kb")
    research_company(
        kb,
        "Example Group",
        _fixture_config(search_fixture, max_depth=0, max_search_results=2),
    )

    documents = _document_count(kb)
    sources = kb.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    skipped = kb.conn.execute(
        "SELECT COUNT(*) FROM audit_events WHERE event_type = ? AND message = ?",
        ("source_skipped", "duplicate document content skipped"),
    ).fetchone()[0]
    kb.close()

    assert documents == 1
    assert sources == 2
    assert skipped == 1
