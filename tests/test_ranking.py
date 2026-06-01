from company_research.models import SearchResult
from company_research.ranking import score_source


def test_source_ranking_accepts_relevant_annual_report():
    result = SearchResult(
        query="Example Group 2024 annual report",
        url="https://example.com/investors/annual-report-2024.pdf",
        title="Example Group Annual Report 2024",
        snippet="FY2024 financial report",
    )
    decision = score_source(result, "Example Group")

    assert decision.accepted
    assert decision.source_type == "official_report"


def test_source_ranking_rejects_social_login():
    result = SearchResult(
        query="Example Group",
        url="https://facebook.com/login/example",
        title="Example Group",
        snippet="",
    )
    decision = score_source(result, "Example Group")

    assert not decision.accepted
    assert "outside assignment scope" in decision.reason or "login" in decision.reason
