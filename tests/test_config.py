from company_research.config import load_config


def test_config_defaults_and_redaction(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-search")
    monkeypatch.setenv("LLM_API_KEY", "secret-llm")
    config = load_config(env_file=None)

    redacted = config.redacted()

    assert redacted["brave_search_api_key"] == "<redacted>"
    assert redacted["llm"]["api_key"] == "<redacted>"
    assert config.budget.max_search_results == 100
    assert config.budget.max_visited_pages == 50
    assert config.budget.max_downloads == 16
    assert config.budget.max_depth == 10
    assert config.qa.fact_evidence_limit == 25
    assert config.qa.chunk_evidence_limit == 25
    assert config.llm.timeout_seconds == 60
