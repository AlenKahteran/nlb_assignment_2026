from company_research.llm import LLMClient
from company_research.models import LLMConfig


def test_unsupported_llm_provider_uses_fallback_warning():
    client = LLMClient(
        LLMConfig(
            provider="custom-provider",
            endpoint="https://llm.example.test/v1/chat/completions",
            api_key="secret",
            model_name="example-model",
        )
    )

    payload = client.complete_json("What happened?", [])

    assert payload is not None
    assert "Unsupported LLM_PROVIDER" in payload["warnings"][0]


def test_llm_malformed_json_response_falls_back_with_warning(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "not json"}}]}

    class FakeHttpx:
        @staticmethod
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeResponse()

    monkeypatch.setattr("company_research.llm.httpx", FakeHttpx)
    client = LLMClient(
        LLMConfig(
            provider="openai-compatible",
            endpoint="https://llm.example.test/v1/chat/completions",
            api_key="secret",
            model_name="example-model",
            max_retries=1,
        )
    )

    payload = client.complete_json("What happened?", [{"evidence": "Example Group profit after tax was 456."}])

    assert payload is not None
    assert len(calls) == 2
    assert "LLM provider failed" in payload["warnings"][0]
