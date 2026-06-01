from company_research.fetch import _failed_fetch_result, _short_error


def test_failed_fetch_result_records_warnings_without_body():
    result = _failed_fetch_result("https://example.test/profile", ("robots allowed", "rendered fetch failed"))

    assert result.status_code == 0
    assert result.body == b""
    assert result.final_url == "https://example.test/profile"
    assert result.warnings == ("robots allowed", "rendered fetch failed")


def test_short_error_keeps_first_line_only():
    error = RuntimeError("first line\nsecond line")

    assert _short_error(error) == "RuntimeError: first line"
