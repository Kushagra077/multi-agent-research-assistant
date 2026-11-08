from typing import ClassVar

import pytest

from research_assistant.tools import search
from research_assistant.tools.search import RAW_CONTENT_CHARS, search_web


class FakeTavily:
    """Stands in for TavilySearch so no network call or API key is needed."""

    last_kwargs: ClassVar[dict] = {}
    response: ClassVar[dict] = {"results": []}
    error: ClassVar[Exception | None] = None

    def __init__(self, **kwargs):
        FakeTavily.last_kwargs = kwargs

    def invoke(self, payload):
        if FakeTavily.error:
            raise FakeTavily.error
        return FakeTavily.response


@pytest.fixture(autouse=True)
def fake_tavily(monkeypatch):
    FakeTavily.last_kwargs = {}
    FakeTavily.response = {"results": []}
    FakeTavily.error = None
    monkeypatch.setattr(search, "TavilySearch", FakeTavily)
    return FakeTavily


def _result(**overrides) -> dict:
    result = {
        "url": "https://example.com/a",
        "title": "Page A",
        "content": "The snippet.",
        "raw_content": "The full page text.",
    }
    result.update(overrides)
    return result


def test_result_becomes_source_with_snippet_then_page_text(fake_tavily):
    fake_tavily.response = {"results": [_result()]}
    [source] = search_web("anything")
    assert source.url == "https://example.com/a"
    assert source.title == "Page A"
    assert source.content == "The snippet.\n\nThe full page text."


def test_page_text_is_cut_to_the_cap_but_snippet_is_kept_whole(fake_tavily):
    fake_tavily.response = {
        "results": [_result(content="S" * 5000, raw_content="R" * 10_000)]
    }
    [source] = search_web("anything")
    assert source.content == "S" * 5000 + "\n\n" + "R" * RAW_CONTENT_CHARS


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_page_text_falls_back_to_snippet_only(fake_tavily, missing):
    fake_tavily.response = {"results": [_result(raw_content=missing)]}
    [source] = search_web("anything")
    assert source.content == "The snippet."


def test_page_text_key_absent_falls_back_to_snippet_only(fake_tavily):
    result = _result()
    del result["raw_content"]
    fake_tavily.response = {"results": [result]}
    [source] = search_web("anything")
    assert source.content == "The snippet."


def test_missing_snippet_falls_back_to_page_text_only(fake_tavily):
    fake_tavily.response = {"results": [_result(content="")]}
    [source] = search_web("anything")
    assert source.content == "The full page text."


def test_result_with_no_text_at_all_is_skipped(fake_tavily):
    fake_tavily.response = {
        "results": [
            _result(url="https://empty.com", content="", raw_content=None),
            _result(url="https://good.com"),
        ]
    }
    sources = search_web("anything")
    assert [s.url for s in sources] == ["https://good.com"]


def test_no_results_returns_empty_list(fake_tavily):
    fake_tavily.response = {"results": []}
    assert search_web("anything") == []


def test_tavily_is_asked_for_full_page_text_and_capped_results(fake_tavily):
    search_web("anything")
    assert fake_tavily.last_kwargs == {
        "max_results": search.MAX_RESULTS,
        "include_raw_content": True,
    }


def test_errors_propagate_so_the_researcher_can_report_them(fake_tavily):
    fake_tavily.error = RuntimeError("rate limited")
    with pytest.raises(RuntimeError):
        search_web("anything")
