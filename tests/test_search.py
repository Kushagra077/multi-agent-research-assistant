from typing import ClassVar

import pytest

from research_assistant.tools import search
from research_assistant.tools.search import (
    MIN_PROSE_LINE_CHARS,
    RAW_CONTENT_CHARS,
    _clean_page_text,
    search_web,
)

# A line long enough to count as prose, so it survives _clean_page_text.
PAGE_LINE = (
    "The full page text, written as one real sentence of prose, long enough "
    "that it is clearly a paragraph and not a menu."
)
assert len(PAGE_LINE) >= MIN_PROSE_LINE_CHARS


class FakeTavily:
    """Stands in for TavilySearch so no network call or API key is needed."""

    last_kwargs: ClassVar[dict] = {}
    response: ClassVar[dict | str] = {"results": []}
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
        "raw_content": PAGE_LINE,
    }
    result.update(overrides)
    return result


def test_result_becomes_source_with_snippet_then_page_text(fake_tavily):
    fake_tavily.response = {"results": [_result()]}
    [source] = search_web("anything")
    assert source.url == "https://example.com/a"
    assert source.title == "Page A"
    assert source.content == f"The snippet.\n\n{PAGE_LINE}"


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
    assert source.content == PAGE_LINE


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


# --- Tavily's real habit: failures and empty searches are *returned*, not raised ---


def test_error_dict_from_tavily_is_raised_with_the_real_message(fake_tavily):
    fake_tavily.response = {"error": "ReadTimeout('')"}
    with pytest.raises(RuntimeError, match="ReadTimeout"):
        search_web("anything")


def test_no_results_message_string_means_no_sources(fake_tavily):
    fake_tavily.response = "No search results found for 'q'. Suggestions: try again."
    assert search_web("anything") == []


# These two use the REAL TavilySearch class (only the HTTP layer is stubbed), so they
# fail if the library changes how it reports errors or empty searches.


@pytest.fixture
def real_tavily(monkeypatch):
    from langchain_tavily import TavilySearch
    from langchain_tavily._utilities import TavilySearchAPIWrapper

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(search, "TavilySearch", TavilySearch)
    return TavilySearchAPIWrapper


def test_real_tavily_http_failure_surfaces_as_runtime_error(monkeypatch, real_tavily):
    def boom(self, **kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(real_tavily, "raw_results", boom)
    with pytest.raises(RuntimeError, match="read timed out"):
        search_web("anything")


def test_real_tavily_empty_search_gives_no_sources(monkeypatch, real_tavily):
    monkeypatch.setattr(
        real_tavily, "raw_results", lambda self, **kwargs: {"results": []}
    )
    assert search_web("anything") == []


# --- _clean_page_text: real junk copied from pages Tavily returned ---------------

FREEBSD_PAGE_HEAD = """[![](/logo-02-terminal.svg)FreeBSD.software](/)



[Home](/)/[Guides](/blog.html)/SQLite on FreeBSD: Embedded Database Review

review·2026-04-09·12 min read

# SQLite on FreeBSD: Embedded Database Review

In-depth review of SQLite on FreeBSD: embedded database use cases, WAL mode, performance tuning, concurrent access patterns, and comparison with PostgreSQL for small applications.

By FreeBSD Software editors. Verify changes in the[FreeBSD Handbook](https://docs.freebsd.org/en/books/handbook/) and [manual pages](https://man.freebsd.org/).
"""

REDDIT_PAGE_HEAD = """[Skip to main content](https://www.reddit.com/r/PostgreSQL/#main-content)PostgreSQL wal = 1 concurrent writer? : r/PostgreSQL

Open menu Open navigation[](https://www.reddit.com/)Go to Reddit Home

[Sign Up](https://www.reddit.com/register/)Sign up for Reddit[Log In](https://www.reddit.com/login/)Log in to Reddit

[![Image 1](https://styles.redditmedia.com/icon.png?width=96&s=575f38c1) Go to PostgreSQL](https://www.reddit.com/r/PostgreSQL/)

As i understand pg uses wal mode, is it the same as sqlite in wal mode(1 writer multiple readers) or pg can have real concurrent writers?
"""


def test_cleaner_drops_logos_nav_and_images_but_keeps_the_prose():
    cleaned = _clean_page_text(FREEBSD_PAGE_HEAD)
    assert "In-depth review of SQLite on FreeBSD" in cleaned
    assert "logo-02-terminal" not in cleaned
    assert "![" not in cleaned
    assert "review·2026-04-09·12 min read" not in cleaned


def test_cleaner_removes_menus_and_signup_links_but_keeps_the_question():
    cleaned = _clean_page_text(REDDIT_PAGE_HEAD)
    assert "pg can have real concurrent writers?" in cleaned
    for junk in (
        "Open menu",
        "Sign Up",
        "Log In",
        "Reddit Home",
        "https://",
        "Image 1",
    ):
        assert junk not in cleaned


def test_cleaner_keeps_link_text_and_separates_it_from_neighbouring_words():
    cleaned = _clean_page_text(FREEBSD_PAGE_HEAD)
    assert "the FreeBSD Handbook and manual pages" in cleaned  # not "theFreeBSD"
    assert "](" not in cleaned


def test_cleaner_drops_lines_shorter_than_the_prose_threshold():
    short = "x" * (MIN_PROSE_LINE_CHARS - 1)
    exact = "y" * MIN_PROSE_LINE_CHARS
    assert _clean_page_text(f"{short}\n{exact}") == exact


def test_cleaner_of_nothing_or_all_junk_is_empty():
    assert _clean_page_text("") == ""
    assert _clean_page_text("[Home](/)\n![logo](/x.svg)\nMenu") == ""


def test_search_web_does_not_pass_page_chrome_to_the_model(fake_tavily):
    fake_tavily.response = {
        "results": [_result(content="The snippet.", raw_content=REDDIT_PAGE_HEAD)]
    }
    [source] = search_web("anything")
    assert source.content.startswith("The snippet.\n\n")
    assert "pg can have real concurrent writers?" in source.content
    assert "Sign Up" not in source.content
    assert "Open menu" not in source.content


def test_page_text_cap_of_zero_means_snippet_only(fake_tavily, monkeypatch):
    monkeypatch.setattr(search, "RAW_CONTENT_CHARS", 0)
    fake_tavily.response = {"results": [_result()]}
    [source] = search_web("anything")
    assert source.content == "The snippet."
