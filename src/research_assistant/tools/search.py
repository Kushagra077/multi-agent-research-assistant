import logging
import re
import time

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from research_assistant.schemas.findings import Source

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RESULTS = 2
# How much cleaned page text to add after Tavily's snippet. 0 = snippet only.
RAW_CONTENT_CHARS = 1500
# After link/image removal, lines shorter than this are menus, buttons and labels
# ("Skip to main content", "Sign Up"), not sentences.
MIN_PROSE_LINE_CHARS = 80

_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?])")


def _clean_page_text(raw: str) -> str:
    """Strip the page chrome from Tavily's raw page markdown and keep prose lines.

    Raw pages open with logos, nav bars, cookie banners and sign-up links, which would
    otherwise fill the token budget. Images and URLs are removed, links keep only
    their text, and short menu-like lines are dropped.
    """
    text = _MARKDOWN_IMAGE.sub("", raw)
    # Spaces on both sides: "the[a](u)b" must not become "theab". Extra spaces are
    # collapsed below, and the one left before punctuation is removed.
    text = _MARKDOWN_LINK.sub(r" \1 ", text)
    text = _BARE_URL.sub("", text)
    lines = (
        _SPACE_BEFORE_PUNCTUATION.sub(r"\1", " ".join(line.split()))
        for line in text.splitlines()
    )
    return "\n".join(line for line in lines if len(line) >= MIN_PROSE_LINE_CHARS)


def search_web(query: str) -> list[Source]:
    logger.info("search start query=%r", query)
    started = time.perf_counter()
    tool = TavilySearch(max_results=MAX_RESULTS, include_raw_content=True)
    response = tool.invoke({"query": query})
    elapsed = time.perf_counter() - started

    # TavilySearch does not raise on failure; invoke() can return three shapes:
    #   {"results": [...]}  normal
    #   {"error": "..."}    any failure (timeout, 429, bad key) is returned, not raised
    #   "No search results found for ..."  a plain string when nothing matched
    if isinstance(response, str):
        logger.info("search no results query=%r elapsed=%.1fs", query, elapsed)
        return []
    if "error" in response:
        # Log the real Tavily message here, before raising: the caller turns the
        # exception into a failed finding, and this is the only place that has it.
        logger.warning(
            "search error query=%r elapsed=%.1fs tavily_error=%s",
            query,
            elapsed,
            response["error"],
        )
        raise RuntimeError(f"Tavily search failed: {response['error']}")

    sources = []
    for result in response["results"]:
        page_text = _clean_page_text(result.get("raw_content") or "")
        parts = [result.get("content"), page_text[:RAW_CONTENT_CHARS]]
        text = "\n\n".join(p for p in parts if p)
        if not text:
            logger.debug("search dropped empty result url=%s", result.get("url"))
            continue
        sources.append(Source(url=result["url"], title=result["title"], content=text))
    logger.info(
        "search done query=%r results=%d sources=%d elapsed=%.1fs",
        query,
        len(response["results"]),
        len(sources),
        elapsed,
    )
    return sources
