import logging
import re
import time
from urllib.parse import urlparse

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


def domain_of(url: str) -> str:
    """The site a url belongs to, in the form Tavily's exclude_domains takes.

    "https://www.amazon.com/dp/123" -> "amazon.com". A leading "www." is dropped; other
    subdomains are kept, so a rejected "lessons.example.com" does not exclude all of
    "example.com". Returns "" if the url has no host.
    """
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def search_web(query: str, exclude_domains: list[str] | None = None) -> list[Source]:
    """Search the web and return cleaned sources. `exclude_domains` are skipped."""
    logger.info(
        "search start query=%r exclude_domains=%s", query, exclude_domains or []
    )
    started = time.perf_counter()
    tool_kwargs: dict = {"max_results": MAX_RESULTS, "include_raw_content": True}
    # Only sent when there is something to exclude, so a normal search is unchanged.
    if exclude_domains:
        tool_kwargs["exclude_domains"] = exclude_domains
    tool = TavilySearch(**tool_kwargs)
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
