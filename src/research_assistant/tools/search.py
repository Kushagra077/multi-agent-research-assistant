from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from research_assistant.schemas.findings import Source

load_dotenv()

MAX_RESULTS = 2
RAW_CONTENT_CHARS = 1500  # how much of the full page to keep per source


def search_web(query: str) -> list[Source]:
    tool = TavilySearch(max_results=MAX_RESULTS, include_raw_content=True)
    response = tool.invoke({"query": query})

    sources = []
    for result in response["results"]:
        parts = [
            result.get("content"),
            (result.get("raw_content") or "")[:RAW_CONTENT_CHARS],
        ]
        text = "\n\n".join(p for p in parts if p)
        if not text:
            continue
        sources.append(Source(url=result["url"], title=result["title"], content=text))
    return sources
