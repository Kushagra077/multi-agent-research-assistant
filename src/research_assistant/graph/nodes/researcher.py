import logging
import time

from langchain.messages import HumanMessage, SystemMessage

from research_assistant.llm.providers import get_researcher_llm
from research_assistant.schemas.findings import Finding, ResearchSummary, Source
from research_assistant.state import ResearcherInput
from research_assistant.tools.search import search_web

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant. You are given one \
research question and a set of numbered web sources. Answer the question using \
only the information in those sources.

Rules:
- Use only facts that appear in the sources. Do not add outside knowledge, and do \
not guess.
- Be concise and specific. Keep concrete details such as numbers, dates, names and \
units.
- If sources disagree, say so and state what each one claims.
- If the sources do not contain enough information to answer, say exactly that and \
state what is missing. Do not make up an answer.
- When you state a fact, mention which source it came from, for example \
"(Source 2)".
- Do not invent sources or URLs, and do not include a list of sources. The sources \
are recorded separately.
- Write the answer as plain prose, with no headings."""


def _format_sources(sources: list[Source]) -> str:
    blocks = []
    for number, source in enumerate(sources, start=1):
        blocks.append(
            f"Source {number}: {source.title}\nURL: {source.url}\n{source.content}"
        )
    return "\n\n---\n\n".join(blocks)


def researcher(state: ResearcherInput) -> dict:
    subtask = state["subtask"]
    exclude_domains = state.get("exclude_domains", [])
    started = time.perf_counter()
    # Researchers run in parallel, so their log lines interleave: every line below
    # carries subtask_id to tell them apart.
    logger.info(
        "researcher start subtask_id=%s description=%r exclude_domains=%s",
        subtask.id,
        subtask.description,
        exclude_domains,
    )
    try:
        sources = search_web(subtask.description, exclude_domains=exclude_domains)
        if not sources:
            finding = Finding(subtask_id=subtask.id, status="no_results")
            logger.info(
                "researcher done subtask_id=%s status=%s elapsed=%.1fs",
                subtask.id,
                finding.status,
                time.perf_counter() - started,
            )
            return {"findings": [finding]}

        llm = get_researcher_llm().with_structured_output(
            ResearchSummary, method="json_schema"
        )
        human_text = f"Question: {subtask.description}\n\n{_format_sources(sources)}"
        logger.info("llm start subtask_id=%s sources=%d", subtask.id, len(sources))
        llm_started = time.perf_counter()
        summary = llm.invoke(
            [SystemMessage(RESEARCHER_SYSTEM_PROMPT), HumanMessage(human_text)]
        )
        # Lengths only: the prompt and the answer hold page text and stay out of the log.
        logger.info(
            "llm done subtask_id=%s elapsed=%.1fs answer_chars=%d",
            subtask.id,
            time.perf_counter() - llm_started,
            len(summary.answer),
        )

        finding = Finding(
            subtask_id=subtask.id,
            status="found",
            answer=summary.answer,
            sources=sources,
        )
    except Exception as e:  # deliberate: any failure becomes a typed Finding
        # `or type(e).__name__` because some exceptions have an empty message, and a
        # 'failed' Finding must carry a non-empty error.
        # logger.exception keeps the full traceback in the log; the Finding keeps only
        # the short message for the graph.
        logger.exception("researcher failed subtask_id=%s", subtask.id)
        finding = Finding(
            subtask_id=subtask.id, status="failed", error=str(e) or type(e).__name__
        )

    logger.info(
        "researcher done subtask_id=%s status=%s elapsed=%.1fs",
        subtask.id,
        finding.status,
        time.perf_counter() - started,
    )
    return {"findings": [finding]}
