from langchain.messages import HumanMessage, SystemMessage

from research_assistant.llm.providers import get_researcher_llm
from research_assistant.schemas.findings import Finding, ResearchSummary, Source
from research_assistant.state import ResearcherInput
from research_assistant.tools.search import search_web

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
        blocks.append(f"Source {number}: {source.title}\nURL: {source.url}\n{source.content}")
    return "\n\n---\n\n".join(blocks)

def researcher(state: ResearcherInput) -> dict:
    subtask = state["subtask"]
    try:
        sources = search_web(subtask.description)
        if not sources:
            finding = Finding(subtask_id=subtask.id, status="no_results")
            return {"findings": [finding]}

        llm = get_researcher_llm().with_structured_output(
            ResearchSummary, method="json_schema"
        )
        human_text = f"Question: {subtask.description}\n\n{_format_sources(sources)}"
        summary = llm.invoke(
            [SystemMessage(RESEARCHER_SYSTEM_PROMPT), HumanMessage(human_text)]
        )

        finding = Finding(
            subtask_id=subtask.id,
            status="found",
            answer=summary.answer,
            sources=sources,
        )
    except Exception as e:
        finding = Finding(subtask_id=subtask.id, status="failed", error=str(e))

    return {"findings": [finding]}
