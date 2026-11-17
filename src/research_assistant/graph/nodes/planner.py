# Calls the LLM with structured output to turn a question into a Plan.
# TODO: def planner(state: GraphState) -> dict


import logging
import time

from langchain.messages import HumanMessage, SystemMessage

from research_assistant.llm.providers import get_planner_llm
from research_assistant.schemas.plan import Plan
from research_assistant.state import GraphState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a research planner. Break the user's research \
question into focused subtasks that separate researchers can work on in parallel.

Rules:
- Produce 2 to 3 subtasks. Use fewer only if the question is genuinely narrow.
- Each subtask must be a specific, self-contained question that can be answered \
with web search. Name the key entities, time frames and constraints from the \
original question; do not rely on the other subtasks for context.
- Every subtask must name the main subject of the original question (for example the \
instrument, product, technology or field), even if that repeats words. "What daily \
practice routine suits adult beginners?" is too vague; "What daily practice routine \
suits adult beginners learning guitar?" is right.
- Subtasks must not overlap, and together they must cover the whole question.
- Do not answer the question yourself.
- Do not include steps like "summarize", "compare findings" or "write the report". \
Those happen later; you only plan the research."""


def planner(state: GraphState) -> dict:
    question = state["question"]
    logger.info("planner start question=%r", question)
    started = time.perf_counter()
    try:
        llm = get_planner_llm().with_structured_output(Plan)
        plan = llm.invoke(
            [SystemMessage(PLANNER_SYSTEM_PROMPT), HumanMessage(question)]
        )
    except Exception:
        # Unlike the researcher, the planner does not turn failures into a result: the
        # run stops here. Log the traceback so the log says where it stopped, then
        # let the exception continue.
        logger.exception("planner failed after %.1fs", time.perf_counter() - started)
        raise
    logger.info(
        "planner done subtasks=%d ids=%s elapsed=%.1fs",
        len(plan.subtasks),
        [str(subtask.id) for subtask in plan.subtasks],
        time.perf_counter() - started,
    )
    return {"plan": plan}
