# Calls the LLM with structured output to turn a question into a Plan.
# TODO: def planner(state: GraphState) -> dict


from langchain.messages import HumanMessage, SystemMessage

from research_assistant.llm.providers import get_planner_llm
from research_assistant.schemas.plan import Plan
from research_assistant.state import GraphState

PLANNER_SYSTEM_PROMPT = """You are a research planner. Break the user's research \
question into focused subtasks that separate researchers can work on in parallel.

Rules:
- Produce 2 to 3 subtasks. Use fewer only if the question is genuinely narrow.
- Each subtask must be a specific, self-contained question that can be answered \
with web search. Name the key entities, time frames and constraints from the \
original question; do not rely on the other subtasks for context.
- Subtasks must not overlap, and together they must cover the whole question.
- Do not answer the question yourself.
- Do not include steps like "summarize", "compare findings" or "write the report". \
Those happen later; you only plan the research."""


def planner(state: GraphState) -> dict:
    llm = get_planner_llm().with_structured_output(Plan)
    question = state["question"]
    plan = llm.invoke([SystemMessage(PLANNER_SYSTEM_PROMPT), HumanMessage(question)])

    return {"plan": plan}
