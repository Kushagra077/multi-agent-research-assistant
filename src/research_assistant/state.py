# GraphState: the single shared object passed between every node.
# Needs so far: question, plan, findings, reviews.
# ResearcherInput: what each parallel researcher receives via Send (not the full state).

from operator import add
from typing import Annotated, NotRequired, TypedDict

from research_assistant.schemas.findings import Finding, Review
from research_assistant.schemas.plan import Plan, Subtask


class GraphState(TypedDict):
    question: str
    plan: NotRequired[Plan]
    # Reducer: parallel researchers each return [finding]; `add` concatenates them.
    findings: NotRequired[Annotated[list[Finding], add]]
    # Written once by merge, so no reducer: it replaces the value. `findings` stays as
    # the researchers produced it; reviews says what the Jev checks decided about it.
    reviews: NotRequired[list[Review]]


class ResearcherInput(TypedDict):
    subtask: Subtask
