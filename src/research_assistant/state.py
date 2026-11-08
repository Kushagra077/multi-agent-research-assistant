# GraphState: the single shared object passed between every node.
# Needs so far: question, plan, findings.
# ResearcherInput: what each parallel researcher receives via Send (not the full state).

from operator import add
from typing import Annotated, NotRequired, TypedDict

from research_assistant.schemas.findings import Finding
from research_assistant.schemas.plan import Plan, Subtask


class GraphState(TypedDict):
    question: str
    plan: NotRequired[Plan]
    # Reducer: parallel researchers each return [finding]; `add` concatenates them.
    findings: NotRequired[Annotated[list[Finding], add]]


class ResearcherInput(TypedDict):
    subtask: Subtask
