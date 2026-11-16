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
    # One Review per (subtask, attempt), appended by merge like findings are. `findings`
    # stays as the researchers produced it; reviews says what the Jev checks decided.
    # A retry adds a new review and keeps the old one, so the history is not lost.
    reviews: NotRequired[Annotated[list[Review], add]]


class ResearcherInput(TypedDict):
    subtask: Subtask
    # Sites the search must skip. Set only when a subtask is retried after the Jev
    # source gate rejected pages from them.
    exclude_domains: NotRequired[list[str]]
