# GraphState: the single shared object passed between every node.
# Step 1 needs: question, plan.

from typing import NotRequired, TypedDict
from research_assistant.schemas.plan import Plan


class GraphState(TypedDict):
    question: str
    plan: NotRequired[Plan]
