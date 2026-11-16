# The human approval gate. interrupt() pauses the graph and saves its state in the
# checkpoint; the run stays paused, even across a process restart, until someone resumes
# it with Command(resume=<decision>). Because the node starts again from the top when it
# resumes, everything before interrupt() must be harmless to repeat: here there is
# nothing but reading state.

import logging

from langgraph.types import interrupt

from research_assistant.schemas.report import FinalReport, HumanDecision
from research_assistant.state import GraphState

logger = logging.getLogger(__name__)


def human_review(state: GraphState) -> dict:
    draft = state["draft"]
    ship = state.get("ship")
    revisions = len(state.get("feedback", []))
    logger.info("human_review waiting revisions=%d", revisions)

    # What the reviewer is shown. The full picture (plan, sources) is in the state; the
    # API reads it from there.
    payload = interrupt(
        {
            "draft": draft.text,
            "ship_ready": ship.ready if ship else None,
            "revisions": revisions,
        }
    )

    decision = HumanDecision.model_validate(payload)
    logger.info("human_review resumed action=%s", decision.action)
    if decision.action == "approve":
        return {"final": FinalReport(text=draft.text, how="approved")}
    if decision.action == "edit":
        return {"final": FinalReport(text=decision.text, how="edited")}
    return {"feedback": [decision.feedback.strip()]}
