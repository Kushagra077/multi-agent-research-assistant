# Decision 4: is the draft good enough to be shown to a human for approval?
# A draft with an unanswered subtask cannot be complete, so that case is decided
# without asking Jev. If Jev is unreachable the answer is "not ready": a person should
# look at a draft nobody has checked. Nothing here blocks delivery by itself; the
# verdict is for the Phase 5 review screen.

import logging

from research_assistant.decisions.jev import ask_boolean
from research_assistant.schemas.report import ShipVerdict
from research_assistant.state import GraphState

logger = logging.getLogger(__name__)

SHIP_QUESTION = (
    "Is this draft a complete and coherent answer to the research question, covering "
    "every part of the question without contradicting itself?"
)
# When Jev is unreachable: do not call the draft ready.
SHIP_DEFAULT = False


def ship_check(state: GraphState) -> dict:
    draft = state["draft"]

    if draft.unanswered or not draft.answered:
        verdict = ShipVerdict(ready=False, confidence=None, reason="unanswered")
    else:
        decision = ask_boolean(
            f"Research question: {state['question']}\n\nDraft report:\n{draft.text}",
            SHIP_QUESTION,
            default=SHIP_DEFAULT,
        )
        verdict = ShipVerdict(
            ready=bool(decision.value),
            confidence=decision.confidence,
            reason="fallback" if decision.fallback else "jev",
        )

    logger.info(
        "ship_check ready=%s reason=%s confidence=%s",
        verdict.ready,
        verdict.reason,
        verdict.confidence,
    )
    return {"ship": verdict}
