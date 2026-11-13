# Validates/passes through the incoming question.
# TODO: def intake(state: GraphState) -> dict

import logging

from research_assistant.state import GraphState

logger = logging.getLogger(__name__)


def intake(state: GraphState) -> dict:
    question = state["question"].strip()
    if not question:
        # Nothing above this node catches the error, so the log is where a rejected
        # run leaves a record.
        logger.warning("intake rejected a blank question")
        raise ValueError("Question cannot be empty.")
    logger.info("intake ok question=%r", question)
    return {"question": question}
