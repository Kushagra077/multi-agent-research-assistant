# Validates/passes through the incoming question.
# TODO: def intake(state: GraphState) -> dict

from research_assistant.state import GraphState


def intake(state: GraphState) -> dict:
    question = state["question"].strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return {"question": question}
