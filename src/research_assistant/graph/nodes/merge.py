# Join point after the parallel researchers.
# LangGraph already waits for every researcher before this node runs, so there is
# nothing to do yet. Later phases attach here: the Jev source-quality gate, then the writer.

from research_assistant.state import GraphState


def merge(state: GraphState) -> dict:
    return {}
