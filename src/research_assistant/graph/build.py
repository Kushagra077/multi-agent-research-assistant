# Assembles the StateGraph and compiles it.
# Step 1: intake -> planner, nothing else yet.

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_assistant.graph.nodes.intake import intake
from research_assistant.graph.nodes.planner import planner
from research_assistant.state import GraphState


def build_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(state_schema=GraphState)

    graph_builder.add_node("intake", intake)
    graph_builder.add_node("planner", planner)

    graph_builder.add_edge(START, "intake")
    graph_builder.add_edge("intake", "planner")
    graph_builder.add_edge("planner", END)

    return graph_builder.compile()
