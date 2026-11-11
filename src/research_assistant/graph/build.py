# Assembles the StateGraph and compiles it.
# intake -> planner -> researchers (parallel, one per subtask, via Send) -> merge -> END

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from research_assistant.graph.nodes.intake import intake
from research_assistant.graph.nodes.merge import merge
from research_assistant.graph.nodes.planner import planner
from research_assistant.graph.nodes.researcher import researcher
from research_assistant.state import GraphState


def dispatch(state: GraphState) -> list[Send]:
    return [Send("researcher", {"subtask": s}) for s in state["plan"].subtasks]


def build_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(state_schema=GraphState)

    graph_builder.add_node("intake", intake)
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("merge", merge)

    graph_builder.add_edge(START, "intake")
    graph_builder.add_edge("intake", "planner")
    graph_builder.add_conditional_edges("planner", dispatch, ["researcher"])
    graph_builder.add_edge("researcher", "merge")
    graph_builder.add_edge("merge", END)

    return graph_builder.compile()
