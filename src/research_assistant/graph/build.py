# Assembles the StateGraph and compiles it.
# intake -> planner -> researchers (parallel, one per subtask, via Send) -> merge -> decide
# decide -> researchers again for subtasks that need a retry, otherwise END

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from research_assistant.graph.nodes.decide import decide
from research_assistant.graph.nodes.intake import intake
from research_assistant.graph.nodes.merge import merge
from research_assistant.graph.nodes.planner import planner
from research_assistant.graph.nodes.researcher import researcher
from research_assistant.schemas.findings import attempts_by_subtask
from research_assistant.state import GraphState


def dispatch(state: GraphState) -> list[Send]:
    return [Send("researcher", {"subtask": s}) for s in state["plan"].subtasks]


def retry_dispatch(state: GraphState) -> list[Send] | str:
    """After decide: send each subtask that should be retried back to the researcher."""
    subtasks = {s.id: s for s in state["plan"].subtasks}
    attempts = attempts_by_subtask(state.get("findings", []))
    sends = []
    for decision in state.get("decisions", []):
        # A decision is acted on once: after the retry the attempt count has moved on.
        if decision.action == "escalate" or decision.attempt != attempts.get(
            decision.subtask_id
        ):
            continue
        payload = {"subtask": subtasks[decision.subtask_id]}
        if decision.exclude_domains:
            payload["exclude_domains"] = decision.exclude_domains
        sends.append(Send("researcher", payload))
    return sends or END


# The caller chooses the checkpointer (InMemorySaver now, SqliteSaver later).
# None means no checkpoints are saved, which is the behaviour before Phase 3.
def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph_builder = StateGraph(state_schema=GraphState)

    graph_builder.add_node("intake", intake)
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("merge", merge)
    graph_builder.add_node("decide", decide)

    graph_builder.add_edge(START, "intake")
    graph_builder.add_edge("intake", "planner")
    graph_builder.add_conditional_edges("planner", dispatch, ["researcher"])
    graph_builder.add_edge("researcher", "merge")
    graph_builder.add_edge("merge", "decide")
    graph_builder.add_conditional_edges("decide", retry_dispatch, ["researcher", END])

    return graph_builder.compile(checkpointer=checkpointer)
