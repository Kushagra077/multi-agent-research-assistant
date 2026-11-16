# Assembles the StateGraph and compiles it.
# intake -> planner -> researchers (parallel, one per subtask, via Send) -> merge -> decide
# decide -> researchers again for subtasks that need a retry, otherwise writer -> ship_check
# ship_check -> END, or with human_review=True -> human_review (pauses) -> END, or back to
# writer when the reviewer rejects the draft

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from research_assistant.graph.nodes.decide import decide
from research_assistant.graph.nodes.human_review import human_review
from research_assistant.graph.nodes.intake import intake
from research_assistant.graph.nodes.merge import merge
from research_assistant.graph.nodes.planner import planner
from research_assistant.graph.nodes.researcher import researcher
from research_assistant.graph.nodes.ship_check import ship_check
from research_assistant.graph.nodes.writer import writer
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
    return sends or "writer"


def after_review(state: GraphState) -> str:
    """After the reviewer: a final report ends the run; a rejection goes to the writer."""
    return END if state.get("final") else "writer"


# The caller chooses the checkpointer (InMemorySaver now, SqliteSaver later).
# None means no checkpoints are saved, which is the behaviour before Phase 3.
def build_graph(
    checkpointer: BaseCheckpointSaver | None = None, *, human_review_gate: bool = False
) -> CompiledStateGraph:
    """human_review_gate=True pauses the run for a person before it ends. It needs a
    checkpointer (the pause is saved there), so it is off unless asked for."""
    if human_review_gate and checkpointer is None:
        raise ValueError("The human review gate needs a checkpointer.")
    graph_builder = StateGraph(state_schema=GraphState)

    graph_builder.add_node("intake", intake)
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("merge", merge)
    graph_builder.add_node("decide", decide)
    graph_builder.add_node("writer", writer)
    graph_builder.add_node("ship_check", ship_check)

    graph_builder.add_edge(START, "intake")
    graph_builder.add_edge("intake", "planner")
    graph_builder.add_conditional_edges("planner", dispatch, ["researcher"])
    graph_builder.add_edge("researcher", "merge")
    graph_builder.add_edge("merge", "decide")
    graph_builder.add_conditional_edges(
        "decide", retry_dispatch, ["researcher", "writer"]
    )
    graph_builder.add_edge("writer", "ship_check")
    if human_review_gate:
        graph_builder.add_node("human_review", human_review)
        graph_builder.add_edge("ship_check", "human_review")
        graph_builder.add_conditional_edges(
            "human_review", after_review, ["writer", END]
        )
    else:
        graph_builder.add_edge("ship_check", END)

    return graph_builder.compile(checkpointer=checkpointer)
