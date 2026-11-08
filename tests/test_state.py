from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_assistant.schemas.findings import Finding
from research_assistant.schemas.plan import Plan, Subtask
from research_assistant.state import GraphState, ResearcherInput


def _build_fake_graph(subtask_count: int):
    """A graph with the real state and fan-out wiring but fake nodes (no API calls)."""

    def planner(state: GraphState) -> dict:
        subtasks = [Subtask(description=f"task {i}") for i in range(subtask_count)]
        return {"plan": Plan(subtasks=subtasks)}

    def dispatch(state: GraphState) -> list[Send]:
        return [Send("researcher", {"subtask": s}) for s in state["plan"].subtasks]

    def researcher(state: ResearcherInput) -> dict:
        finding = Finding(subtask_id=state["subtask"].id, status="no_results")
        return {"findings": [finding]}

    def merge(state: GraphState) -> dict:
        return {}

    builder = StateGraph(GraphState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("merge", merge)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch, ["researcher"])
    builder.add_edge("researcher", "merge")
    builder.add_edge("merge", END)
    return builder.compile()


def test_parallel_researchers_all_land_in_findings():
    result = _build_fake_graph(subtask_count=3).invoke({"question": "q"})
    assert len(result["findings"]) == 3


def test_each_finding_matches_one_subtask():
    result = _build_fake_graph(subtask_count=3).invoke({"question": "q"})
    subtask_ids = {s.id for s in result["plan"].subtasks}
    finding_ids = [f.subtask_id for f in result["findings"]]
    assert set(finding_ids) == subtask_ids
    assert len(finding_ids) == len(subtask_ids)  # no duplicates, none missing
