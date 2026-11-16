"""Time travel: replay and fork from an old checkpoint. Fakes only, so no network and
no API quota. The fakes count calls, so each test can show which nodes really ran."""

from typing import ClassVar

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from research_assistant.graph.build import build_graph
from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.schemas.findings import ResearchSummary, Source
from research_assistant.schemas.plan import Plan, Subtask

SUBTASK_COUNT = 3
CONFIG = {"configurable": {"thread_id": "t1"}}


class Calls:
    planner = 0
    searches = 0


class FakePlannerLLM:
    plan: ClassVar[Plan]

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        Calls.planner += 1
        return FakePlannerLLM.plan


class FakeResearcherLLM:
    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return ResearchSummary(answer="A fake answer.")


def fake_search(query: str, exclude_domains=None) -> list[Source]:
    Calls.searches += 1
    return [Source(url="https://a.com", title="Page A", content=f"About {query}.")]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    Calls.planner = 0
    Calls.searches = 0
    FakePlannerLLM.plan = Plan(
        subtasks=[Subtask(description=f"question {i}") for i in range(SUBTASK_COUNT)]
    )
    monkeypatch.setattr(planner_module, "get_planner_llm", lambda: FakePlannerLLM())
    monkeypatch.setattr(
        researcher_module, "get_researcher_llm", lambda: FakeResearcherLLM()
    )
    monkeypatch.setattr(researcher_module, "search_web", fake_search)


@pytest.fixture
def finished_run():
    """A graph that has run once to the end. Counters are reset afterwards, so each
    test counts only what happens AFTER the time travel."""
    graph = build_graph(InMemorySaver())
    graph.invoke({"question": "What is X?"}, CONFIG)
    assert (Calls.planner, Calls.searches) == (1, SUBTASK_COUNT)
    Calls.planner = 0
    Calls.searches = 0
    return graph


def _checkpoint_about_to_run(graph, *nodes: str):
    """The saved checkpoint whose `next` is exactly these nodes."""
    return next(s for s in graph.get_state_history(CONFIG) if s.next == nodes)


def _before_researchers(graph):
    return _checkpoint_about_to_run(graph, *["researcher"] * SUBTASK_COUNT)


# --- replay: same state, run forward again --------------------------------------


def test_replay_from_before_the_researchers_re_runs_them_but_not_the_planner(
    finished_run,
):
    snapshot = _before_researchers(finished_run)
    finished_run.invoke(None, snapshot.config)
    assert Calls.planner == 0  # the plan was already saved in that checkpoint
    assert Calls.searches == SUBTASK_COUNT  # replay really searches again


def test_replay_does_not_double_the_findings(finished_run):
    snapshot = _before_researchers(finished_run)
    result = finished_run.invoke(None, snapshot.config)
    # The old findings were produced AFTER this checkpoint, so they are not in it.
    assert len(result["findings"]) == SUBTASK_COUNT


def test_replay_adds_a_branch_and_keeps_the_old_checkpoints(finished_run):
    before = list(finished_run.get_state_history(CONFIG))
    finished_run.invoke(None, _before_researchers(finished_run).config)
    after = list(finished_run.get_state_history(CONFIG))
    # 4 new checkpoints: the researcher step, merge, decide, and the final one
    assert len(before) == 7
    assert len(after) == 11
    old_ids = {s.config["configurable"]["checkpoint_id"] for s in before}
    new_ids = {s.config["configurable"]["checkpoint_id"] for s in after}
    assert old_ids <= new_ids


def test_replay_from_before_the_planner_calls_the_planner_again(finished_run):
    snapshot = _checkpoint_about_to_run(finished_run, "planner")
    finished_run.invoke(None, snapshot.config)
    assert Calls.planner == 1
    assert Calls.searches == SUBTASK_COUNT


def test_replaying_a_finished_checkpoint_runs_nothing(finished_run):
    latest = finished_run.get_state(CONFIG)
    assert latest.next == ()
    finished_run.invoke(None, latest.config)
    assert (Calls.planner, Calls.searches) == (0, 0)


# --- fork: change the state, then run forward -----------------------------------


def test_fork_with_a_new_plan_researches_only_the_new_plan(finished_run):
    snapshot = _before_researchers(finished_run)
    new_plan = Plan(subtasks=[Subtask(description="only question")])

    fork = finished_run.update_state(
        snapshot.config, {"plan": new_plan}, as_node="planner"
    )
    result = finished_run.invoke(None, fork)

    assert [s.description for s in result["plan"].subtasks] == ["only question"]
    assert len(result["findings"]) == 1
    assert Calls.planner == 0  # we supplied the plan, so the planner never ran
    assert Calls.searches == 1


def test_fork_gets_a_new_checkpoint_id_and_marks_it_as_an_update(finished_run):
    snapshot = _before_researchers(finished_run)
    new_plan = Plan(subtasks=[Subtask(description="only question")])
    fork = finished_run.update_state(
        snapshot.config, {"plan": new_plan}, as_node="planner"
    )
    assert (
        fork["configurable"]["checkpoint_id"]
        != snapshot.config["configurable"]["checkpoint_id"]
    )
    assert finished_run.get_state(fork).metadata["source"] == "update"
    assert finished_run.get_state(fork).next == ("researcher",)  # one new subtask


def test_fork_does_not_change_the_checkpoint_it_started_from(finished_run):
    snapshot = _before_researchers(finished_run)
    new_plan = Plan(subtasks=[Subtask(description="only question")])
    fork = finished_run.update_state(
        snapshot.config, {"plan": new_plan}, as_node="planner"
    )
    finished_run.invoke(None, fork)

    # read the original checkpoint again, by its own id
    original = finished_run.get_state(snapshot.config)
    assert len(original.values["plan"].subtasks) == SUBTASK_COUNT
    assert original.next == ("researcher",) * SUBTASK_COUNT


def test_fork_works_without_as_node_on_this_graph(finished_run):
    # LangGraph infers that the planner was the last writer. Passing as_node
    # explicitly is clearer, and is required only for parallel branches, a fresh
    # thread, or skipping nodes.
    snapshot = _before_researchers(finished_run)
    new_plan = Plan(subtasks=[Subtask(description="only question")])
    fork = finished_run.update_state(snapshot.config, {"plan": new_plan})
    result = finished_run.invoke(None, fork)
    assert len(result["findings"]) == 1
