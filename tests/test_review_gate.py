"""The human approval gate: the graph pauses with interrupt() and a person decides.

Everything external is faked (planner, researcher LLM, search, writer, Jev), so nothing
touches the network or uses any quota."""

from types import SimpleNamespace
from typing import ClassVar

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import ValidationError

from research_assistant.graph.build import build_graph
from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.graph.nodes import writer as writer_module
from research_assistant.schemas.findings import ResearchSummary, Source
from research_assistant.schemas.plan import Plan, Subtask
from research_assistant.schemas.report import FinalReport, HumanDecision

SOURCE = Source(url="https://good.com/a", title="Good", content="Real facts.")
CONFIG = {"configurable": {"thread_id": "t1"}}


class FakePlannerLLM:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return Plan(subtasks=[Subtask(description="What is X?")])


class FakeResearcherLLM:
    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return ResearchSummary(answer="A fake answer.")


class FakeWriterLLM:
    """Writes 'Draft N' (N = how many drafts so far) and records every prompt."""

    prompts: ClassVar[list[str]] = []

    def invoke(self, messages):
        FakeWriterLLM.prompts.append(messages[1].content)
        return SimpleNamespace(content=f"Draft {len(FakeWriterLLM.prompts)}")


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeWriterLLM.prompts = []
    monkeypatch.setattr(planner_module, "get_planner_llm", lambda: FakePlannerLLM())
    monkeypatch.setattr(
        researcher_module, "get_researcher_llm", lambda: FakeResearcherLLM()
    )
    monkeypatch.setattr(
        researcher_module, "search_web", lambda q, exclude_domains=None: [SOURCE]
    )
    monkeypatch.setattr(writer_module, "get_writer_llm", lambda: FakeWriterLLM())


def _graph(saver=None):
    return build_graph(saver or InMemorySaver(), human_review_gate=True)


def _paused(graph):
    """Start a run and return the graph once it is waiting for the reviewer."""
    result = graph.invoke({"question": "What is X?"}, CONFIG)
    assert "__interrupt__" in result
    return graph


def test_the_gate_needs_a_checkpointer():
    with pytest.raises(ValueError, match="checkpointer"):
        build_graph(None, human_review_gate=True)


def test_without_the_gate_a_run_just_ends_as_before():
    result = build_graph(InMemorySaver()).invoke({"question": "q"}, CONFIG)
    assert "__interrupt__" not in result
    assert "final" not in result
    assert "draft" in result


def test_the_run_pauses_after_the_ship_check_and_shows_the_draft():
    graph = _graph()
    result = graph.invoke({"question": "What is X?"}, CONFIG)

    (interruption,) = result["__interrupt__"]
    assert interruption.value["draft"].startswith("Draft 1")
    assert interruption.value["revisions"] == 0
    assert interruption.value["ship_ready"] is False  # Jev is offline in tests
    snapshot = graph.get_state(CONFIG)
    assert snapshot.next == ("human_review",)
    assert "final" not in snapshot.values


def test_approving_delivers_the_draft_as_it_is():
    graph = _paused(_graph())
    result = graph.invoke(Command(resume={"action": "approve"}), CONFIG)
    assert result["final"] == FinalReport(text=result["draft"].text, how="approved")
    assert graph.get_state(CONFIG).next == ()  # the run is over


def test_taking_over_delivers_the_reviewers_own_text():
    graph = _paused(_graph())
    result = graph.invoke(
        Command(resume={"action": "edit", "text": "My own report."}), CONFIG
    )
    assert result["final"] == FinalReport(text="My own report.", how="edited")
    assert result["draft"].text.startswith("Draft 1")  # the writer's draft is kept
    assert graph.get_state(CONFIG).next == ()


def test_rejecting_sends_the_feedback_and_the_old_draft_to_the_writer():
    graph = _paused(_graph())
    result = graph.invoke(
        Command(resume={"action": "reject", "feedback": "  Make it shorter.  "}), CONFIG
    )

    # The writer ran again, saw the feedback and the earlier draft, and the graph is
    # waiting for the reviewer once more, now about a new draft.
    assert len(FakeWriterLLM.prompts) == 2
    second = FakeWriterLLM.prompts[1]
    assert "- Make it shorter." in second
    assert "Earlier draft:\nDraft 1" in second
    assert result["draft"].text.startswith("Draft 2")
    assert result["feedback"] == ["Make it shorter."]
    (interruption,) = result["__interrupt__"]
    assert interruption.value["revisions"] == 1
    assert "final" not in result


def test_the_first_draft_prompt_has_no_feedback_section():
    _paused(_graph())
    assert "reviewer" not in FakeWriterLLM.prompts[0]


def test_feedback_piles_up_and_every_round_is_shown_to_the_writer():
    graph = _paused(_graph())
    graph.invoke(Command(resume={"action": "reject", "feedback": "First."}), CONFIG)
    result = graph.invoke(
        Command(resume={"action": "reject", "feedback": "Second."}), CONFIG
    )
    third = FakeWriterLLM.prompts[2]
    assert third.index("- First.") < third.index("- Second.")
    assert result["feedback"] == ["First.", "Second."]
    done = graph.invoke(Command(resume={"action": "approve"}), CONFIG)
    assert done["final"].text.startswith("Draft 3")


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "reject"},
        {"action": "reject", "feedback": "   "},
        {"action": "edit"},
        {"action": "edit", "text": ""},
        {"action": "shrug"},
    ],
)
def test_a_decision_missing_what_it_needs_is_refused(decision):
    with pytest.raises(ValidationError):
        HumanDecision.model_validate(decision)


def test_a_run_waiting_for_the_reviewer_survives_a_process_restart(tmp_path):
    """The brief's proof: pause, restart the process, resume, state intact."""
    db = str(tmp_path / "gate.db")

    with SqliteSaver.from_conn_string(db) as saver:  # the first "process"
        graph = _graph(saver)
        paused = graph.invoke({"question": "What is X?"}, CONFIG)
        draft_before = graph.get_state(CONFIG).values["draft"].text
        assert "__interrupt__" in paused
    # the connection is closed: the process "died" while waiting for the reviewer

    FakeWriterLLM.prompts = []
    with SqliteSaver.from_conn_string(db) as saver:  # a new "process"
        graph = _graph(saver)
        snapshot = graph.get_state(CONFIG)
        assert snapshot.next == ("human_review",)
        assert snapshot.values["draft"].text == draft_before
        result = graph.invoke(Command(resume={"action": "approve"}), CONFIG)

    assert result["final"] == FinalReport(text=draft_before, how="approved")
    # Nothing was researched or written again after the restart.
    assert FakeWriterLLM.prompts == []
