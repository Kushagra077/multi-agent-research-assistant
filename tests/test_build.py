"""End-to-end tests of the real graph. Planner LLM, researcher LLM and web search are
faked, so nothing touches the network or uses any quota."""

from typing import ClassVar

import pytest

from research_assistant.graph.build import build_graph
from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.schemas.findings import ResearchSummary, Source
from research_assistant.schemas.plan import Plan, Subtask

SOURCES = [Source(url="https://a.com", title="Page A", content="Fact from A.")]


class FakePlannerLLM:
    plan: ClassVar[Plan]

    def with_structured_output(self, schema):
        assert schema is Plan
        return self

    def invoke(self, messages):
        return FakePlannerLLM.plan


class FakeResearcherLLM:
    reply: ClassVar[object] = ResearchSummary(answer="A fake answer.")

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        if isinstance(FakeResearcherLLM.reply, Exception):
            raise FakeResearcherLLM.reply
        return FakeResearcherLLM.reply


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakePlannerLLM.plan = Plan(
        subtasks=[Subtask(description=f"question {i}") for i in range(3)]
    )
    FakeResearcherLLM.reply = ResearchSummary(answer="A fake answer.")
    monkeypatch.setattr(planner_module, "get_planner_llm", lambda: FakePlannerLLM())
    monkeypatch.setattr(
        researcher_module, "get_researcher_llm", lambda: FakeResearcherLLM()
    )
    monkeypatch.setattr(researcher_module, "search_web", lambda query: list(SOURCES))


def test_question_goes_in_and_one_finding_per_subtask_comes_out():
    result = build_graph().invoke({"question": "  What is X?  "})

    assert result["question"] == "What is X?"  # intake cleaned it
    subtask_ids = {s.id for s in result["plan"].subtasks}
    assert len(result["findings"]) == 3
    assert {f.subtask_id for f in result["findings"]} == subtask_ids
    assert all(f.status == "found" for f in result["findings"])


def test_blank_question_is_rejected_before_any_work_happens():
    with pytest.raises(ValueError):
        build_graph().invoke({"question": "   "})


def test_one_failing_researcher_does_not_sink_the_run(monkeypatch):
    def flaky_search(query):
        if query == "question 1":
            raise RuntimeError("tavily is down")
        return list(SOURCES)

    monkeypatch.setattr(researcher_module, "search_web", flaky_search)

    result = build_graph().invoke({"question": "What is X?"})

    by_status = sorted(f.status for f in result["findings"])
    assert by_status == ["failed", "found", "found"]
    [failed] = [f for f in result["findings"] if f.status == "failed"]
    assert "tavily is down" in failed.error


def test_every_researcher_failing_still_finishes_with_failed_findings():
    FakeResearcherLLM.reply = TimeoutError("groq timed out")

    result = build_graph().invoke({"question": "What is X?"})

    assert len(result["findings"]) == 3
    assert all(f.status == "failed" for f in result["findings"])


def test_no_search_results_gives_no_results_findings(monkeypatch):
    monkeypatch.setattr(researcher_module, "search_web", lambda query: [])

    result = build_graph().invoke({"question": "What is X?"})

    assert all(f.status == "no_results" for f in result["findings"])
