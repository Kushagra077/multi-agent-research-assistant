import logging
from typing import ClassVar

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.graph.nodes.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    _format_sources,
    researcher,
)
from research_assistant.schemas.findings import ResearchSummary, Source
from research_assistant.schemas.plan import Plan, Subtask
from research_assistant.state import GraphState


class FakeLLM:
    """Stands in for the Groq client: records what it was asked, returns a canned reply."""

    reply: ClassVar[object] = ResearchSummary(answer="A fake answer.")
    structured_calls: ClassVar[list] = []
    invoked_with: ClassVar[list] = []

    def with_structured_output(self, schema, **kwargs):
        FakeLLM.structured_calls.append((schema, kwargs))
        return self

    def invoke(self, messages):
        FakeLLM.invoked_with.append(messages)
        if isinstance(FakeLLM.reply, Exception):
            raise FakeLLM.reply
        return FakeLLM.reply


SOURCES = [
    Source(url="https://a.com", title="Page A", content="Fact from A."),
    Source(url="https://b.com", title="Page B", content="Fact from B."),
]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeLLM.reply = ResearchSummary(answer="A fake answer.")
    FakeLLM.structured_calls = []
    FakeLLM.invoked_with = []
    monkeypatch.setattr(researcher_module, "get_researcher_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        researcher_module,
        "search_web",
        lambda query, exclude_domains=None: list(SOURCES),
    )


def _run(description: str = "What is X?") -> tuple[Subtask, dict]:
    subtask = Subtask(description=description)
    return subtask, researcher({"subtask": subtask})


def _only_finding(result: dict):
    assert list(result) == ["findings"]
    assert len(result["findings"]) == 1
    return result["findings"][0]


# --- the happy path -------------------------------------------------------


def test_found_finding_carries_our_ids_and_the_original_sources():
    subtask, result = _run()
    finding = _only_finding(result)
    assert finding.status == "found"
    assert finding.subtask_id == subtask.id
    assert finding.answer == "A fake answer."
    assert finding.sources == SOURCES


def test_llm_is_asked_for_a_research_summary_in_json_schema_mode():
    _run()
    assert FakeLLM.structured_calls == [(ResearchSummary, {"method": "json_schema"})]


def test_prompt_has_system_prompt_question_and_numbered_sources():
    _run("What is X?")
    system_message, human_message = FakeLLM.invoked_with[0]
    assert system_message.content == RESEARCHER_SYSTEM_PROMPT
    assert "What is X?" in human_message.content
    assert "Source 1: Page A" in human_message.content
    assert "Source 2: Page B" in human_message.content
    assert "Fact from B." in human_message.content


# --- failures become findings, never exceptions ---------------------------


def test_no_sources_gives_no_results_and_never_calls_the_llm(monkeypatch):
    monkeypatch.setattr(
        researcher_module, "search_web", lambda query, exclude_domains=None: []
    )
    subtask, result = _run()
    finding = _only_finding(result)
    assert finding.status == "no_results"
    assert finding.subtask_id == subtask.id
    assert FakeLLM.invoked_with == []
    assert FakeLLM.structured_calls == []


def test_search_error_becomes_a_failed_finding(monkeypatch):
    def boom(query, exclude_domains=None):
        raise RuntimeError("tavily is down")

    monkeypatch.setattr(researcher_module, "search_web", boom)
    subtask, result = _run()
    finding = _only_finding(result)
    assert finding.status == "failed"
    assert finding.subtask_id == subtask.id
    assert "tavily is down" in finding.error


def test_llm_error_becomes_a_failed_finding():
    FakeLLM.reply = TimeoutError("groq timed out")
    _, result = _run()
    finding = _only_finding(result)
    assert finding.status == "failed"
    assert "groq timed out" in finding.error


def test_empty_answer_becomes_a_failed_finding():
    FakeLLM.reply = ResearchSummary(answer="")
    _, result = _run()
    finding = _only_finding(result)
    assert finding.status == "failed"
    assert finding.error  # the Finding validator's complaint, not blank


def test_exception_with_empty_message_still_gives_a_usable_error():
    FakeLLM.reply = TimeoutError()  # str(TimeoutError()) == ""
    _, result = _run()
    finding = _only_finding(result)
    assert finding.status == "failed"
    assert finding.error == "TimeoutError"


# --- logging --------------------------------------------------------------


def test_success_logs_start_llm_and_done_with_the_subtask_id(caplog):
    with caplog.at_level(logging.INFO, logger=researcher_module.logger.name):
        subtask, _ = _run()
    messages = [r.getMessage() for r in caplog.records]
    sid = str(subtask.id)
    assert any(m.startswith("researcher start") and sid in m for m in messages)
    assert any(m.startswith("llm done") and sid in m for m in messages)
    assert any("researcher done" in m and "status=found" in m for m in messages)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_failure_logs_one_error_with_traceback_and_the_subtask_id(monkeypatch, caplog):
    def boom(query, exclude_domains=None):
        raise RuntimeError("tavily is down")

    monkeypatch.setattr(researcher_module, "search_web", boom)
    with caplog.at_level(logging.INFO, logger=researcher_module.logger.name):
        subtask, _ = _run()
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert str(subtask.id) in record.getMessage()
    assert record.exc_info is not None  # the traceback, not just the message
    assert "tavily is down" in caplog.text
    assert any("status=failed" in r.getMessage() for r in caplog.records)


def test_no_results_is_logged_as_done_not_as_an_error(monkeypatch, caplog):
    monkeypatch.setattr(
        researcher_module, "search_web", lambda query, exclude_domains=None: []
    )
    with caplog.at_level(logging.INFO, logger=researcher_module.logger.name):
        _run()
    assert any("status=no_results" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_log_lines_do_not_contain_answer_or_source_text(caplog):
    with caplog.at_level(logging.DEBUG, logger=researcher_module.logger.name):
        _run()
    assert "A fake answer." not in caplog.text
    assert "Fact from A." not in caplog.text


# --- _format_sources ------------------------------------------------------


def test_format_sources_numbers_from_one_and_separates_with_dividers():
    text = _format_sources(SOURCES)
    assert text.startswith("Source 1: Page A\nURL: https://a.com\nFact from A.")
    assert "\n\n---\n\nSource 2: Page B\nURL: https://b.com\nFact from B." in text
    assert "Source 0" not in text


def test_format_sources_of_nothing_is_an_empty_string():
    assert _format_sources([]) == ""


# --- the real node inside the real fan-out wiring -------------------------


def test_real_researcher_runs_in_parallel_inside_a_graph():
    def planner(state: GraphState) -> dict:
        return {"plan": Plan(subtasks=[Subtask(description=f"q{i}") for i in range(3)])}

    def dispatch(state: GraphState) -> list[Send]:
        return [Send("researcher", {"subtask": s}) for s in state["plan"].subtasks]

    builder = StateGraph(GraphState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("merge", lambda state: {})
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch, ["researcher"])
    builder.add_edge("researcher", "merge")
    builder.add_edge("merge", END)

    result = builder.compile().invoke({"question": "q"})

    assert len(result["findings"]) == 3
    assert {f.subtask_id for f in result["findings"]} == {
        s.id for s in result["plan"].subtasks
    }
    assert all(f.status == "found" for f in result["findings"])


# --- retries: sites the source gate rejected are skipped -------------------


def test_exclude_domains_from_the_input_reach_the_search(monkeypatch):
    seen = {}

    def fake_search(query, exclude_domains=None):
        seen["exclude_domains"] = exclude_domains
        return list(SOURCES)

    monkeypatch.setattr(researcher_module, "search_web", fake_search)
    subtask = Subtask(description="What is X?")
    result = researcher({"subtask": subtask, "exclude_domains": ["amazon.com"]})
    assert seen["exclude_domains"] == ["amazon.com"]
    assert _only_finding(result).status == "found"


def test_a_first_attempt_has_no_exclusions(monkeypatch):
    seen = {}

    def fake_search(query, exclude_domains=None):
        seen["exclude_domains"] = exclude_domains
        return list(SOURCES)

    monkeypatch.setattr(researcher_module, "search_web", fake_search)
    _run()  # the input has no "exclude_domains" key at all
    assert seen["exclude_domains"] == []


def test_the_start_log_line_shows_the_exclusions(caplog):
    subtask = Subtask(description="What is X?")
    with caplog.at_level(logging.INFO, logger=researcher_module.logger.name):
        researcher({"subtask": subtask, "exclude_domains": ["amazon.com"]})
    assert "exclude_domains=['amazon.com']" in caplog.text
