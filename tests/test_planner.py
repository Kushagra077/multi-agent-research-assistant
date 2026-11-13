"""Logging behaviour of the planner node. The LLM is faked: no network, no quota."""

import logging
from typing import ClassVar

import pytest

from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes.planner import planner
from research_assistant.schemas.plan import Plan, Subtask


class FakePlannerLLM:
    reply: ClassVar[object]

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if isinstance(FakePlannerLLM.reply, Exception):
            raise FakePlannerLLM.reply
        return FakePlannerLLM.reply


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    FakePlannerLLM.reply = Plan(subtasks=[Subtask(description="q1")])
    monkeypatch.setattr(planner_module, "get_planner_llm", lambda: FakePlannerLLM())


def test_planner_returns_the_plan_and_logs_start_and_done(caplog):
    with caplog.at_level(logging.INFO, logger=planner_module.logger.name):
        result = planner({"question": "What is RAG?"})
    assert result == {"plan": FakePlannerLLM.reply}
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("planner start") for m in messages)
    assert any("planner done" in m and "subtasks=1" in m for m in messages)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_planner_failure_is_logged_with_traceback_and_still_raised(caplog):
    FakePlannerLLM.reply = TimeoutError("groq timed out")
    with (
        caplog.at_level(logging.INFO, logger=planner_module.logger.name),
        pytest.raises(TimeoutError, match="groq timed out"),
    ):
        planner({"question": "What is RAG?"})
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert record.exc_info is not None
    assert "groq timed out" in caplog.text
