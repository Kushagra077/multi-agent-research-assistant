import logging
from types import SimpleNamespace
from typing import ClassVar
from uuid import uuid4

import pytest
from typesafe_sdk import TypeSafeAPIConnectionError

from research_assistant.decisions import jev
from research_assistant.graph.nodes import ship_check as ship_module
from research_assistant.graph.nodes.ship_check import SHIP_QUESTION, ship_check
from research_assistant.schemas.report import Draft, ShipVerdict


class FakeJev:
    noul: ClassVar[float] = 0.9
    calls: ClassVar[list[tuple[str, str]]] = []  # (question text, state)

    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        FakeJev.calls.append((questions["q"].instructions, state))
        return SimpleNamespace(
            model="jev-1.13.0", nouls={"q": SimpleNamespace(noul=FakeJev.noul)}
        )


class DownJev:
    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        raise TypeSafeAPIConnectionError("jev is down")


@pytest.fixture(autouse=True)
def fake_jev(monkeypatch):
    FakeJev.noul = 0.9
    FakeJev.calls = []
    monkeypatch.setattr(jev, "TypeSafeClient", FakeJev)


def _state(*, answered=1, unanswered=0):
    draft = Draft(
        text="The whole report.",
        answered=[uuid4() for _ in range(answered)],
        unanswered=[uuid4() for _ in range(unanswered)],
    )
    return {"question": "What is X?", "draft": draft}


def test_jev_says_ready():
    verdict = ship_check(_state())["ship"]
    assert verdict == ShipVerdict(ready=True, confidence=0.9, reason="jev")


def test_jev_says_not_ready():
    FakeJev.noul = 0.1
    verdict = ship_check(_state())["ship"]
    assert verdict.ready is False
    assert verdict.reason == "jev"
    assert verdict.confidence == pytest.approx(0.9)


def test_jev_is_shown_the_question_and_the_whole_draft():
    ship_check(_state())
    ((question, state),) = FakeJev.calls
    assert question == SHIP_QUESTION
    assert "Research question: What is X?" in state
    assert "The whole report." in state


def test_a_draft_with_an_unanswered_subtask_is_not_ready_and_jev_is_not_asked():
    verdict = ship_check(_state(answered=2, unanswered=1))["ship"]
    assert verdict == ShipVerdict(ready=False, confidence=None, reason="unanswered")
    assert FakeJev.calls == []


def test_a_draft_with_nothing_answered_is_not_ready_either():
    verdict = ship_check(_state(answered=0, unanswered=0))["ship"]
    assert verdict.reason == "unanswered"
    assert FakeJev.calls == []


def test_with_jev_down_the_draft_is_not_called_ready(monkeypatch):
    monkeypatch.setattr(jev, "TypeSafeClient", DownJev)
    verdict = ship_check(_state())["ship"]
    assert verdict == ShipVerdict(ready=False, confidence=None, reason="fallback")


def test_the_verdict_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger=ship_module.logger.name):
        ship_check(_state())
    assert "ship_check ready=True reason=jev confidence=0.9" in caplog.text
