import logging
from types import SimpleNamespace
from typing import ClassVar

import httpx2
import pytest
from typesafe_sdk import (
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from research_assistant.decisions import jev
from research_assistant.decisions.jev import (
    Decision,
    ask_boolean,
    ask_choice,
    ask_score,
)


class FakeClient:
    """Stands in for TypeSafeClient so no network call or API key is needed."""

    probability: ClassVar[float] = 0.9
    score: ClassVar[float] = 1.7
    score_confidence: ClassVar[float] = 0.55
    choice: ClassVar[str] = "retry_different"
    choice_confidence: ClassVar[float] = 0.8
    error: ClassVar[Exception | None] = None
    last_init: ClassVar[dict] = {}
    last_call: ClassVar[dict] = {}

    def __init__(self, **kwargs):
        FakeClient.last_init = kwargs

    def system_one(self, state, questions):
        FakeClient.last_call = {"state": state, "questions": questions}
        if FakeClient.error:
            raise FakeClient.error
        return SimpleNamespace(
            model="jev-1.13.0",
            nouls={"q": SimpleNamespace(noul=FakeClient.probability)},
            scores={
                "q": SimpleNamespace(
                    score=FakeClient.score, confidence=FakeClient.score_confidence
                )
            },
            choices={
                "q": SimpleNamespace(
                    choice=FakeClient.choice,
                    confidence=FakeClient.choice_confidence,
                    probabilities={FakeClient.choice: 0.8, "other": 0.2},
                )
            },
        )


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    FakeClient.probability = 0.9
    FakeClient.score = 1.7
    FakeClient.score_confidence = 0.55
    FakeClient.choice = "retry_different"
    FakeClient.choice_confidence = 0.8
    FakeClient.error = None
    FakeClient.last_init = {}
    FakeClient.last_call = {}
    monkeypatch.setattr(jev, "TypeSafeClient", FakeClient)


def _api_error(cls, status):
    return cls(status, {}, httpx2.Headers())


def test_yes_answer():
    FakeClient.probability = 0.9
    decision = ask_boolean("some state", "Is it good?", default=False)
    assert decision == Decision(
        value=True, confidence=0.9, fallback=False, model="jev-1.13.0"
    )


def test_no_answer_confidence_is_in_the_chosen_direction():
    FakeClient.probability = 0.1
    decision = ask_boolean("some state", "Is it good?", default=True)
    assert decision.value is False
    assert decision.confidence == pytest.approx(0.9)
    assert decision.fallback is False


def test_threshold_is_inclusive_and_configurable():
    FakeClient.probability = 0.7
    assert ask_boolean("s", "q?", default=False, threshold=0.7).value is True
    assert ask_boolean("s", "q?", default=True, threshold=0.71).value is False


def test_request_carries_state_and_question_and_the_pinned_model():
    ask_boolean("the state text", "Is it good?", default=False)
    assert FakeClient.last_init["model"] == jev.JEV_MODEL
    assert FakeClient.last_init["retry"].max_retries == jev.JEV_MAX_RETRIES
    assert FakeClient.last_call["state"] == "the state text"
    question = FakeClient.last_call["questions"]["q"]
    assert question.type == "noul"
    assert question.instructions == "Is it good?"


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize(
    "error",
    [
        TypeSafeAPITimeoutError(10),
        _api_error(TypeSafeRateLimitError, 429),
        _api_error(TypeSafeAuthenticationError, 401),
    ],
    ids=["timeout", "rate_limit", "bad_key"],
)
def test_sdk_failure_returns_the_default_marked_as_fallback(error, default):
    FakeClient.error = error
    decision = ask_boolean("s", "q?", default=default)
    assert decision == Decision(
        value=default, confidence=None, fallback=True, model=None
    )


def test_non_sdk_errors_are_not_swallowed():
    # Only TypeSafeError is caught: anything else is our own bug and must surface.
    FakeClient.error = KeyError("boom")
    with pytest.raises(KeyError):
        ask_boolean("s", "q?", default=False)


def test_our_own_4xx_is_logged_as_error_but_an_outage_as_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=jev.logger.name)

    FakeClient.error = _api_error(TypeSafeAuthenticationError, 401)
    ask_boolean("s", "q?", default=False)
    FakeClient.error = _api_error(TypeSafeRateLimitError, 429)
    ask_boolean("s", "q?", default=False)

    levels = [r.levelno for r in caplog.records]
    assert levels == [logging.ERROR, logging.WARNING]


def test_log_has_the_decision_but_not_the_state_text(caplog):
    caplog.set_level(logging.INFO, logger=jev.logger.name)
    secret = "PAGE TEXT THAT MUST NOT BE LOGGED"
    ask_boolean(secret, "Is it good?", default=False)

    (record,) = caplog.records
    message = record.getMessage()
    assert "Is it good?" in message
    assert "p=0.900" in message
    assert "jev-1.13.0" in message
    assert f"state_chars={len(secret)}" in message
    assert secret not in message


LEVELS = ["useless", "partly relevant", "directly relevant"]


def test_score_returns_the_expected_level_and_confidence():
    decision = ask_score("some state", "How useful is it?", LEVELS, default=0.0)
    assert decision == Decision(
        value=1.7, confidence=0.55, fallback=False, model="jev-1.13.0"
    )


def test_score_request_carries_the_levels_in_order():
    ask_score("the state text", "How useful is it?", LEVELS, default=0.0)
    assert FakeClient.last_call["state"] == "the state text"
    question = FakeClient.last_call["questions"]["q"]
    assert question.type == "score"
    assert question.instructions == "How useful is it?"
    assert list(question.criteria) == LEVELS


@pytest.mark.parametrize("default", [0.0, 2.0])
def test_score_failure_returns_the_default_marked_as_fallback(default):
    FakeClient.error = TypeSafeAPITimeoutError(10)
    decision = ask_score("s", "q?", LEVELS, default=default)
    assert decision == Decision(
        value=default, confidence=None, fallback=True, model=None
    )


def test_score_needs_at_least_two_levels():
    # Our own bug, so it raises instead of falling back.
    with pytest.raises(ValueError):
        ask_score("s", "q?", ["only one"], default=0.0)


def test_score_log_has_the_result_but_not_the_state_text(caplog):
    caplog.set_level(logging.INFO, logger=jev.logger.name)
    secret = "PAGE TEXT THAT MUST NOT BE LOGGED"
    ask_score(secret, "How useful is it?", LEVELS, default=0.0)

    (record,) = caplog.records
    message = record.getMessage()
    assert "score=1.70" in message
    assert "levels=3" in message
    assert f"state_chars={len(secret)}" in message
    assert secret not in message


OPTIONS = {
    "retry_same": "Search again with the same query.",
    "retry_different": "Search again, skipping the sites that were rejected.",
    "escalate": "Stop and ask a human.",
}


def test_choice_returns_the_chosen_option_name_and_confidence():
    decision = ask_choice("some state", "What next?", OPTIONS, default="escalate")
    assert decision == Decision(
        value="retry_different", confidence=0.8, fallback=False, model="jev-1.13.0"
    )
    assert isinstance(decision.value, str)


def test_choice_request_carries_the_options_with_their_descriptions():
    ask_choice("the state text", "What next?", OPTIONS, default="escalate")
    assert FakeClient.last_call["state"] == "the state text"
    question = FakeClient.last_call["questions"]["q"]
    assert question.type == "choice"
    assert question.instructions == "What next?"
    assert dict(question.criteria) == OPTIONS


@pytest.mark.parametrize("default", ["escalate", "retry_same"])
def test_choice_failure_returns_the_default_marked_as_fallback(default):
    FakeClient.error = TypeSafeAPITimeoutError(10)
    decision = ask_choice("s", "q?", OPTIONS, default=default)
    assert decision == Decision(
        value=default, confidence=None, fallback=True, model=None
    )


def test_choice_needs_at_least_two_options():
    with pytest.raises(ValueError, match="at least two"):
        ask_choice("s", "q?", {"only": "one"}, default="only")


def test_choice_default_must_be_one_of_the_options():
    # Our own bug, so it raises instead of falling back, and before any Jev call.
    with pytest.raises(ValueError, match="not one of the options"):
        ask_choice("s", "q?", OPTIONS, default="give_up")
    assert FakeClient.last_call == {}


def test_choice_log_has_the_result_but_not_the_state_text(caplog):
    caplog.set_level(logging.INFO, logger=jev.logger.name)
    secret = "PAGE TEXT THAT MUST NOT BE LOGGED"
    ask_choice(secret, "What next?", OPTIONS, default="escalate")

    (record,) = caplog.records
    message = record.getMessage()
    assert "choice=retry_different" in message
    assert "confidence=0.800" in message
    assert "'other': 0.2" in message
    assert f"state_chars={len(secret)}" in message
    assert secret not in message
