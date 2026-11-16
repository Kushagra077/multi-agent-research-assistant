"""The decide node (retry or escalate) and the retry loop in the real graph.

Jev, the planner, the researcher LLM and web search are all faked, so nothing touches
the network or uses any quota.
"""

import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typesafe_sdk import TypeSafeAPIConnectionError

from research_assistant.decisions import jev
from research_assistant.graph.build import build_graph
from research_assistant.graph.nodes import decide as decide_module
from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.graph.nodes.decide import MAX_ATTEMPTS, decide
from research_assistant.schemas.findings import (
    Finding,
    ResearchSummary,
    Review,
    Source,
)
from research_assistant.schemas.plan import Plan, Subtask
from research_assistant.schemas.retry import RetryDecision

GOOD = Source(url="https://good.com/a", title="Good", content="Real facts.")
SHOP = Source(url="https://www.amazon.com/dp/1", title="Shop", content="Buy now.")
BLOG = Source(url="https://lessons.studio.com/blog", title="Blog", content="Sales.")


class FakeJev:
    """Stands in for TypeSafeClient. Pages are scored by url, booleans use `noul`, and a
    choice question returns `choice`."""

    scores: ClassVar[dict[str, float]] = {}
    noul: ClassVar[float] = 0.9
    choice: ClassVar[str] = "retry_different"
    calls: ClassVar[list[tuple[str, str]]] = []  # (question type, state)

    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        kind = questions["q"].type
        FakeJev.calls.append((kind, state))
        if kind == "noul":
            return SimpleNamespace(
                model="jev-1.13.0", nouls={"q": SimpleNamespace(noul=FakeJev.noul)}
            )
        if kind == "choice":
            return SimpleNamespace(
                model="jev-1.13.0",
                choices={
                    "q": SimpleNamespace(
                        choice=FakeJev.choice,
                        confidence=0.8,
                        probabilities={FakeJev.choice: 0.8},
                    )
                },
            )
        score = next((v for url, v in FakeJev.scores.items() if url in state), 2.0)
        return SimpleNamespace(
            model="jev-1.13.0",
            scores={"q": SimpleNamespace(score=score, confidence=0.8)},
        )


class DownJev:
    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        raise TypeSafeAPIConnectionError("jev is down")


@pytest.fixture(autouse=True)
def fake_jev(monkeypatch):
    FakeJev.scores = {"amazon.com": 0.5, "lessons.studio.com": 0.9, "good.com": 1.9}
    FakeJev.noul = 0.9
    FakeJev.choice = "retry_different"
    FakeJev.calls = []
    monkeypatch.setattr(jev, "TypeSafeClient", FakeJev)


def _choice_calls():
    return [state for kind, state in FakeJev.calls if kind == "choice"]


# --- the decide node on its own --------------------------------------------


def _review(subtask, attempt=1, *, dropped=(), kept=(), addresses=False):
    scores = {url: 0.5 for url in dropped} | {url: 1.9 for url in kept}
    return Review(
        subtask_id=subtask.id,
        attempt=attempt,
        kept_urls=list(kept),
        dropped_urls=list(dropped),
        source_scores=scores,
        addresses_subtask=addresses,
        used_fallback=False,
    )


def _found(subtask, *sources):
    return Finding(
        subtask_id=subtask.id,
        status="found",
        answer="An answer.",
        sources=list(sources),
    )


def _state(subtask, findings, reviews=(), decisions=()):
    return {
        "question": "q",
        "plan": Plan(subtasks=[subtask]),
        "findings": list(findings),
        "reviews": list(reviews),
        "decisions": list(decisions),
    }


def test_a_subtask_that_passed_gets_no_decision_and_no_jev_call():
    subtask = Subtask(description="What is X?")
    state = _state(
        subtask,
        [_found(subtask, GOOD)],
        [_review(subtask, kept=[GOOD.url], addresses=True)],
    )
    assert decide(state) == {"decisions": []}
    assert FakeJev.calls == []


def test_a_rejected_answer_asks_jev_and_records_its_choice_with_the_sites_to_skip():
    subtask = Subtask(description="What is X?")
    state = _state(
        subtask,
        [_found(subtask, SHOP, BLOG)],
        [_review(subtask, dropped=[SHOP.url, BLOG.url])],
    )
    (decision,) = decide(state)["decisions"]
    assert decision == RetryDecision(
        subtask_id=subtask.id,
        attempt=1,
        action="retry_different",
        reason="jev",
        confidence=0.8,
        exclude_domains=["amazon.com", "lessons.studio.com"],  # www. dropped
    )


def test_jev_is_shown_the_subtask_the_attempt_and_every_source_with_its_score():
    subtask = Subtask(description="What is X?")
    state = _state(
        subtask,
        [_found(subtask, SHOP, GOOD)],
        [_review(subtask, dropped=[SHOP.url], kept=[GOOD.url])],
    )
    decide(state)
    (text,) = _choice_calls()
    assert "Subtask: What is X?" in text
    assert f"Attempt 1 of at most {MAX_ATTEMPTS}" in text
    assert "The answer addresses the subtask: no" in text
    assert f"Source rejected, scored 0.50 of 2: {SHOP.url}" in text
    assert f"Source kept, scored 1.90 of 2: {GOOD.url}" in text


def test_retry_same_skips_nothing():
    FakeJev.choice = "retry_same"
    subtask = Subtask(description="What is X?")
    state = _state(
        subtask, [_found(subtask, SHOP)], [_review(subtask, dropped=[SHOP.url])]
    )
    (decision,) = decide(state)["decisions"]
    assert decision.action == "retry_same"
    assert decision.exclude_domains == []


@pytest.mark.parametrize("status", ["no_results", "failed"])
def test_a_finding_without_an_answer_is_decided_on_too(status):
    subtask = Subtask(description="What is X?")
    finding = Finding(
        subtask_id=subtask.id,
        status=status,
        error="tavily is down" if status == "failed" else None,
    )
    (decision,) = decide(_state(subtask, [finding]))["decisions"]
    assert decision.reason == "jev"
    (text,) = _choice_calls()
    assert f"Outcome of this attempt: {status}" in text


def test_the_retry_limit_escalates_without_asking_jev():
    subtask = Subtask(description="What is X?")
    findings = [_found(subtask, SHOP) for _ in range(MAX_ATTEMPTS)]
    reviews = [
        _review(subtask, a, dropped=[SHOP.url]) for a in range(1, MAX_ATTEMPTS + 1)
    ]
    (decision,) = decide(_state(subtask, findings, reviews))["decisions"]
    assert decision.action == "escalate"
    assert decision.reason == "cap"
    assert decision.attempt == MAX_ATTEMPTS
    assert decision.confidence is None
    assert FakeJev.calls == []


def test_with_jev_down_the_subtask_escalates_instead_of_retrying_blindly(monkeypatch):
    monkeypatch.setattr(jev, "TypeSafeClient", DownJev)
    subtask = Subtask(description="What is X?")
    state = _state(
        subtask, [_found(subtask, SHOP)], [_review(subtask, dropped=[SHOP.url])]
    )
    (decision,) = decide(state)["decisions"]
    assert decision.action == "escalate"
    assert decision.reason == "fallback"
    assert decision.exclude_domains == []


def test_an_attempt_that_was_already_decided_costs_no_jev_call():
    subtask = Subtask(description="What is X?")
    earlier = RetryDecision(
        subtask_id=subtask.id,
        attempt=1,
        action="escalate",
        reason="jev",
        confidence=0.9,
        exclude_domains=[],
    )
    state = _state(
        subtask,
        [_found(subtask, SHOP)],
        [_review(subtask, dropped=[SHOP.url])],
        [earlier],
    )
    assert decide(state) == {"decisions": []}
    assert FakeJev.calls == []


def test_the_second_retry_skips_the_sites_rejected_on_every_earlier_attempt():
    subtask = Subtask(description="What is X?")
    state = _state(
        subtask,
        [_found(subtask, SHOP), _found(subtask, BLOG)],
        [
            _review(subtask, 1, dropped=[SHOP.url]),
            _review(subtask, 2, dropped=[BLOG.url]),
        ],
    )
    # Pretend the limit is higher, so attempt 2 is still allowed to ask Jev.
    decide_module.MAX_ATTEMPTS, original = 3, decide_module.MAX_ATTEMPTS
    try:
        (decision,) = decide(state)["decisions"]
    finally:
        decide_module.MAX_ATTEMPTS = original
    assert decision.attempt == 2
    assert decision.exclude_domains == ["amazon.com", "lessons.studio.com"]


def test_the_decision_is_logged(caplog):
    subtask = Subtask(description="What is X?")
    state = _state(
        subtask, [_found(subtask, SHOP)], [_review(subtask, dropped=[SHOP.url])]
    )
    with caplog.at_level(logging.INFO, logger=decide_module.logger.name):
        decide(state)
    assert "action=retry_different reason=jev confidence=0.8" in caplog.text
    assert "exclude=['amazon.com']" in caplog.text


# --- the loop in the real graph --------------------------------------------


class FakePlannerLLM:
    plan: ClassVar[Plan]

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return FakePlannerLLM.plan


class FakeResearcherLLM:
    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return ResearchSummary(answer="A fake answer.")


class FakeSearch:
    """Returns the sources in `script` one search at a time and records every call."""

    script: ClassVar[list[list[Source]]] = []
    calls: ClassVar[list[tuple[str, list[str] | None]]] = []

    @staticmethod
    def search(query, exclude_domains=None):
        FakeSearch.calls.append((query, exclude_domains))
        return FakeSearch.script[len(FakeSearch.calls) - 1]


@pytest.fixture
def graph(monkeypatch):
    subtask = Subtask(description="question 0")
    FakePlannerLLM.plan = Plan(subtasks=[subtask])
    FakeSearch.script = []
    FakeSearch.calls = []
    monkeypatch.setattr(planner_module, "get_planner_llm", lambda: FakePlannerLLM())
    monkeypatch.setattr(
        researcher_module, "get_researcher_llm", lambda: FakeResearcherLLM()
    )
    monkeypatch.setattr(researcher_module, "search_web", FakeSearch.search)
    return build_graph(InMemorySaver())


def _run(graph):
    return graph.invoke({"question": "q"}, {"configurable": {"thread_id": "t"}})


def test_a_rejected_first_attempt_is_retried_skipping_the_rejected_sites(graph):
    FakeSearch.script = [[SHOP, BLOG], [GOOD]]
    result = _run(graph)

    assert [exclude for _, exclude in FakeSearch.calls] == [
        [],
        ["amazon.com", "lessons.studio.com"],
    ]
    # Both attempts are kept as history; the newest passed.
    assert len(result["findings"]) == 2
    assert [r.attempt for r in result["reviews"]] == [1, 2]
    assert result["reviews"][0].addresses_subtask is False
    assert result["reviews"][1].addresses_subtask is True
    # Only the first attempt needed a decision, and nothing was left unresolved.
    (decision,) = result["decisions"]
    assert (decision.attempt, decision.action) == (1, "retry_different")


def test_a_subtask_that_never_recovers_stops_at_the_limit_and_escalates(graph):
    FakeSearch.script = [[SHOP], [BLOG]]
    result = _run(graph)

    assert len(FakeSearch.calls) == MAX_ATTEMPTS
    actions = [(d.attempt, d.action, d.reason) for d in result["decisions"]]
    assert actions == [(1, "retry_different", "jev"), (2, "escalate", "cap")]


def test_when_everything_passes_the_graph_ends_without_a_decision(graph):
    FakeSearch.script = [[GOOD]]
    result = _run(graph)
    assert result["decisions"] == []
    assert len(FakeSearch.calls) == 1


def test_an_escalation_ends_the_run_without_another_search(graph):
    FakeJev.choice = "escalate"
    FakeSearch.script = [[SHOP]]
    result = _run(graph)
    assert len(FakeSearch.calls) == 1
    (decision,) = result["decisions"]
    assert decision.action == "escalate"


def test_retry_same_searches_again_with_nothing_excluded(graph):
    FakeJev.choice = "retry_same"
    FakeSearch.script = [[SHOP], [GOOD]]
    _run(graph)
    assert [exclude for _, exclude in FakeSearch.calls] == [[], []]
