import logging
from types import SimpleNamespace
from typing import ClassVar
from uuid import uuid4

import pytest
from typesafe_sdk import TypeSafeAPIConnectionError

from research_assistant.decisions import jev
from research_assistant.graph.nodes import merge as merge_module
from research_assistant.graph.nodes.merge import MIN_SOURCE_SCORE, merge
from research_assistant.schemas.findings import Finding, Review, Source
from research_assistant.schemas.plan import Plan, Subtask

GOOD = Source(url="https://good.com", title="Good page", content="Real facts.")
OKAY = Source(url="https://okay.com", title="Okay page", content="Some facts.")
BAD = Source(url="https://youtube.com/watch", title="A video", content="Intro music.")


class FakeJev:
    """Stands in for TypeSafeClient. A page is scored by its URL; booleans use `noul`."""

    scores: ClassVar[dict[str, float]] = {}
    noul: ClassVar[float] = 0.9
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
        score = next((v for url, v in FakeJev.scores.items() if url in state), 2.0)
        return SimpleNamespace(
            model="jev-1.13.0",
            scores={"q": SimpleNamespace(score=score, confidence=0.8)},
        )


@pytest.fixture(autouse=True)
def fake_jev(monkeypatch):
    FakeJev.scores = {"youtube.com": 0.2, "okay.com": 1.0, "good.com": 1.9}
    FakeJev.noul = 0.9
    FakeJev.calls = []
    monkeypatch.setattr(jev, "TypeSafeClient", FakeJev)


def _state(*findings: Finding, subtask: Subtask | None = None) -> dict:
    subtask = subtask or Subtask(description="What is X?")
    return {
        "question": "q",
        "plan": Plan(subtasks=[subtask]),
        "findings": list(findings),
    }


def _found(subtask: Subtask, *sources: Source) -> Finding:
    return Finding(
        subtask_id=subtask.id, status="found", answer="X is Y.", sources=list(sources)
    )


def test_weak_sources_are_dropped_and_good_ones_kept():
    subtask = Subtask(description="What is X?")
    state = _state(_found(subtask, GOOD, BAD), subtask=subtask)
    (review,) = merge(state)["reviews"]
    assert review == Review(
        subtask_id=subtask.id,
        attempt=1,
        kept_urls=[GOOD.url],
        dropped_urls=[BAD.url],
        source_scores={GOOD.url: 1.9, BAD.url: 0.2},
        addresses_subtask=True,
        used_fallback=False,
    )


def test_a_source_exactly_at_the_cutoff_is_kept():
    assert FakeJev.scores["okay.com"] == MIN_SOURCE_SCORE
    subtask = Subtask(description="What is X?")
    (review,) = merge(_state(_found(subtask, OKAY), subtask=subtask))["reviews"]
    assert review.kept_urls == [OKAY.url]


def test_findings_are_not_changed_only_reviews_are_returned():
    subtask = Subtask(description="What is X?")
    state = _state(_found(subtask, GOOD, BAD), subtask=subtask)
    assert list(merge(state)) == ["reviews"]
    assert state["findings"][0].sources == [GOOD, BAD]


def test_jev_sees_the_subtask_and_the_page_but_the_answer_check_sees_the_answer():
    subtask = Subtask(description="What is X?")
    merge(_state(_found(subtask, GOOD), subtask=subtask))
    (score_kind, score_state), (bool_kind, bool_state) = FakeJev.calls
    assert score_kind == "score"
    assert "What is X?" in score_state
    assert GOOD.url in score_state
    assert GOOD.content in score_state
    assert bool_kind == "noul"
    assert "What is X?" in bool_state
    assert "X is Y." in bool_state


def test_when_the_answer_check_says_no_the_review_says_so():
    FakeJev.noul = 0.1
    subtask = Subtask(description="What is X?")
    (review,) = merge(_state(_found(subtask, GOOD), subtask=subtask))["reviews"]
    assert review.addresses_subtask is False
    assert review.kept_urls == [GOOD.url]


def test_when_every_source_is_dropped_the_answer_fails_without_asking_jev():
    subtask = Subtask(description="What is X?")
    (review,) = merge(_state(_found(subtask, BAD), subtask=subtask))["reviews"]
    assert review.kept_urls == []
    assert review.dropped_urls == [BAD.url]
    assert review.addresses_subtask is False
    assert [kind for kind, _ in FakeJev.calls] == ["score"]  # no boolean call


def test_no_results_and_failed_findings_get_no_review():
    subtask = Subtask(description="What is X?")
    state = _state(
        Finding(subtask_id=subtask.id, status="no_results"),
        Finding(subtask_id=subtask.id, status="failed", error="boom"),
        subtask=subtask,
    )
    assert merge(state) == {"reviews": []}
    assert FakeJev.calls == []


def test_a_finding_that_is_not_in_the_plan_is_skipped(caplog):
    stray = Finding(subtask_id=uuid4(), status="found", answer="x", sources=[GOOD])
    with caplog.at_level(logging.WARNING, logger=merge_module.logger.name):
        assert merge(_state(stray)) == {"reviews": []}
    assert "not in plan" in caplog.text


class DownJev:
    """Every call fails like an outage."""

    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        raise TypeSafeAPIConnectionError("jev is down")


def test_with_jev_down_everything_passes_and_the_review_says_so(monkeypatch):
    monkeypatch.setattr(jev, "TypeSafeClient", DownJev)
    subtask = Subtask(description="What is X?")
    (review,) = merge(_state(_found(subtask, GOOD, BAD), subtask=subtask))["reviews"]
    assert review.kept_urls == [GOOD.url, BAD.url]
    assert review.dropped_urls == []
    assert review.source_scores == {GOOD.url: None, BAD.url: None}
    assert review.addresses_subtask is True
    assert review.used_fallback is True


def test_merge_logs_a_count_per_status(caplog):
    subtask = Subtask(description="What is X?")
    state = _state(
        _found(subtask, GOOD),
        Finding(subtask_id=uuid4(), status="no_results"),
        Finding(subtask_id=uuid4(), status="failed", error="boom"),
        subtask=subtask,
    )
    with caplog.at_level(logging.INFO, logger=merge_module.logger.name):
        merge(state)
    messages = [r.getMessage() for r in caplog.records]
    assert "merge findings=3 found=1 no_results=1 failed=1" in messages
    assert any("merge reviewed" in m and "kept=1 dropped=0" in m for m in messages)


def test_merge_with_no_findings_logs_zeros_and_returns_no_reviews(caplog):
    state = {"question": "q", "plan": Plan(subtasks=[Subtask(description="d")])}
    with caplog.at_level(logging.INFO, logger=merge_module.logger.name):
        assert merge(state) == {"reviews": []}
    assert "findings=0 found=0 no_results=0 failed=0" in caplog.text


# --- retries: merge runs again and must only review what is new ---------------


def _review(subtask, attempt=1, **overrides):
    fields = {
        "subtask_id": subtask.id,
        "attempt": attempt,
        "kept_urls": [GOOD.url],
        "dropped_urls": [],
        "source_scores": {GOOD.url: 1.9},
        "addresses_subtask": True,
        "used_fallback": False,
    }
    return Review(**{**fields, **overrides})


def test_an_attempt_that_is_already_reviewed_costs_no_jev_calls():
    subtask = Subtask(description="What is X?")
    state = _state(_found(subtask, GOOD), subtask=subtask)
    state["reviews"] = [_review(subtask, attempt=1)]
    assert merge(state) == {"reviews": []}
    assert FakeJev.calls == []


def test_a_retry_is_reviewed_as_the_next_attempt_using_only_the_newest_finding():
    subtask = Subtask(description="What is X?")
    first = _found(subtask, BAD)  # attempt 1: the gate rejected its only source
    retry = _found(subtask, GOOD)  # attempt 2
    state = _state(first, retry, subtask=subtask)
    state["reviews"] = [
        _review(
            subtask,
            attempt=1,
            kept_urls=[],
            dropped_urls=[BAD.url],
            source_scores={BAD.url: 0.2},
            addresses_subtask=False,
        )
    ]
    (review,) = merge(state)["reviews"]
    assert review.attempt == 2
    assert review.kept_urls == [GOOD.url]
    assert review.addresses_subtask is True
    # Jev saw only the retry's page, never the old attempt's.
    assert [kind for kind, _ in FakeJev.calls] == ["score", "noul"]
    assert GOOD.url in FakeJev.calls[0][1]
    assert not any(BAD.url in text for _, text in FakeJev.calls)


def test_only_the_subtask_that_was_retried_is_reviewed_again():
    done = Subtask(description="What is X?")
    retried = Subtask(description="What is Z?")
    state = {
        "question": "q",
        "plan": Plan(subtasks=[done, retried]),
        "findings": [
            _found(done, GOOD),
            _found(retried, BAD),
            _found(retried, OKAY),
        ],
        "reviews": [_review(done), _review(retried, attempt=1, kept_urls=[])],
    }
    (review,) = merge(state)["reviews"]
    assert review.subtask_id == retried.id
    assert review.attempt == 2


def test_the_status_counts_describe_the_newest_finding_of_each_subtask(caplog):
    subtask = Subtask(description="What is X?")
    state = _state(
        Finding(subtask_id=subtask.id, status="failed", error="boom"),
        _found(subtask, GOOD),
        subtask=subtask,
    )
    with caplog.at_level(logging.INFO, logger=merge_module.logger.name):
        merge(state)
    assert "merge findings=1 found=1 no_results=0 failed=0" in caplog.text


def test_a_retry_that_failed_gets_no_review_and_keeps_the_old_one_as_history():
    subtask = Subtask(description="What is X?")
    state = _state(
        _found(subtask, BAD),
        Finding(subtask_id=subtask.id, status="no_results"),
        subtask=subtask,
    )
    state["reviews"] = [_review(subtask, attempt=1)]
    assert merge(state) == {"reviews": []}
    assert FakeJev.calls == []


# --- sources with too little text are dropped by a rule, without asking Jev -----


def test_a_page_with_too_little_text_is_dropped_without_a_jev_call(monkeypatch):
    monkeypatch.setattr(merge_module, "MIN_SOURCE_CHARS", 300)
    thin = Source(url="https://instagram.com/p/1", title="Post", content="x" * 299)
    full = Source(url="https://good.com", title="Good", content="x" * 300)
    subtask = Subtask(description="What is X?")
    (review,) = merge(_state(_found(subtask, thin, full), subtask=subtask))["reviews"]
    assert review.dropped_urls == [thin.url]
    assert review.kept_urls == [full.url]  # exactly 300 characters is enough
    assert review.source_scores == {thin.url: None, full.url: 1.9}
    # Jev was asked about the full page only (one score), then about the answer.
    assert [kind for kind, _ in FakeJev.calls] == ["score", "noul"]
    assert thin.url not in FakeJev.calls[0][1]


def test_dropping_a_thin_page_is_not_reported_as_a_jev_outage(monkeypatch):
    monkeypatch.setattr(merge_module, "MIN_SOURCE_CHARS", 300)
    thin = Source(url="https://instagram.com/p/1", title="Post", content="x")
    subtask = Subtask(description="What is X?")
    (review,) = merge(_state(_found(subtask, thin), subtask=subtask))["reviews"]
    assert review.used_fallback is False
    assert review.addresses_subtask is False  # nothing credible is left
    assert FakeJev.calls == []


def test_the_log_says_a_page_was_dropped_for_being_too_short(monkeypatch, caplog):
    monkeypatch.setattr(merge_module, "MIN_SOURCE_CHARS", 300)
    thin = Source(url="https://instagram.com/p/1", title="Post", content="x" * 50)
    subtask = Subtask(description="What is X?")
    with caplog.at_level(logging.INFO, logger=merge_module.logger.name):
        merge(_state(_found(subtask, thin), subtask=subtask))
    assert "too_short chars=50" in caplog.text
