from uuid import uuid4

from research_assistant.main import print_summary
from research_assistant.schemas.findings import Finding, Review, Source
from research_assistant.schemas.plan import Plan, Subtask
from research_assistant.schemas.retry import RetryDecision

SOURCE = Source(url="https://a.com", title="Page A", content="Fact from A.")


def _result():
    first, second, third = (Subtask(description=f"question {i}") for i in (1, 2, 3))
    findings = [
        # deliberately not in plan order: the researchers finish in any order
        Finding(subtask_id=third.id, status="failed", error="tavily is down"),
        Finding(subtask_id=first.id, status="found", answer="x", sources=[SOURCE]),
        Finding(subtask_id=second.id, status="no_results"),
    ]
    return {"plan": Plan(subtasks=[first, second, third]), "findings": findings}


def test_summary_lists_the_plan_in_order_and_matches_findings_by_id(capsys):
    print_summary(_result())
    out = capsys.readouterr().out
    assert "1. question 1" in out
    assert "[found] question 1" in out
    assert "[no_results] question 2" in out
    assert "[failed] question 3" in out
    assert "error: tavily is down" in out


def test_summary_marks_a_subtask_with_no_finding_instead_of_crashing(capsys):
    result = _result()
    result["findings"] = []
    print_summary(result)
    assert "[missing] question 2" in capsys.readouterr().out


def test_summary_ignores_findings_that_do_not_belong_to_the_plan(capsys):
    result = _result()
    result["findings"].append(Finding(subtask_id=uuid4(), status="no_results"))
    print_summary(result)  # must not raise
    assert "[found] question 1" in capsys.readouterr().out


def _review(subtask_id, **overrides):
    fields = {
        "subtask_id": subtask_id,
        "attempt": 1,
        "kept_urls": ["https://a.com"],
        "dropped_urls": [],
        "source_scores": {"https://a.com": 1.83},
        "addresses_subtask": True,
        "used_fallback": False,
    }
    return Review(**{**fields, **overrides})


def test_summary_shows_kept_and_dropped_sources_and_the_answer_verdict(capsys):
    result = _result()
    first = result["plan"].subtasks[0]
    result["reviews"] = [
        _review(
            first.id,
            dropped_urls=["https://youtube.com/watch"],
            source_scores={"https://a.com": 1.83, "https://youtube.com/watch": 0.2},
            addresses_subtask=False,
        )
    ]
    print_summary(result)
    out = capsys.readouterr().out
    assert "sources: 1 kept, 1 dropped; answers the subtask: NO" in out
    assert "kept    1.83  https://a.com" in out
    assert "dropped 0.20  https://youtube.com/watch" in out
    assert "Jev was unreachable" not in out


def test_summary_says_when_jev_defaults_were_used(capsys):
    result = _result()
    result["reviews"] = [
        _review(
            result["plan"].subtasks[0].id,
            used_fallback=True,
            source_scores={"https://a.com": None},
        )
    ]
    print_summary(result)
    out = capsys.readouterr().out
    assert "kept     n/a  https://a.com" in out  # a default is never shown as a score
    assert "answers the subtask: yes" in out
    assert "note: Jev was unreachable, defaults were used" in out


def test_summary_without_reviews_prints_no_review_lines(capsys):
    print_summary(_result())  # no "reviews" key, as in a run saved before Phase 4
    out = capsys.readouterr().out
    assert "sources:" not in out


def _decision(subtask_id, action="escalate", reason="cap"):
    return RetryDecision(
        subtask_id=subtask_id,
        attempt=2,
        action=action,
        reason=reason,
        confidence=None,
        exclude_domains=[],
    )


def test_summary_flags_an_escalated_subtask_and_says_why(capsys):
    result = _result()
    second = result["plan"].subtasks[1]  # its finding is no_results: no review
    result["decisions"] = [_decision(second.id, reason="cap")]
    print_summary(result)
    out = capsys.readouterr().out
    assert "NEEDS A HUMAN: escalated (retry limit reached)" in out


def test_summary_says_how_many_attempts_a_retried_subtask_took(capsys):
    result = _result()
    first = result["plan"].subtasks[0]
    result["findings"].append(
        Finding(subtask_id=first.id, status="found", answer="y", sources=[SOURCE])
    )
    print_summary(result)
    assert "attempts: 2 (the newest is shown)" in capsys.readouterr().out


def test_summary_stays_quiet_about_a_retry_that_was_chosen_but_is_not_an_escalation(
    capsys,
):
    result = _result()
    result["decisions"] = [
        _decision(result["plan"].subtasks[0].id, action="retry_same", reason="jev")
    ]
    print_summary(result)
    assert "NEEDS A HUMAN" not in capsys.readouterr().out
