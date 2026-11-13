from uuid import uuid4

from research_assistant.main import print_summary
from research_assistant.schemas.findings import Finding, Source
from research_assistant.schemas.plan import Plan, Subtask

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
