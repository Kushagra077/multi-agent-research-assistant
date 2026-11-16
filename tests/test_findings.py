from uuid import uuid4

import pytest
from pydantic import ValidationError

from research_assistant.schemas.findings import (
    Finding,
    Review,
    Source,
    attempts_by_subtask,
    latest_per_subtask,
)


@pytest.fixture
def source() -> Source:
    return Source(url="https://example.com", title="Example", content="Some text.")


def test_found_with_answer_and_sources_is_valid(source):
    finding = Finding(
        subtask_id=uuid4(), status="found", answer="An answer.", sources=[source]
    )
    assert finding.status == "found"


def test_found_without_answer_is_rejected(source):
    with pytest.raises(ValidationError):
        Finding(subtask_id=uuid4(), status="found", sources=[source])


def test_found_without_sources_is_rejected():
    with pytest.raises(ValidationError):
        Finding(subtask_id=uuid4(), status="found", answer="An answer.")


def test_found_with_empty_answer_is_rejected(source):
    with pytest.raises(ValidationError):
        Finding(subtask_id=uuid4(), status="found", answer="", sources=[source])


def test_failed_without_error_is_rejected():
    with pytest.raises(ValidationError):
        Finding(subtask_id=uuid4(), status="failed")


def test_failed_with_error_is_valid():
    finding = Finding(subtask_id=uuid4(), status="failed", error="Search timed out.")
    assert finding.error == "Search timed out."


def test_no_results_needs_nothing_else():
    finding = Finding(subtask_id=uuid4(), status="no_results")
    assert finding.answer is None
    assert finding.sources == []


def test_unknown_status_is_rejected():
    with pytest.raises(ValidationError):
        Finding(subtask_id=uuid4(), status="banana")


def test_finding_survives_json_round_trip(source):
    original = Finding(
        subtask_id=uuid4(), status="found", answer="An answer.", sources=[source]
    )
    restored = Finding.model_validate(original.model_dump(mode="json"))
    assert restored == original


def test_findings_do_not_share_the_default_sources_list():
    a = Finding(subtask_id=uuid4(), status="no_results")
    b = Finding(subtask_id=uuid4(), status="no_results")
    assert a.sources is not b.sources


# --- retries: newest finding wins, attempts are counted from the findings ----


def _finding(subtask_id, answer):
    return Finding(
        subtask_id=subtask_id,
        status="found",
        answer=answer,
        sources=[Source(url="https://a.com", title="A", content="x")],
    )


def test_latest_per_subtask_picks_the_last_finding_of_each_subtask():
    one, two = uuid4(), uuid4()
    first, retry, other = (
        _finding(one, "first"),
        _finding(one, "retry"),
        _finding(two, "o"),
    )
    latest = latest_per_subtask([first, other, retry])
    assert latest == {one: retry, two: other}


def test_attempts_count_findings_per_subtask():
    one, two = uuid4(), uuid4()
    findings = [_finding(one, "a"), _finding(two, "b"), _finding(one, "c")]
    assert attempts_by_subtask(findings) == {one: 2, two: 1}


def test_no_findings_means_no_subtasks_and_no_attempts():
    assert latest_per_subtask([]) == {}
    assert attempts_by_subtask([]) == {}


def test_a_review_needs_an_attempt_of_at_least_one():
    with pytest.raises(ValidationError):
        Review(
            subtask_id=uuid4(),
            attempt=0,
            kept_urls=[],
            dropped_urls=[],
            source_scores={},
            addresses_subtask=False,
            used_fallback=False,
        )
