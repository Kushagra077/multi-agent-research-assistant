import pytest
from pydantic import ValidationError

from research_assistant.schemas.plan import Plan, Subtask


def test_subtask_gets_unique_ids():
    a = Subtask(description="x")
    b = Subtask(description="y")
    assert a.id != b.id


def test_plan_rejects_empty_subtasks():
    with pytest.raises(ValidationError):
        Plan(subtasks=[])


def test_subtask_requires_description():
    with pytest.raises(ValidationError):
        Subtask()
