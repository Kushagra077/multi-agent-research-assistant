# Pydantic models for the supervisor's output.

from uuid import UUID, uuid7
from pydantic import BaseModel, Field


class Subtask(BaseModel):
    """One focused research question a single researcher can answer."""

    id: UUID = Field(default_factory=uuid7)
    description: str = Field(
        description="A specific, self-contained research question. It must make sense on its own, without the other subtasks."
    )


class Plan(BaseModel):
    """A research plan: the question broken into independent subtasks."""

    subtasks: list[Subtask] = Field(
        min_length=1, description="2 to 5 subtasks that together answer the question."
    )
