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
        min_length=1,
        description="2 to 3 subtasks that together answer the question.",
    )


class PlanDraft(BaseModel):
    """What a model that cannot take a Plan's generated ids writes: descriptions only.

    Used with Groq's strict JSON mode, where every field must be required. The code, not
    the model, makes the ids (see llm/providers.py).
    """

    subtasks: list[str] = Field(
        description="2 to 3 specific, self-contained research questions that together "
        "answer the question. Each must make sense on its own."
    )
