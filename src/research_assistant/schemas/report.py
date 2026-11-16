from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class Draft(BaseModel):
    """The report the writer produced, and which subtasks it could and could not cover."""

    text: str = Field(description="The full report: body, sources, and open questions.")
    answered: list[UUID] = Field(
        description="Subtasks whose findings the report is written from."
    )
    unanswered: list[UUID] = Field(
        description="Subtasks with no usable answer (escalated); listed in the report "
        "as open questions, never written about."
    )


class ShipVerdict(BaseModel):
    """Whether the draft meets the minimum bar to be shown to a human for approval."""

    ready: bool
    confidence: float | None = Field(
        description="Jev's confidence in the verdict; None unless reason is 'jev'."
    )
    reason: Literal["jev", "unanswered", "fallback"] = Field(
        description="jev: Jev judged the draft. unanswered: a subtask has no usable "
        "answer, so the draft cannot be complete and Jev was not asked. fallback: Jev "
        "was unreachable, so the safe default (not ready) was used."
    )


class HumanDecision(BaseModel):
    """What the reviewer sent back when the graph paused for them."""

    action: Literal["approve", "reject", "edit"]
    feedback: str | None = Field(
        default=None, description="Why the draft was rejected (required for reject)."
    )
    text: str | None = Field(
        default=None, description="The reviewer's own final report (required for edit)."
    )

    @model_validator(mode="after")
    def check_action_has_what_it_needs(self) -> HumanDecision:
        if self.action == "reject" and not (self.feedback or "").strip():
            raise ValueError("A reject needs feedback for the writer.")
        if self.action == "edit" and not (self.text or "").strip():
            raise ValueError("An edit needs the replacement text.")
        return self


class FinalReport(BaseModel):
    """The report that was delivered, and how it got approved."""

    text: str
    how: Literal["approved", "edited"]
