from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RetryDecision(BaseModel):
    """What to do about one subtask whose latest attempt did not produce a usable answer.

    One is recorded per (subtask, attempt) and the list only grows, so it is also the
    history of why each retry or escalation happened.
    """

    subtask_id: UUID = Field(
        description="The id of the Subtask this decision is about."
    )
    attempt: int = Field(
        ge=1, description="The attempt that was judged (1 = the first one)."
    )
    action: Literal["retry_same", "retry_different", "escalate"]
    reason: Literal["jev", "cap", "fallback"] = Field(
        description="jev: Jev chose the action. cap: the retry limit was reached, so "
        "Jev was not asked. fallback: Jev was unreachable, so the default was used."
    )
    confidence: float | None = Field(
        description="Jev's confidence in its choice; None unless reason is 'jev'."
    )
    exclude_domains: list[str] = Field(
        description="Sites the retry must skip: those whose pages the source gate "
        "rejected on any earlier attempt. Empty unless action is 'retry_different'."
    )
