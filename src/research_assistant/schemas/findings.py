from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class Source(BaseModel):
    """A web page a finding is based on."""

    url: str = Field(description="The URL of the source.")
    title: str = Field(description="The title of the source.")
    content: str = Field(description="The extracted text of the source.")


class Finding(BaseModel):
    """The outcome of one researcher working on one subtask."""

    subtask_id: UUID = Field(description="The id of the Subtask this finding answers.")
    status: Literal["found", "no_results", "failed"]
    answer: str | None = None
    sources: list[Source] = Field(
        default_factory=list, description="Sources the answer is based on."
    )
    error: str | None = None

    @model_validator(mode="after")
    def check_status_matches_fields(self) -> Finding:
        if self.status == "found" and not (self.answer and self.sources):
            raise ValueError("A 'found' finding needs a non-empty answer and sources.")
        if self.status == "failed" and not self.error:
            raise ValueError("A 'failed' finding needs an error message.")
        return self


class ResearchSummary(BaseModel):
    answer: str = Field(
        description="A concise answer to the research question, based only on the provided sources."
    )


def latest_per_subtask(findings: list[Finding]) -> dict[UUID, Finding]:
    """The newest finding of each subtask.

    `findings` only ever grows (the reducer appends), and a retry is appended after the
    attempt it replaces, so the last finding of a subtask in the list is its newest.
    The older ones stay in the list as history.
    """
    return {finding.subtask_id: finding for finding in findings}


def attempts_by_subtask(findings: list[Finding]) -> dict[UUID, int]:
    """How many times each subtask has been researched (1 = first attempt only).

    Derived from the findings themselves, so it cannot drift out of step with them.
    """
    attempts: dict[UUID, int] = {}
    for finding in findings:
        attempts[finding.subtask_id] = attempts.get(finding.subtask_id, 0) + 1
    return attempts


class Review(BaseModel):
    """What the merge step decided about one 'found' Finding.

    Holds urls, not whole Sources, so page text is not stored twice in every checkpoint.
    The Finding keeps all of its sources; `kept_urls` says which ones passed the gate.
    """

    subtask_id: UUID = Field(description="The id of the Subtask this review is about.")
    attempt: int = Field(
        ge=1,
        description="Which attempt at the subtask this review covers (1 = the first).",
    )
    kept_urls: list[str] = Field(description="Sources that passed the quality gate.")
    dropped_urls: list[str] = Field(description="Sources the gate removed.")
    source_scores: dict[str, float | None] = Field(
        description="Jev's score (0 to 2) for every source, by url, in source order. "
        "None where Jev was unreachable: the default used then is not a real score."
    )
    addresses_subtask: bool = Field(
        description="Whether the answer actually responds to its subtask."
    )
    used_fallback: bool = Field(
        description="True if Jev was unreachable for any check, so a default was used."
    )
