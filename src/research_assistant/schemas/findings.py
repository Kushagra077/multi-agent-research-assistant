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
