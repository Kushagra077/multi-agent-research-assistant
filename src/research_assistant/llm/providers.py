# One place to build LLM clients (Gemini via langchain-google-genai, Groq via langchain-groq)
# so nodes never import a provider SDK directly.

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from research_assistant.schemas.plan import Plan, PlanDraft, Subtask

load_dotenv()  # Load environment variables from .env file

# The planner can run on Groq (default) or Gemini: set PLANNER_PROVIDER=gemini to switch.
# Gemini's free tier kept answering 504 DEADLINE_EXCEEDED at the 30 s timeout (and
# gemini-3.8-flash earlier failed with 503/504), which stopped every run at the planner.
DEFAULT_PLANNER_PROVIDER = "groq"
PLANNER_MODEL = "gemini-3.6-flash"  # used when the provider is gemini
GROQ_PLANNER_MODEL = "openai/gpt-oss-20b"
RESEARCHER_MODEL = "openai/gpt-oss-20b"
# The writer reuses the researcher's Groq model: Gemini has been unreliable.
WRITER_MODEL = RESEARCHER_MODEL

# Free tiers are tight on requests, and every parallel researcher shares one Groq
# token budget. So: fail fast on a hung call, and retry at most once.
# The planner has the same caps: a Gemini 503 outage once kept one run retrying for ~5
# minutes and burned daily quota.
PLANNER_TIMEOUT_SECONDS = 30
PLANNER_MAX_RETRIES = 1
RESEARCHER_TIMEOUT_SECONDS = 30
RESEARCHER_MAX_RETRIES = 1
# A whole report is longer than one researcher answer, so it gets more time.
WRITER_TIMEOUT_SECONDS = 60
WRITER_MAX_RETRIES = 1
# gpt-oss is a reasoning model: its hidden reasoning counts against the output limit, so
# a redraft (longer prompt: findings + earlier draft + feedback) can run out before the
# report ends. A generous explicit limit, and low reasoning effort, leave room for it.
WRITER_MAX_TOKENS = 4096
WRITER_REASONING_EFFORT = "low"


class _GroqPlanner:
    """Gives a Groq model the interface the planner node uses with Gemini:
    with_structured_output(Plan).invoke(messages) returns a Plan.

    Groq's strict JSON mode needs every field required, and a Subtask's id is generated,
    so the model is asked for PlanDraft (descriptions only) and the ids are made here:
    an LLM should not invent ids (the same rule as the researcher's one-field summary).
    """

    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(PlanDraft, method="json_schema")

    def with_structured_output(self, schema):
        if schema is not Plan:
            raise ValueError("This planner only produces a Plan.")
        return self

    def invoke(self, messages) -> Plan:
        draft = self._llm.invoke(messages)
        descriptions = [d.strip() for d in draft.subtasks if d.strip()]
        # An empty list fails Plan's own validation, so the planner node reports it.
        return Plan(subtasks=[Subtask(description=d) for d in descriptions])


def get_planner_llm():
    provider = os.getenv("PLANNER_PROVIDER", DEFAULT_PLANNER_PROVIDER).strip().lower()
    if provider == "gemini":
        return init_chat_model(
            model_provider="google-genai",
            model=PLANNER_MODEL,
            temperature=0.0,
            timeout=PLANNER_TIMEOUT_SECONDS,
            max_retries=PLANNER_MAX_RETRIES,
        )
    if provider == "groq":
        return _GroqPlanner(
            init_chat_model(
                model_provider="groq",
                model=GROQ_PLANNER_MODEL,
                temperature=0.0,
                timeout=PLANNER_TIMEOUT_SECONDS,
                max_retries=PLANNER_MAX_RETRIES,
            )
        )
    raise ValueError(f"PLANNER_PROVIDER must be 'groq' or 'gemini', not {provider!r}.")


def get_researcher_llm() -> BaseChatModel:
    return init_chat_model(
        model_provider="groq",
        model=RESEARCHER_MODEL,
        temperature=0.0,
        timeout=RESEARCHER_TIMEOUT_SECONDS,
        max_retries=RESEARCHER_MAX_RETRIES,
    )


def get_writer_llm() -> BaseChatModel:
    return init_chat_model(
        model_provider="groq",
        model=WRITER_MODEL,
        temperature=0.0,
        timeout=WRITER_TIMEOUT_SECONDS,
        max_retries=WRITER_MAX_RETRIES,
        max_tokens=WRITER_MAX_TOKENS,
        reasoning_effort=WRITER_REASONING_EFFORT,
    )
