# One place to build LLM clients (Gemini via langchain-google-genai, Groq via langchain-groq)
# so nodes never import a provider SDK directly.

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()  # Load environment variables from .env file

# gemini-3.8-flash kept failing with 503/504 on the free tier, so the planner uses 3.6.
PLANNER_MODEL = "gemini-3.6-flash"
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


def get_planner_llm() -> BaseChatModel:
    return init_chat_model(
        model_provider="google-genai",
        model=PLANNER_MODEL,
        temperature=0.0,
        timeout=PLANNER_TIMEOUT_SECONDS,
        max_retries=PLANNER_MAX_RETRIES,
    )


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
    )
