# One place to build LLM clients (Gemini via langchain-google-genai, Groq via langchain-groq)
# so nodes never import a provider SDK directly.

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()  # Load environment variables from .env file

PLANNER_MODEL = "gemini-3.8-flash"
RESEARCHER_MODEL = "openai/gpt-oss-20b"

# Free tiers are tight on requests, and every parallel researcher shares one Groq
# token budget. So: fail fast on a hung call, and retry at most once.
RESEARCHER_TIMEOUT_SECONDS = 30
RESEARCHER_MAX_RETRIES = 1


def get_planner_llm() -> BaseChatModel:
    return init_chat_model(
        model_provider="google-genai", model=PLANNER_MODEL, temperature=0.0
    )


def get_researcher_llm() -> BaseChatModel:
    return init_chat_model(
        model_provider="groq",
        model=RESEARCHER_MODEL,
        temperature=0.0,
        timeout=RESEARCHER_TIMEOUT_SECONDS,
        max_retries=RESEARCHER_MAX_RETRIES,
    )
