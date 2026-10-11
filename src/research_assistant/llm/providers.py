# One place to build LLM clients (Gemini via langchain-google-genai, Groq via langchain-groq)
# so nodes never import a provider SDK directly.
# TODO: a get_planner_llm() (or similar) factory function.

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()  # Load environment variables from .env file

PLANNER_MODEL = "gemini-3.8-flash"


def get_planner_llm() -> BaseChatModel:
    return init_chat_model(
        model_provider="google-genai", model=PLANNER_MODEL, temperature=0.0
    )
