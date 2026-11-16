"""Shared pytest setup.

LangGraph reads LANGGRAPH_STRICT_MSGPACK once, when it is first imported. pytest imports
this file before any test module, so setting it here makes every test run in strict
mode, the same as the real entry point (main.py). In strict mode only the types in the
graph's state schema may be loaded from a checkpoint.
"""

import os

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
# Tracing sends research text to LangSmith. A developer may switch it on in .env for real
# runs; tests must never send anything, so it is forced off here (and inherited by the
# subprocess that tests/crash_runner.py runs in).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


from types import SimpleNamespace

import pytest
from typesafe_sdk import TypeSafeAPIConnectionError

from research_assistant.decisions import jev
from research_assistant.graph.nodes import merge as merge_module
from research_assistant.graph.nodes import writer as writer_module


class _OfflineJevClient:
    """Every Jev call fails like an outage, so a test can never reach the real API.

    The failure makes ask_* return their defaults (fallback=True). Tests that care about
    Jev's answers (test_jev.py, test_merge.py) replace this with their own fake.
    """

    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        raise TypeSafeAPIConnectionError("tests run offline")


@pytest.fixture(autouse=True)
def _no_real_jev_calls(monkeypatch):
    monkeypatch.setattr(jev, "TypeSafeClient", _OfflineJevClient)


class _OfflineWriterLLM:
    """Stands in for the Groq writer, so a graph test never reaches the network.

    Tests that care about the report (test_writer.py) replace this with their own fake.
    """

    def invoke(self, messages):
        return SimpleNamespace(content="A fake report.")


@pytest.fixture(autouse=True)
def _no_real_writer_calls(monkeypatch):
    monkeypatch.setattr(writer_module, "get_writer_llm", lambda: _OfflineWriterLLM())


@pytest.fixture(autouse=True)
def _short_fake_pages_are_allowed(monkeypatch):
    """Test pages are a few words long, so the minimum-text rule is off by default.

    test_merge.py turns it back on for the tests that are about it.
    """
    monkeypatch.setattr(merge_module, "MIN_SOURCE_CHARS", 0)
