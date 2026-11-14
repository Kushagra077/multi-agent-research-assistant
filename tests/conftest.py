"""Shared pytest setup.

LangGraph reads LANGGRAPH_STRICT_MSGPACK once, when it is first imported. pytest imports
this file before any test module, so setting it here makes every test run in strict
mode, the same as the real entry point (main.py). In strict mode only the types in the
graph's state schema may be loaded from a checkpoint.
"""

import os

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


import pytest
from typesafe_sdk import TypeSafeAPIConnectionError

from research_assistant.decisions import jev


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
