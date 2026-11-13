"""Shared pytest setup.

LangGraph reads LANGGRAPH_STRICT_MSGPACK once, when it is first imported. pytest imports
this file before any test module, so setting it here makes every test run in strict
mode, the same as the real entry point (main.py). In strict mode only the types in the
graph's state schema may be loaded from a checkpoint.
"""

import os

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
