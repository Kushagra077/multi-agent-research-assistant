"""Runs the real entry point (main.main) with every external call faked.

Used only by test_crash_resume.py, which starts this file as a separate process, kills
it mid-run, then starts it again with --resume. It is not a test file itself.

Fakes:
  - planner LLM: a fixed plan with two fast subtasks and one slow one
  - researcher LLM: a canned answer
  - writer LLM: a canned report
  - Jev: every call fails like an outage, so merge falls back to its defaults
  - web search: appends every query to the file named by CRASH_TEST_CALLS_FILE (so the
    test can count how many times each subtask was searched), and sleeps for a long
    time on the FIRST search of the slow subtask only.

Nothing here touches the network or uses any API quota.
"""

import os
import sys
import time
from pathlib import Path

from typesafe_sdk import TypeSafeAPIConnectionError

from research_assistant.decisions import jev
from research_assistant.graph.nodes import merge as merge_module
from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.graph.nodes import writer as writer_module
from research_assistant.schemas.findings import ResearchSummary, Source
from research_assistant.schemas.plan import Plan, Subtask

FAST_QUERIES = ["fast question 1", "fast question 2"]
SLOW_QUERY = "slow question"
SLOW_SECONDS = 60  # long enough that the test always kills the process first

CALLS_FILE = Path(os.environ["CRASH_TEST_CALLS_FILE"])


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return self.reply


def fake_search(query: str, exclude_domains=None) -> list[Source]:
    already_searched = CALLS_FILE.exists() and query in CALLS_FILE.read_text().split(
        "\n"
    )
    with CALLS_FILE.open("a") as f:
        f.write(query + "\n")
    if query == SLOW_QUERY and not already_searched:
        time.sleep(SLOW_SECONDS)  # the test kills the process during this sleep
    return [Source(url="https://a.com", title="Page A", content=f"About {query}.")]


plan = Plan(subtasks=[Subtask(description=q) for q in [*FAST_QUERIES, SLOW_QUERY]])
planner_module.get_planner_llm = lambda: FakeLLM(plan)
researcher_module.get_researcher_llm = lambda: FakeLLM(
    ResearchSummary(answer="A fake answer.")
)
researcher_module.search_web = fake_search


class OfflineJevClient:
    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        raise TypeSafeAPIConnectionError("crash test runs offline")


jev.TypeSafeClient = OfflineJevClient


class FakeWriterLLM:
    def invoke(self, messages):
        return type("Reply", (), {"content": "A fake report."})()


writer_module.get_writer_llm = lambda: FakeWriterLLM()
merge_module.MIN_SOURCE_CHARS = 0  # the fake pages are a few words long

if __name__ == "__main__":
    from research_assistant.main import main

    sys.exit(main())
