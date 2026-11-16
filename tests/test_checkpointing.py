"""Checkpointing with InMemorySaver. Planner LLM, researcher LLM and web search are
faked, so nothing touches the network or uses any quota."""

from typing import ClassVar

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from research_assistant.graph.build import build_graph
from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.schemas.findings import Finding, ResearchSummary, Source
from research_assistant.schemas.plan import Plan, Subtask

SOURCES = [Source(url="https://a.com", title="Page A", content="Fact from A.")]
SUBTASK_COUNT = 3


class FakePlannerLLM:
    plan: ClassVar[Plan]

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return FakePlannerLLM.plan


class FakeResearcherLLM:
    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return ResearchSummary(answer="A fake answer.")


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakePlannerLLM.plan = Plan(
        subtasks=[Subtask(description=f"question {i}") for i in range(SUBTASK_COUNT)]
    )
    monkeypatch.setattr(planner_module, "get_planner_llm", lambda: FakePlannerLLM())
    monkeypatch.setattr(
        researcher_module, "get_researcher_llm", lambda: FakeResearcherLLM()
    )
    monkeypatch.setattr(
        researcher_module,
        "search_web",
        lambda query, exclude_domains=None: list(SOURCES),
    )


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _run(thread_id: str = "t1"):
    """Run the full graph once; return (graph, config, final result)."""
    graph = build_graph(InMemorySaver())
    config = _config(thread_id)
    result = graph.invoke({"question": "What is X?"}, config)
    return graph, config, result


def _oldest_first(graph, config):
    return list(reversed(list(graph.get_state_history(config))))


# --- how many checkpoints, and where ---------------------------------------


def test_a_full_run_saves_nine_checkpoints_numbered_minus_one_to_seven():
    graph, config, _ = _run()
    steps = [s.metadata["step"] for s in _oldest_first(graph, config)]
    assert steps == [-1, 0, 1, 2, 3, 4, 5, 6, 7]


def test_each_checkpoint_says_which_nodes_run_next():
    graph, config, _ = _run()
    next_nodes = [s.next for s in _oldest_first(graph, config)]
    assert next_nodes == [
        ("__start__",),
        ("intake",),
        ("planner",),
        ("researcher",) * SUBTASK_COUNT,
        ("merge",),
        ("decide",),
        ("writer",),
        ("ship_check",),
        (),
    ]


def test_all_parallel_researchers_share_one_checkpoint():
    graph, config, _ = _run()
    with_researchers = [
        s for s in graph.get_state_history(config) if "researcher" in s.next
    ]
    assert len(with_researchers) == 1
    assert len(with_researchers[0].tasks) == SUBTASK_COUNT


def test_the_first_checkpoint_is_the_input_and_the_rest_come_from_the_loop():
    graph, config, _ = _run()
    sources = [s.metadata["source"] for s in _oldest_first(graph, config)]
    assert sources == ["input"] + ["loop"] * 8


# --- the saved state is the real state -------------------------------------


def test_latest_checkpoint_is_finished_and_matches_the_returned_result():
    graph, config, result = _run()
    latest = graph.get_state(config)
    assert latest.next == ()
    assert latest.values["question"] == result["question"]
    assert latest.values["plan"] == result["plan"]
    assert latest.values["findings"] == result["findings"]


def test_pydantic_models_come_back_from_the_checkpointer_as_the_same_models():
    graph, config, result = _run()
    saved = graph.get_state(config).values
    assert isinstance(saved["plan"], Plan)
    assert all(isinstance(s, Subtask) for s in saved["plan"].subtasks)
    assert len(saved["findings"]) == SUBTASK_COUNT
    assert all(isinstance(f, Finding) for f in saved["findings"])
    assert saved["findings"] == result["findings"]


def test_a_checkpoint_in_the_middle_holds_only_what_had_run_by_then():
    graph, config, _ = _run()
    # the snapshot taken after the planner, just before the researchers run
    after_planner = next(
        s for s in graph.get_state_history(config) if "researcher" in s.next
    )
    assert "plan" in after_planner.values
    assert not after_planner.values.get("findings")


# --- threads ---------------------------------------------------------------


def test_two_thread_ids_keep_separate_histories():
    graph = build_graph(InMemorySaver())
    graph.invoke({"question": "first"}, _config("a"))
    graph.invoke({"question": "second"}, _config("b"))
    assert graph.get_state(_config("a")).values["question"] == "first"
    assert graph.get_state(_config("b")).values["question"] == "second"
    assert len(list(graph.get_state_history(_config("a")))) == 9


def test_an_unknown_thread_has_no_saved_state():
    graph = build_graph(InMemorySaver())
    snapshot = graph.get_state(_config("never-used"))
    assert snapshot.values == {}
    assert snapshot.next == ()
    assert list(graph.get_state_history(_config("never-used"))) == []


def test_without_a_checkpointer_nothing_is_saved_and_reading_state_fails():
    graph = build_graph()  # the default: no checkpointer
    graph.invoke({"question": "q"})
    with pytest.raises(ValueError):
        graph.get_state(_config("t1"))


# --- SqliteSaver: state survives closing the saver and starting again ------------
# Closing the saver and opening a new one on the same file imitates a restart inside
# one process. The real kill-and-resume test needs a separate process.


def test_state_and_history_survive_closing_and_reopening_the_database(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    config = _config("t1")

    with SqliteSaver.from_conn_string(db) as saver:
        result = build_graph(saver).invoke({"question": "What is X?"}, config)
    # The saver is closed here. Nothing from the first graph is reused below.

    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(saver)
        saved = graph.get_state(config)
        history = list(graph.get_state_history(config))

    assert saved.next == ()
    assert saved.values["question"] == "What is X?"
    assert saved.values["plan"] == result["plan"]
    assert saved.values["findings"] == result["findings"]
    assert len(history) == 9


def test_models_come_back_as_models_from_the_database_file(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    config = _config("t1")
    with SqliteSaver.from_conn_string(db) as saver:
        build_graph(saver).invoke({"question": "q"}, config)

    with SqliteSaver.from_conn_string(db) as saver:
        values = build_graph(saver).get_state(config).values

    assert isinstance(values["plan"], Plan)
    assert all(isinstance(f, Finding) for f in values["findings"])


def test_a_new_thread_id_in_the_same_database_starts_empty(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(saver)
        graph.invoke({"question": "q"}, _config("t1"))
        assert graph.get_state(_config("t2")).values == {}


def test_resuming_a_finished_thread_runs_no_node_again(tmp_path, monkeypatch):
    calls = []
    real_invoke = FakePlannerLLM.invoke

    def counting_invoke(self, messages):
        calls.append("planner")
        return real_invoke(self, messages)

    monkeypatch.setattr(FakePlannerLLM, "invoke", counting_invoke)
    db = str(tmp_path / "checkpoints.db")
    config = _config("t1")

    with SqliteSaver.from_conn_string(db) as saver:
        first = build_graph(saver).invoke({"question": "q"}, config)
    assert calls == ["planner"]

    with SqliteSaver.from_conn_string(db) as saver:
        again = build_graph(saver).invoke(None, config)  # None = "just continue"

    assert calls == ["planner"]  # the planner was not called a second time
    assert again["findings"] == first["findings"]
