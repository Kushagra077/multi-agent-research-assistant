"""The HTTP API, driven through FastAPI's test client. The planner, the researcher LLM,
web search, the writer and Jev are all faked, so nothing touches the network or uses any
quota. (TestClient waits for the background run, so a GET right after a POST sees the
finished run.)"""

from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from research_assistant.api import create_app
from research_assistant.decisions import jev
from research_assistant.graph.nodes import planner as planner_module
from research_assistant.graph.nodes import researcher as researcher_module
from research_assistant.schemas.findings import ResearchSummary, Source
from research_assistant.schemas.plan import Plan, Subtask

GOOD = Source(url="https://good.com/a", title="Good", content="Real facts.")
SHOP = Source(url="https://shop.com/b", title="Shop", content="Buy now.")


class FakePlannerLLM:
    plan: ClassVar[Plan]
    error: ClassVar[Exception | None] = None

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if FakePlannerLLM.error:
            raise FakePlannerLLM.error
        return FakePlannerLLM.plan


class FakeResearcherLLM:
    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return ResearchSummary(answer="A fake answer.")


class FakeJev:
    """Scores a page by its url; booleans say yes; the retry choice is `choice`."""

    scores: ClassVar[dict[str, float]] = {}
    choice: ClassVar[str] = "escalate"

    def __init__(self, **kwargs):
        pass

    def system_one(self, state, questions):
        kind = questions["q"].type
        if kind == "noul":
            return SimpleNamespace(
                model="jev-1.13.0", nouls={"q": SimpleNamespace(noul=0.9)}
            )
        if kind == "choice":
            return SimpleNamespace(
                model="jev-1.13.0",
                choices={
                    "q": SimpleNamespace(
                        choice=FakeJev.choice,
                        confidence=0.9,
                        probabilities={FakeJev.choice: 0.9},
                    )
                },
            )
        score = next((v for url, v in FakeJev.scores.items() if url in state), 2.0)
        return SimpleNamespace(
            model="jev-1.13.0",
            scores={"q": SimpleNamespace(score=score, confidence=0.8)},
        )


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakePlannerLLM.plan = Plan(
        subtasks=[Subtask(description="What is X?"), Subtask(description="What is Y?")]
    )
    FakePlannerLLM.error = None
    monkeypatch.setattr(planner_module, "get_planner_llm", lambda: FakePlannerLLM())
    monkeypatch.setattr(
        researcher_module, "get_researcher_llm", lambda: FakeResearcherLLM()
    )
    monkeypatch.setattr(
        researcher_module, "search_web", lambda q, exclude_domains=None: [GOOD]
    )


def _client(tmp_path, name="api.db"):
    return TestClient(create_app(str(tmp_path / name), configure_logging=False))


def test_health(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/health").json() == {"ok": True}


def test_a_run_is_accepted_at_once_and_finishes_with_a_report(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/runs", json={"question": "  What are X and Y?  "})
        assert response.status_code == 202
        started = response.json()
        assert started["status"] == "running"

        run = client.get(f"/runs/{started['thread_id']}").json()
    assert run["status"] == "done"
    assert run["question"] == "What are X and Y?"  # cleaned
    assert [s["description"] for s in run["subtasks"]] == ["What is X?", "What is Y?"]
    assert all(s["status"] == "found" and s["attempts"] == 1 for s in run["subtasks"])
    assert "A fake report." in run["report"]
    assert "## Sources" in run["report"]
    # Jev is offline in tests (conftest), so the safe default applies: not ready.
    assert run["ship"] == {"ready": False, "confidence": None, "reason": "fallback"}
    assert run["error"] is None


def test_a_chosen_thread_id_is_used_and_cannot_be_reused(tmp_path):
    with _client(tmp_path) as client:
        first = client.post("/runs", json={"question": "q", "thread_id": "my-run_1"})
        assert first.json()["thread_id"] == "my-run_1"
        again = client.post("/runs", json={"question": "q", "thread_id": "my-run_1"})
    assert again.status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"question": "   "},
        {"question": ""},
        {},
        {"question": "q", "thread_id": "has spaces"},
        {"question": "q", "thread_id": "../etc"},
        {"question": "q", "thread_id": "x" * 65},
    ],
)
def test_bad_requests_are_rejected_before_anything_runs(tmp_path, body):
    with _client(tmp_path) as client:
        assert client.post("/runs", json=body).status_code == 422


def test_unknown_runs_are_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/runs/nope").status_code == 404
        assert client.post("/runs/nope/resume").status_code == 404


def test_the_sources_come_back_with_their_scores_and_whether_they_were_kept(
    tmp_path, monkeypatch
):
    long_text = "Real facts. " * 40
    good = Source(url="https://good.com/a", title="Good", content=long_text)
    shop = Source(url="https://shop.com/b", title="Shop", content=long_text)
    monkeypatch.setattr(
        researcher_module, "search_web", lambda q, exclude_domains=None: [good, shop]
    )
    monkeypatch.setattr(jev, "TypeSafeClient", FakeJev)
    FakeJev.scores = {"shop.com": 0.5, "good.com": 1.9}
    with _client(tmp_path) as client:
        thread_id = client.post("/runs", json={"question": "q"}).json()["thread_id"]
        run = client.get(f"/runs/{thread_id}").json()

    subtask = run["subtasks"][0]
    assert subtask["addresses_subtask"] is True
    assert subtask["sources"] == [
        {"url": good.url, "kept": True, "score": 1.9},
        {"url": shop.url, "kept": False, "score": 0.5},
    ]
    assert shop.url not in run["report"]
    assert run["ship"]["reason"] == "jev"
    assert run["ship"]["ready"] is True


def test_a_subtask_that_escalated_is_flagged_for_a_human(tmp_path, monkeypatch):
    monkeypatch.setattr(
        researcher_module, "search_web", lambda q, exclude_domains=None: [SHOP]
    )
    monkeypatch.setattr(jev, "TypeSafeClient", FakeJev)
    FakeJev.scores = {"shop.com": 0.5}  # every page is rejected
    FakeJev.choice = "escalate"
    with _client(tmp_path) as client:
        thread_id = client.post("/runs", json={"question": "q"}).json()["thread_id"]
        run = client.get(f"/runs/{thread_id}").json()

    assert all(
        s["needs_human"] and s["next_step"] == "escalate" for s in run["subtasks"]
    )
    assert "## Could not be answered" in run["report"]
    assert run["ship"] == {"ready": False, "confidence": None, "reason": "unanswered"}


def test_a_failed_run_reports_its_error_and_can_be_resumed(tmp_path):
    FakePlannerLLM.error = RuntimeError("gemini 504")
    with _client(tmp_path) as client:
        thread_id = client.post("/runs", json={"question": "q"}).json()["thread_id"]
        failed = client.get(f"/runs/{thread_id}").json()
        assert failed["status"] == "failed"
        assert failed["error"] == "gemini 504"
        assert failed["subtasks"] == []  # the planner never produced a plan

        FakePlannerLLM.error = None  # Gemini is back
        resumed = client.post(f"/runs/{thread_id}/resume")
        assert resumed.status_code == 202
        done = client.get(f"/runs/{thread_id}").json()
    assert done["status"] == "done"
    assert done["error"] is None
    assert len(done["subtasks"]) == 2


def test_a_finished_run_cannot_be_resumed(tmp_path):
    with _client(tmp_path) as client:
        thread_id = client.post("/runs", json={"question": "q"}).json()["thread_id"]
        assert client.post(f"/runs/{thread_id}/resume").status_code == 409


def test_after_a_server_restart_a_half_finished_run_is_interrupted_and_resumable(
    tmp_path,
):
    FakePlannerLLM.error = RuntimeError("gemini 504")
    with _client(tmp_path) as before_restart:
        thread_id = before_restart.post("/runs", json={"question": "q"}).json()[
            "thread_id"
        ]
        assert before_restart.get(f"/runs/{thread_id}").json()["status"] == "failed"

    # A new process: the in-memory error is gone, but the checkpoint is not.
    FakePlannerLLM.error = None
    with _client(tmp_path) as after_restart:
        assert after_restart.get(f"/runs/{thread_id}").json()["status"] == "interrupted"
        assert after_restart.post(f"/runs/{thread_id}/resume").status_code == 202
        run = after_restart.get(f"/runs/{thread_id}").json()
    assert run["status"] == "done"
    assert "A fake report." in run["report"]
