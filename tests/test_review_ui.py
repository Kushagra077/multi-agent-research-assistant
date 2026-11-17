"""The Streamlit review page, run with Streamlit's own test harness. The API client is
replaced by a fake, so no server and no network are involved."""

from pathlib import Path
from typing import ClassVar

import pytest
from streamlit.testing.v1 import AppTest

import research_assistant
from research_assistant import ui_client
from research_assistant.ui_client import ApiError

UI_FILE = Path(research_assistant.__file__).parent / "review_ui.py"


def _run(**overrides):
    run = {
        "thread_id": "t1",
        "status": "awaiting_review",
        "question": "What is X?",
        "subtasks": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "description": "Sub one",
                "status": "found",
                "attempts": 2,
                "answer": "An answer.",
                "addresses_subtask": True,
                "sources": [
                    {"url": "https://good.com/a", "kept": True, "score": 1.9},
                    {"url": "https://quora.com/q", "kept": False, "score": 0.01},
                ],
                "next_step": "retry_different",
                "needs_human": False,
            },
            {
                "id": "00000000-0000-0000-0000-000000000002",
                "description": "Sub two",
                "status": "found",
                "attempts": 2,
                "answer": None,
                "addresses_subtask": False,
                "sources": [],
                "next_step": "escalate",
                "needs_human": True,
            },
        ],
        "report": "The draft report.",
        "ship": {"ready": False, "confidence": None, "reason": "unanswered"},
        "feedback": [],
        "final": None,
        "error": None,
    }
    run.update(overrides)
    return run


class FakeApi:
    run: ClassVar[dict | None] = None
    calls: ClassVar[list] = []
    error: ClassVar[ApiError | None] = None

    def __init__(self, base_url):
        pass

    def get(self, thread_id):
        return FakeApi.run

    def review(self, thread_id, action, feedback=None, text=None):
        FakeApi.calls.append(("review", thread_id, action, feedback, text))
        if FakeApi.error:
            raise FakeApi.error
        return {}

    def resume(self, thread_id):
        FakeApi.calls.append(("resume", thread_id))
        return {}

    def start(self, question):
        FakeApi.calls.append(("start", question))
        return {"thread_id": "new-run"}


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    FakeApi.run = _run()
    FakeApi.calls = []
    FakeApi.error = None
    monkeypatch.setattr(ui_client, "ApiClient", FakeApi)


def _app(thread_id="t1"):
    app = AppTest.from_file(str(UI_FILE), default_timeout=20)
    if thread_id:
        app.session_state["thread_id"] = thread_id
    return app.run()


def _text(app):
    parts = [m.value for m in app.markdown] + [m.value for m in app.info]
    parts += [m.value for m in app.warning] + [m.value for m in app.error]
    parts += [m.value for m in app.success] + [c.value for c in app.caption]
    return "\n".join(parts)


def _submit(app):
    (button,) = [b for b in app.button if b.label == "Submit decision"]
    button.click()
    return app.run()


def test_with_no_run_open_the_page_says_how_to_start():
    app = _app(thread_id=None)
    assert not app.exception
    assert "Start a run in the sidebar" in _text(app)


def test_an_unknown_run_says_so():
    FakeApi.run = None
    assert "no run with the id 't1'" in _text(_app())


def test_a_waiting_run_shows_the_draft_the_verdict_and_the_decision_form():
    app = _app()
    assert not app.exception
    text = _text(app)
    assert "The draft report." in text
    assert "NOT ready" in text
    assert "a subtask has no usable answer" in text
    assert "it needs a human" in text
    assert "2 attempts" in text
    assert [r.label for r in app.radio] == ["Decision"]
    assert app.radio[0].options == [
        "Approve",
        "Reject with feedback",
        "Take over (edit)",
    ]


def test_the_sources_table_shows_kept_or_dropped_with_the_score():
    app = _app()
    rows = [t.value for t in app.table]
    verdicts = {
        (r["verdict"], r["score (0-2)"], r["url"]) for r in rows[0].to_dict("records")
    }
    assert verdicts == {
        ("kept", "1.90", "https://good.com/a"),
        ("dropped", "0.01", "https://quora.com/q"),
    }


def test_approve_calls_the_api_with_approve():
    app = _app()
    app.radio[0].set_value("Approve")
    _submit(app)
    assert FakeApi.calls == [("review", "t1", "approve", None, None)]


def test_reject_sends_the_feedback():
    app = _app()
    app.radio[0].set_value("Reject with feedback")
    feedback = next(t for t in app.text_area if t.label.startswith("Feedback"))
    feedback.set_value("Shorter, please.")
    _submit(app)
    assert FakeApi.calls == [("review", "t1", "reject", "Shorter, please.", None)]


def test_take_over_sends_the_edited_text_starting_from_the_draft():
    app = _app()
    app.radio[0].set_value("Take over (edit)")
    edited = next(t for t in app.text_area if t.label.startswith("Your own"))
    assert edited.value == "The draft report."
    edited.set_value("My report.")
    _submit(app)
    assert FakeApi.calls == [("review", "t1", "edit", None, "My report.")]


def test_an_api_refusal_is_shown_instead_of_crashing():
    FakeApi.error = ApiError("A reject needs feedback for the writer.")
    app = _app()
    app.radio[0].set_value("Reject with feedback")
    app = _submit(app)
    assert not app.exception
    assert "A reject needs feedback for the writer." in [e.value for e in app.error]


def test_earlier_rejections_are_listed():
    FakeApi.run = _run(feedback=["Too long.", "Add numbers."])
    text = _text(_app())
    assert "1. Too long." in text
    assert "2. Add numbers." in text


def test_a_finished_run_shows_the_delivered_report_and_no_form():
    FakeApi.run = _run(
        status="done", final={"text": "The final text.", "how": "edited"}
    )
    app = _app()
    text = _text(app)
    assert "Delivered report (replaced by the reviewer)" in text
    assert "The final text." in text
    assert not app.radio


def test_a_failed_run_shows_the_error_and_a_resume_button():
    FakeApi.run = _run(status="failed", error="gemini 504", report=None, ship=None)
    app = _app()
    assert "The run failed: gemini 504" in [e.value for e in app.error]
    (button,) = [b for b in app.button if b.label.startswith("Resume")]
    button.click()
    app.run()
    assert FakeApi.calls == [("resume", "t1")]
