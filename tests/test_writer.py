import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest

from research_assistant.graph.nodes import writer as writer_module
from research_assistant.graph.nodes.writer import (
    NO_ANSWERS_BODY,
    WRITER_SYSTEM_PROMPT,
    WriterOutputCutOff,
    writer,
)
from research_assistant.schemas.findings import Finding, Review, Source
from research_assistant.schemas.plan import Plan, Subtask

GOOD = Source(url="https://good.com/a", title="Good", content="PAGE TEXT GOOD")
SHOP = Source(url="https://shop.com/b", title="Shop", content="PAGE TEXT SHOP")


class FakeWriterLLM:
    """Stands in for the Groq writer: records what it was asked, returns a canned reply."""

    reply: ClassVar[object] = SimpleNamespace(content="  The body of the report.  ")
    invoked_with: ClassVar[list] = []

    def invoke(self, messages):
        FakeWriterLLM.invoked_with.append(messages)
        if isinstance(FakeWriterLLM.reply, Exception):
            raise FakeWriterLLM.reply
        return FakeWriterLLM.reply


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeWriterLLM.reply = SimpleNamespace(content="  The body of the report.  ")
    FakeWriterLLM.invoked_with = []
    monkeypatch.setattr(writer_module, "get_writer_llm", lambda: FakeWriterLLM())


def _found(subtask, answer, *sources):
    return Finding(
        subtask_id=subtask.id, status="found", answer=answer, sources=list(sources)
    )


def _review(subtask, *, kept, dropped=(), addresses=True, attempt=1):
    return Review(
        subtask_id=subtask.id,
        attempt=attempt,
        kept_urls=list(kept),
        dropped_urls=list(dropped),
        source_scores={},
        addresses_subtask=addresses,
        used_fallback=False,
    )


def _state(subtasks, findings, reviews=()):
    return {
        "question": "What is X and Y?",
        "plan": Plan(subtasks=list(subtasks)),
        "findings": list(findings),
        "reviews": list(reviews),
    }


def test_the_draft_is_the_models_body_then_the_sources_of_each_answered_subtask():
    x, y = Subtask(description="What is X?"), Subtask(description="What is Y?")
    state = _state(
        [x, y],
        [_found(x, "X is 1.", GOOD), _found(y, "Y is 2.", SHOP)],
        [_review(x, kept=[GOOD.url]), _review(y, kept=[SHOP.url])],
    )
    draft = writer(state)["draft"]
    assert draft.answered == [x.id, y.id]
    assert draft.unanswered == []
    assert draft.text == (
        "The body of the report.\n\n"
        "## Sources\n\n"
        "### What is X?\n- https://good.com/a\n\n"
        "### What is Y?\n- https://shop.com/b"
    )


def test_the_model_sees_the_question_and_the_answers_but_never_the_page_text():
    x = Subtask(description="What is X?")
    writer(_state([x], [_found(x, "X is 1.", GOOD)], [_review(x, kept=[GOOD.url])]))
    system, human = FakeWriterLLM.invoked_with[0]
    assert system.content == WRITER_SYSTEM_PROMPT
    assert "Question: What is X and Y?" in human.content
    assert "Subtask 1: What is X?" in human.content
    assert "X is 1." in human.content
    assert "PAGE TEXT GOOD" not in human.content
    assert GOOD.url not in human.content  # urls are added by code, not by the model


def test_only_sources_that_passed_the_gate_are_listed():
    x = Subtask(description="What is X?")
    state = _state(
        [x],
        [_found(x, "X is 1.", GOOD, SHOP)],
        [_review(x, kept=[GOOD.url], dropped=[SHOP.url])],
    )
    text = writer(state)["draft"].text
    assert GOOD.url in text
    assert SHOP.url not in text


def test_a_found_finding_with_no_review_counts_as_usable_with_all_its_sources():
    x = Subtask(description="What is X?")
    draft = writer(_state([x], [_found(x, "X is 1.", GOOD, SHOP)]))["draft"]
    assert draft.answered == [x.id]
    assert GOOD.url in draft.text
    assert SHOP.url in draft.text


def test_a_subtask_without_a_usable_answer_is_listed_by_code_and_never_sent_to_the_model():
    x, y, z = (
        Subtask(description=d) for d in ("What is X?", "What is Y?", "What is Z?")
    )
    state = _state(
        [x, y, z],
        [
            _found(x, "X is 1.", GOOD),
            Finding(subtask_id=y.id, status="no_results"),
            _found(z, "A dodge.", SHOP),
        ],
        [_review(x, kept=[GOOD.url]), _review(z, kept=[], addresses=False)],
    )
    draft = writer(state)["draft"]
    assert draft.answered == [x.id]
    assert draft.unanswered == [y.id, z.id]
    assert draft.text.endswith("## Could not be answered\n\n- What is Y?\n- What is Z?")
    human = FakeWriterLLM.invoked_with[0][1].content
    assert "What is Y?" not in human
    assert "A dodge." not in human


def test_only_the_newest_attempt_of_a_retried_subtask_is_used():
    x = Subtask(description="What is X?")
    state = _state(
        [x],
        [_found(x, "old answer", SHOP), _found(x, "new answer", GOOD)],
        [
            _review(x, kept=[], dropped=[SHOP.url], addresses=False, attempt=1),
            _review(x, kept=[GOOD.url], attempt=2),
        ],
    )
    draft = writer(state)["draft"]
    assert draft.answered == [x.id]
    human = FakeWriterLLM.invoked_with[0][1].content
    assert "new answer" in human
    assert "old answer" not in human
    assert SHOP.url not in draft.text


def test_when_nothing_is_usable_the_model_is_not_asked_to_write_anyway():
    x = Subtask(description="What is X?")
    state = _state([x], [Finding(subtask_id=x.id, status="failed", error="boom")])
    draft = writer(state)["draft"]
    assert FakeWriterLLM.invoked_with == []
    assert draft.answered == []
    assert (
        draft.text == f"{NO_ANSWERS_BODY}\n\n## Could not be answered\n\n- What is X?"
    )


def test_a_model_failure_is_logged_and_raised_so_a_resumed_run_can_try_again(caplog):
    FakeWriterLLM.reply = RuntimeError("groq is down")
    x = Subtask(description="What is X?")
    state = _state([x], [_found(x, "X is 1.", GOOD)], [_review(x, kept=[GOOD.url])])
    with (
        caplog.at_level(logging.INFO, logger=writer_module.logger.name),
        pytest.raises(RuntimeError, match="groq is down"),
    ):
        writer(state)
    assert [r.levelno for r in caplog.records if r.levelno >= logging.ERROR] == [
        logging.ERROR
    ]
    assert "writer failed" in caplog.text


def test_the_log_has_counts_and_sizes_but_not_the_report(caplog):
    x = Subtask(description="What is X?")
    state = _state(
        [x], [_found(x, "SECRET ANSWER", GOOD)], [_review(x, kept=[GOOD.url])]
    )
    with caplog.at_level(logging.INFO, logger=writer_module.logger.name):
        writer(state)
    assert "writer start answered=1 unanswered=0" in caplog.text
    assert "writer done chars=" in caplog.text
    assert "SECRET ANSWER" not in caplog.text
    assert "The body of the report." not in caplog.text


# --- a reply that was cut off at the output limit is never turned into a report ---


def _reply(finish_reason, tokens=1234):
    return SimpleNamespace(
        content="A report that stops mid-sen",
        response_metadata={
            "finish_reason": finish_reason,
            "token_usage": {"completion_tokens": tokens},
        },
    )


def _one_subtask_state():
    x = Subtask(description="What is X?")
    return _state([x], [_found(x, "X is 1.", GOOD)], [_review(x, kept=[GOOD.url])])


def test_a_reply_cut_off_at_the_output_limit_is_an_error_not_a_report(caplog):
    FakeWriterLLM.reply = _reply("length")
    with (
        caplog.at_level(logging.INFO, logger=writer_module.logger.name),
        pytest.raises(WriterOutputCutOff, match="cut off.*completion_tokens=1234"),
    ):
        writer(_one_subtask_state())
    assert "writer failed" in caplog.text


def test_a_reply_that_finished_normally_is_accepted_and_the_log_says_why_it_stopped(
    caplog,
):
    FakeWriterLLM.reply = _reply("stop", tokens=321)
    with caplog.at_level(logging.INFO, logger=writer_module.logger.name):
        draft = writer(_one_subtask_state())["draft"]
    assert draft.text.startswith("A report that stops mid-sen")
    assert "finish_reason=stop completion_tokens=321" in caplog.text


def test_a_reply_with_no_metadata_is_accepted():
    # The fakes of other tests (and any provider that sends none) carry no metadata.
    draft = writer(_one_subtask_state())["draft"]
    assert draft.text.startswith("The body of the report.")


def test_the_writer_is_told_not_to_fill_a_gap_with_its_own_knowledge():
    assert "do not fill the gap" in WRITER_SYSTEM_PROMPT
