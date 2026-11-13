import logging

import pytest

from research_assistant.graph.nodes import intake as intake_module
from research_assistant.graph.nodes.intake import intake


def test_intake_strips_whitespace():
    assert intake({"question": "  What is RAG?  "}) == {"question": "What is RAG?"}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_intake_rejects_blank_question(blank):
    with pytest.raises(ValueError):
        intake({"question": blank})


def test_intake_logs_the_accepted_question(caplog):
    with caplog.at_level(logging.INFO, logger=intake_module.logger.name):
        intake({"question": "  What is RAG?  "})
    assert any("intake ok" in r.getMessage() for r in caplog.records)
    assert "What is RAG?" in caplog.text


def test_intake_logs_a_warning_when_it_rejects_a_blank_question(caplog):
    with (
        caplog.at_level(logging.INFO, logger=intake_module.logger.name),
        pytest.raises(ValueError),
    ):
        intake({"question": "   "})
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "blank" in record.getMessage()
