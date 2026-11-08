import pytest

from research_assistant.graph.nodes.intake import intake


def test_intake_strips_whitespace():
    assert intake({"question": "  What is RAG?  "}) == {"question": "What is RAG?"}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_intake_rejects_blank_question(blank):
    with pytest.raises(ValueError):
        intake({"question": blank})
