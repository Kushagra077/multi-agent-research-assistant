import logging
from uuid import uuid4

from research_assistant.graph.nodes import merge as merge_module
from research_assistant.graph.nodes.merge import merge
from research_assistant.schemas.findings import Finding, Source

SOURCE = Source(url="https://a.com", title="Page A", content="Fact from A.")


def _findings() -> list[Finding]:
    return [
        Finding(subtask_id=uuid4(), status="found", answer="x", sources=[SOURCE]),
        Finding(subtask_id=uuid4(), status="no_results"),
        Finding(subtask_id=uuid4(), status="failed", error="boom"),
    ]


def test_merge_still_changes_nothing():
    assert merge({"question": "q", "findings": _findings()}) == {}


def test_merge_logs_a_count_per_status(caplog):
    with caplog.at_level(logging.INFO, logger=merge_module.logger.name):
        merge({"question": "q", "findings": _findings()})
    [record] = caplog.records
    assert "findings=3 found=1 no_results=1 failed=1" in record.getMessage()


def test_merge_with_no_findings_logs_zeros_instead_of_crashing(caplog):
    with caplog.at_level(logging.INFO, logger=merge_module.logger.name):
        assert merge({"question": "q"}) == {}
    assert "findings=0 found=0 no_results=0 failed=0" in caplog.text
