# Join point after the parallel researchers.
# LangGraph already waits for every researcher before this node runs, so there is
# nothing to do yet. Later phases attach here: the Jev source-quality gate, then the writer.

import logging
from collections import Counter

from research_assistant.state import GraphState

logger = logging.getLogger(__name__)


def merge(state: GraphState) -> dict:
    findings = state.get("findings", [])
    counts = Counter(finding.status for finding in findings)
    logger.info(
        "merge findings=%d found=%d no_results=%d failed=%d",
        len(findings),
        counts["found"],
        counts["no_results"],
        counts["failed"],
    )
    return {}
