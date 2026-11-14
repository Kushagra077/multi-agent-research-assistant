# Join point after the parallel researchers. LangGraph already waits for every
# researcher before this node runs; what merge adds are the two Jev checks:
#   1. Source-quality gate (score): rate each source of a 'found' finding against its
#      subtask and drop the weak ones.
#   2. Health check (boolean): does the answer actually address its subtask?
# The result goes into state["reviews"]. state["findings"] is left exactly as the
# researchers produced it, so nothing is lost and a human can still see dropped sources.

import logging
from collections import Counter

from research_assistant.decisions.jev import ask_boolean, ask_score
from research_assistant.schemas.findings import Finding, Review, Source
from research_assistant.state import GraphState

logger = logging.getLogger(__name__)

SOURCE_QUESTION = "How useful is this web page for answering the research question?"
# Written worst to best: the position in the list is the level number (0, 1, 2).
SOURCE_LEVELS = [
    (
        "Not useful: unrelated to the question, mostly navigation or boilerplate, or a "
        "video or forum question that gives no answer."
    ),
    (
        "Partly useful: on the topic, but answers only part of the question, or looks "
        "unreliable."
    ),
    "Directly useful: clearly answers the question and comes from a credible source.",
]
# A source scoring below this is dropped. 1.0 = "partly useful" or better stays.
MIN_SOURCE_SCORE = 1.0
# Used when Jev is unreachable: a gate outage must not throw research away.
SOURCE_DEFAULT_SCORE = 2.0

ANSWER_QUESTION = (
    "Does the answer directly address the research question? Answer no if it only "
    "covers the general topic, or says it could not find the information."
)
# Used when Jev is unreachable: let the run continue; used_fallback records it.
ANSWER_DEFAULT = True


def _source_text(question: str, source: Source) -> str:
    return (
        f"Research question: {question}\n\n"
        f"Web page title: {source.title}\nURL: {source.url}\n\n{source.content}"
    )


def _review_finding(finding: Finding, question: str) -> Review:
    kept: list[str] = []
    dropped: list[str] = []
    used_fallback = False

    # One Jev call per source, one after another. Fine for 2 sources x 3 subtasks;
    # batching them into one request is the optimisation if this gets slow.
    for source in finding.sources:
        decision = ask_score(
            _source_text(question, source),
            SOURCE_QUESTION,
            SOURCE_LEVELS,
            default=SOURCE_DEFAULT_SCORE,
        )
        used_fallback = used_fallback or decision.fallback
        if decision.value >= MIN_SOURCE_SCORE:
            kept.append(source.url)
        else:
            dropped.append(source.url)
            logger.info(
                "merge dropped source subtask_id=%s url=%s score=%.2f",
                finding.subtask_id,
                source.url,
                decision.value,
            )

    if kept:
        decision = ask_boolean(
            f"Research question: {question}\n\nAnswer:\n{finding.answer}",
            ANSWER_QUESTION,
            default=ANSWER_DEFAULT,
        )
        addresses_subtask = bool(decision.value)
        used_fallback = used_fallback or decision.fallback
    else:
        # Every source was dropped, so the answer has nothing credible behind it.
        # No Jev call is needed to know that.
        addresses_subtask = False

    logger.info(
        "merge reviewed subtask_id=%s kept=%d dropped=%d addresses_subtask=%s "
        "used_fallback=%s",
        finding.subtask_id,
        len(kept),
        len(dropped),
        addresses_subtask,
        used_fallback,
    )
    return Review(
        subtask_id=finding.subtask_id,
        kept_urls=kept,
        dropped_urls=dropped,
        addresses_subtask=addresses_subtask,
        used_fallback=used_fallback,
    )


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

    questions = {s.id: s.description for s in state["plan"].subtasks}
    reviews = []
    # Only 'found' findings have sources and an answer to check. no_results and failed
    # ones get no review; the retry-vs-escalate decision (later) handles those.
    for finding in findings:
        if finding.status != "found":
            continue
        if finding.subtask_id not in questions:
            logger.warning("merge skipped finding not in plan: %s", finding.subtask_id)
            continue
        reviews.append(_review_finding(finding, questions[finding.subtask_id]))
    return {"reviews": reviews}
