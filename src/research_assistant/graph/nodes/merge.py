# Join point after the parallel researchers. LangGraph already waits for every
# researcher before this node runs; what merge adds are the two Jev checks:
#   1. Source-quality gate: drop sources with too little text (a plain rule), then rate
#      each remaining source against its subtask with Jev (score) and drop weak ones.
#   2. Health check (boolean): does the answer actually address its subtask?
# The result is appended to state["reviews"]. state["findings"] is left exactly as the
# researchers produced it, so nothing is lost and a human can still see dropped sources.
# merge runs again after a retry, so it only reviews what has not been reviewed yet:
# the newest finding of each subtask, once per attempt.

import logging
from collections import Counter

from research_assistant.decisions.jev import ask_boolean, ask_score
from research_assistant.schemas.findings import (
    Finding,
    Review,
    Source,
    attempts_by_subtask,
    latest_per_subtask,
)
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
# A page with less text than this is dropped without asking Jev: there is too little
# to judge or to write from (a real run kept a 400-character Instagram post). The
# check is a plain rule, so it also saves a Jev call. Pages in real runs had 2,000+.
MIN_SOURCE_CHARS = 300
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


def _review_finding(finding: Finding, question: str, attempt: int) -> Review:
    kept: list[str] = []
    dropped: list[str] = []
    scores: dict[str, float | None] = {}
    used_fallback = False

    # One Jev call per source, one after another. Fine for 2 sources x 3 subtasks;
    # batching them into one request is the optimisation if this gets slow.
    for source in finding.sources:
        if len(source.content) < MIN_SOURCE_CHARS:
            # No Jev score exists for it, so the score is None (as when Jev is down),
            # but this is not an outage, so used_fallback is left alone.
            scores[source.url] = None
            dropped.append(source.url)
            logger.info(
                "merge dropped source subtask_id=%s url=%s too_short chars=%d",
                finding.subtask_id,
                source.url,
                len(source.content),
            )
            continue
        decision = ask_score(
            _source_text(question, source),
            SOURCE_QUESTION,
            SOURCE_LEVELS,
            default=SOURCE_DEFAULT_SCORE,
        )
        used_fallback = used_fallback or decision.fallback
        # On a fallback the value is our default, not Jev's opinion: record None.
        scores[source.url] = None if decision.fallback else float(decision.value)
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
        "merge reviewed subtask_id=%s attempt=%d kept=%d dropped=%d "
        "addresses_subtask=%s used_fallback=%s",
        finding.subtask_id,
        attempt,
        len(kept),
        len(dropped),
        addresses_subtask,
        used_fallback,
    )
    return Review(
        subtask_id=finding.subtask_id,
        attempt=attempt,
        kept_urls=kept,
        dropped_urls=dropped,
        source_scores=scores,
        addresses_subtask=addresses_subtask,
        used_fallback=used_fallback,
    )


def merge(state: GraphState) -> dict:
    # Only the newest finding of each subtask counts; older attempts are history.
    findings = state.get("findings", [])
    latest = latest_per_subtask(findings)
    attempts = attempts_by_subtask(findings)
    counts = Counter(finding.status for finding in latest.values())
    logger.info(
        "merge findings=%d found=%d no_results=%d failed=%d",
        len(latest),
        counts["found"],
        counts["no_results"],
        counts["failed"],
    )

    questions = {s.id: s.description for s in state["plan"].subtasks}
    # What is already reviewed, so a second pass over the same attempt costs no Jev call.
    already_reviewed = {(r.subtask_id, r.attempt) for r in state.get("reviews", [])}
    reviews = []
    # Only 'found' findings have sources and an answer to check. no_results and failed
    # ones get no review; the retry-vs-escalate decision handles those.
    for subtask_id, finding in latest.items():
        if finding.status != "found":
            continue
        if subtask_id not in questions:
            logger.warning("merge skipped finding not in plan: %s", subtask_id)
            continue
        attempt = attempts[subtask_id]
        if (subtask_id, attempt) in already_reviewed:
            continue
        reviews.append(_review_finding(finding, questions[subtask_id], attempt))
    return {"reviews": reviews}
