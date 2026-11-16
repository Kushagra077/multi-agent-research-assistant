# Runs after merge. For every subtask whose newest attempt did not produce a usable
# answer, decides what to do next and records it in state["decisions"]:
#   retry_same       search again unchanged
#   retry_different  search again, skipping the sites the source gate rejected
#   escalate         stop and hand the subtask to a human
# Jev (ask_choice) picks the action, but the code decides when to stop: at
# MAX_ATTEMPTS the subtask escalates without asking Jev. If Jev is unreachable the
# default is also "escalate": a blind retry would spend search and LLM quota.
# The edge that acts on the decisions (retry_dispatch) is in build.py.

import logging

from research_assistant.decisions.jev import ask_choice
from research_assistant.schemas.findings import (
    Finding,
    Review,
    attempts_by_subtask,
    latest_per_subtask,
)
from research_assistant.schemas.plan import Subtask
from research_assistant.schemas.retry import RetryDecision
from research_assistant.state import GraphState
from research_assistant.tools.search import domain_of

logger = logging.getLogger(__name__)

# Total attempts per subtask, the first one included: 2 = one retry. Every retry spends
# a search, an LLM call and about three Jev calls, and the free tiers are small.
MAX_ATTEMPTS = 2

RETRY_QUESTION = (
    "A research subtask did not produce a usable answer. What should be done next?"
)
RETRY_OPTIONS = {
    "retry_same": (
        "Search again with the same query. Choose this when the problem looks "
        "temporary, such as a failed or rate-limited search, or when no pages were "
        "found at all."
    ),
    "retry_different": (
        "Search again but skip the websites whose pages were rejected as not useful. "
        "Choose this when pages were found but were off-topic, promotional or too "
        "shallow to answer the question."
    ),
    "escalate": (
        "Stop and hand this subtask to a human. Choose this when searching again is "
        "unlikely to help, for example when the sources were fine but the question "
        "cannot be answered from web pages."
    ),
}
# When Jev is unreachable: do not retry blindly, let a person look.
DEFAULT_ACTION = "escalate"

ANSWER_CHARS = 600


def _situation(
    subtask: Subtask, finding: Finding, review: Review | None, attempt: int
) -> str:
    """The text Jev judges: the subtask, how the attempt went, what the gate decided."""
    lines = [
        f"Subtask: {subtask.description}",
        f"Attempt {attempt} of at most {MAX_ATTEMPTS}.",
        f"Outcome of this attempt: {finding.status}.",
    ]
    if finding.status == "failed":
        lines.append(f"Error: {finding.error}")
    if finding.status == "found":
        lines.append(f"Answer found (start): {(finding.answer or '')[:ANSWER_CHARS]}")
    if review is not None:
        lines.append(
            "The answer addresses the subtask: "
            f"{'yes' if review.addresses_subtask else 'no'}."
        )
        for url, score in review.source_scores.items():
            verdict = "rejected" if url in review.dropped_urls else "kept"
            shown = "unscored" if score is None else f"scored {score:.2f} of 2"
            lines.append(f"Source {verdict}, {shown}: {url}")
    return "\n".join(lines)


def _rejected_domains(reviews: list[Review]) -> list[str]:
    """Sites the gate rejected on any attempt, each once, in the order first seen."""
    domains: dict[str, None] = {}
    for review in reviews:
        for url in review.dropped_urls:
            if domain := domain_of(url):
                domains[domain] = None
    return list(domains)


def _decide_one(
    subtask: Subtask,
    finding: Finding,
    attempt: int,
    reviews: list[Review],
) -> RetryDecision:
    # The review of this very attempt is the newest one (attempt numbers only grow).
    review = next((r for r in reversed(reviews) if r.attempt == attempt), None)

    if attempt >= MAX_ATTEMPTS:
        action, reason, confidence = "escalate", "cap", None
    else:
        decision = ask_choice(
            _situation(subtask, finding, review, attempt),
            RETRY_QUESTION,
            RETRY_OPTIONS,
            default=DEFAULT_ACTION,
        )
        action = str(decision.value)
        reason = "fallback" if decision.fallback else "jev"
        confidence = decision.confidence

    exclude = _rejected_domains(reviews) if action == "retry_different" else []
    logger.info(
        "decide subtask_id=%s attempt=%d action=%s reason=%s confidence=%s exclude=%s",
        subtask.id,
        attempt,
        action,
        reason,
        confidence,
        exclude,
    )
    return RetryDecision(
        subtask_id=subtask.id,
        attempt=attempt,
        action=action,
        reason=reason,
        confidence=confidence,
        exclude_domains=exclude,
    )


def decide(state: GraphState) -> dict:
    findings = state.get("findings", [])
    latest = latest_per_subtask(findings)
    attempts = attempts_by_subtask(findings)
    all_reviews = state.get("reviews", [])
    # Already decided, so running this node twice on the same attempt asks Jev once.
    decided = {(d.subtask_id, d.attempt) for d in state.get("decisions", [])}

    decisions = []
    for subtask in state["plan"].subtasks:
        finding = latest.get(subtask.id)
        if finding is None:
            continue
        attempt = attempts[subtask.id]
        reviews = [r for r in all_reviews if r.subtask_id == subtask.id]
        review = next((r for r in reviews if r.attempt == attempt), None)

        # A usable answer: found, and the Jev check said it addresses the subtask.
        if finding.status == "found" and (review is None or review.addresses_subtask):
            continue
        if (subtask.id, attempt) in decided:
            continue
        decisions.append(_decide_one(subtask, finding, attempt, reviews))

    logger.info("decide decisions=%d", len(decisions))
    return {"decisions": decisions}
