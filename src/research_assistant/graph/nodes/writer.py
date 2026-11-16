# Drafts the report from the findings that passed the Jev checks. The model writes only
# the body; the code appends the sources and the open questions, so a url or an
# unanswered subtask can never be invented or silently dropped by the model.

import logging
import time

from langchain.messages import HumanMessage, SystemMessage

from research_assistant.llm.providers import get_writer_llm
from research_assistant.schemas.findings import (
    attempts_by_subtask,
    latest_per_subtask,
)
from research_assistant.schemas.report import Draft
from research_assistant.state import GraphState

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a research writer. You are given a research \
question and findings from separate researchers, one finding per subtask. Write a \
clear report that answers the question using only these findings.

Rules:
- Use only facts that appear in the findings. Do not add outside knowledge, and do \
not guess.
- Keep concrete details such as numbers, dates, names and units.
- If findings disagree, say so and state what each one claims.
- Start with a short summary of two or three sentences, then write one short section \
per finding with a heading.
- Remove references such as "(Source 2)": the sources are listed separately.
- Do not write a list of sources, and do not mention subtasks that are missing; both \
are added separately."""

NO_ANSWERS_BODY = "No subtask produced a usable answer, so no report could be written."


def _usable(state: GraphState) -> tuple[list, list]:
    """Split the plan into (subtask, finding, kept urls) with a usable answer, and the
    subtasks without one."""
    findings = state.get("findings", [])
    latest = latest_per_subtask(findings)
    attempts = attempts_by_subtask(findings)
    reviews = {(r.subtask_id, r.attempt): r for r in state.get("reviews", [])}

    answered, unanswered = [], []
    for subtask in state["plan"].subtasks:
        finding = latest.get(subtask.id)
        review = reviews.get((subtask.id, attempts.get(subtask.id, 0)))
        # Usable = found, and Jev did not say it misses the subtask. A found finding
        # with no review (the Jev check could not run) counts as usable.
        missed = review is not None and not review.addresses_subtask
        if finding is None or finding.status != "found" or missed:
            unanswered.append(subtask)
        else:
            # Only the sources that passed the gate; all of them if there is no review.
            kept = review.kept_urls if review else [s.url for s in finding.sources]
            answered.append((subtask, finding, kept))
    return answered, unanswered


def _findings_text(answered: list) -> str:
    return "\n\n".join(
        f"Subtask {number}: {subtask.description}\nFinding: {finding.answer}"
        for number, (subtask, finding, _) in enumerate(answered, start=1)
    )


def _sources_section(answered: list) -> str:
    blocks = []
    for subtask, _, kept in answered:
        urls = "\n".join(f"- {url}" for url in kept)
        blocks.append(f"### {subtask.description}\n{urls}")
    return "## Sources\n\n" + "\n\n".join(blocks)


def _open_questions_section(unanswered: list) -> str:
    items = "\n".join(f"- {subtask.description}" for subtask in unanswered)
    return "## Could not be answered\n\n" + items


def writer(state: GraphState) -> dict:
    answered, unanswered = _usable(state)
    logger.info(
        "writer start answered=%d unanswered=%d", len(answered), len(unanswered)
    )
    started = time.perf_counter()

    if answered:
        try:
            human_text = (
                f"Question: {state['question']}\n\nFindings:\n\n"
                f"{_findings_text(answered)}"
            )
            reply = get_writer_llm().invoke(
                [SystemMessage(WRITER_SYSTEM_PROMPT), HumanMessage(human_text)]
            )
            body = str(reply.content).strip()
        except Exception:
            # Like the planner, and unlike the researcher: a missing report is the end
            # of the run, not a result. Log where it stopped; the checkpoint lets a
            # resumed run start again from here without redoing the research.
            logger.exception("writer failed after %.1fs", time.perf_counter() - started)
            raise
        parts = [body, _sources_section(answered)]
    else:
        # Nothing to write from: do not ask the model to write anyway.
        parts = [NO_ANSWERS_BODY]
    if unanswered:
        parts.append(_open_questions_section(unanswered))

    draft = Draft(
        text="\n\n".join(parts),
        answered=[subtask.id for subtask, _, _ in answered],
        unanswered=[subtask.id for subtask in unanswered],
    )
    # Lengths only: the draft holds research text and stays out of the log.
    logger.info(
        "writer done chars=%d answered=%d unanswered=%d elapsed=%.1fs",
        len(draft.text),
        len(draft.answered),
        len(draft.unanswered),
        time.perf_counter() - started,
    )
    return {"draft": draft}
