# Thin wrapper around the TypeSafe SDK (Jev). Nodes call ask_* and get a Decision back;
# they never see SDK objects. Every call is logged, and any Jev failure returns the
# caller's default (fallback=True) instead of raising, so the graph never stops because
# Jev is down.

import logging
import time

from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel
from typesafe_sdk import (
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeAPIError,
    TypeSafeClient,
    TypeSafeError,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Pinned (not "jev-latest") so the decision log stays comparable across runs.
JEV_MODEL = "jev-1.13.0"
# Same discipline as the LLM clients: fail fast, retry at most once.
JEV_TIMEOUT_SECONDS = 10
JEV_MAX_RETRIES = 1

# Id of the single question in a request. It is our own label: the model never sees it.
_QUESTION_ID = "q"


class Decision(BaseModel):
    # bool for ask_boolean, float (the expected level) for ask_score, str (the name of
    # the chosen option) for ask_choice.
    value: bool | float | str
    confidence: float | None
    fallback: bool
    model: str | None


def _trace_inputs(inputs: dict) -> dict:
    """What a LangSmith trace records of an ask_* call: everything except the page text,
    which is replaced by its length (the same rule as the log lines). Tracing is off
    unless LANGSMITH_TRACING=true; then this runs, and nothing else changes."""
    shown = dict(inputs)
    shown["state_chars"] = len(shown.pop("state", ""))
    return shown


def _make_client() -> TypeSafeClient:
    return TypeSafeClient(
        model=JEV_MODEL,
        retry=RetryPolicy(max_retries=JEV_MAX_RETRIES, timeout=JEV_TIMEOUT_SECONDS),
    )


def _fallback(kind: str, question: str, default: bool | float | str, e: TypeSafeError):
    """Log a failed Jev call and return the caller's default, marked as a fallback."""
    # Narrower than the researcher's `except Exception`: the SDK raises only
    # TypeSafeError subclasses, so anything else is our own bug and should surface.
    # A 4xx other than 429 means our request was wrong (bad key, bad payload).
    # That will fail every time, so it is logged louder than an outage.
    is_our_bug = (
        isinstance(e, TypeSafeAPIError) and 400 <= e.status < 500 and e.status != 429
    )
    log = logger.error if is_our_bug else logger.warning
    log(
        "jev %s failed, using default=%s question=%r error=%s: %s",
        kind,
        default,
        question,
        type(e).__name__,
        e,
        exc_info=True,
    )
    return Decision(value=default, confidence=None, fallback=True, model=None)


@traceable(run_type="tool", name="jev_boolean", process_inputs=_trace_inputs)
def ask_boolean(
    state: str, question: str, *, default: bool, threshold: float = 0.5
) -> Decision:
    """Ask Jev a yes/no question about `state`.

    `default` is returned (with fallback=True) if Jev cannot be reached or rejects the
    request. `threshold` is the probability of yes at or above which the answer is True.
    The confidence is how sure Jev is of the answer it gave: max(p, 1 - p).
    """
    started = time.perf_counter()
    try:
        response = _make_client().system_one(
            state=state, questions={_QUESTION_ID: Noul(instructions=question)}
        )
        probability = response.nouls[_QUESTION_ID].noul
    except TypeSafeError as e:
        return _fallback("boolean", question, default, e)

    value = probability >= threshold
    confidence = max(probability, 1 - probability)
    # state is page text: log its length, never the text (same rule as the researcher).
    logger.info(
        "jev boolean question=%r answer=%s p=%.3f threshold=%.2f confidence=%.3f "
        "state_chars=%d model=%s elapsed=%.1fs",
        question,
        value,
        probability,
        threshold,
        confidence,
        len(state),
        response.model,
        time.perf_counter() - started,
    )
    return Decision(
        value=value, confidence=confidence, fallback=False, model=response.model
    )


@traceable(run_type="tool", name="jev_score", process_inputs=_trace_inputs)
def ask_score(
    state: str, question: str, levels: list[str], *, default: float
) -> Decision:
    """Ask Jev to rate `state` on a scale described by `levels`.

    `levels` are written from worst to best; the position in the list is the level
    number (levels[0] is level 0). The returned value is the expected level, so it can
    fall between whole numbers: 1.7 means "between level 1 and level 2, nearer 2".
    `default` is returned (with fallback=True) if Jev cannot be reached.
    """
    # Outside the try on purpose: a bad `levels` is our own bug and should raise.
    if len(levels) < 2:
        raise ValueError("A score needs at least two levels.")

    started = time.perf_counter()
    try:
        response = _make_client().system_one(
            state=state,
            questions={_QUESTION_ID: Score(instructions=question, criteria=levels)},
        )
        answer = response.scores[_QUESTION_ID]
    except TypeSafeError as e:
        return _fallback("score", question, default, e)

    logger.info(
        "jev score question=%r score=%.2f levels=%d confidence=%.3f "
        "state_chars=%d model=%s elapsed=%.1fs",
        question,
        answer.score,
        len(levels),
        answer.confidence,
        len(state),
        response.model,
        time.perf_counter() - started,
    )
    return Decision(
        value=answer.score,
        confidence=answer.confidence,
        fallback=False,
        model=response.model,
    )


@traceable(run_type="tool", name="jev_choice", process_inputs=_trace_inputs)
def ask_choice(
    state: str, question: str, options: dict[str, str], *, default: str
) -> Decision:
    """Ask Jev to pick the one option that fits `state` best.

    `options` maps each option's name to a description of when it applies. The returned
    value is the name of the chosen option. Confidence is Jev's own: high when one
    option clearly stands out, low when the probability is spread over several.
    `default` must be one of the options; it is returned (with fallback=True) if Jev
    cannot be reached.
    """
    # Outside the try on purpose: both are our own bugs and should raise.
    if len(options) < 2:
        raise ValueError("A choice needs at least two options.")
    if default not in options:
        raise ValueError(f"default {default!r} is not one of the options.")

    started = time.perf_counter()
    try:
        response = _make_client().system_one(
            state=state,
            questions={_QUESTION_ID: Choice(instructions=question, criteria=options)},
        )
        answer = response.choices[_QUESTION_ID]
    except TypeSafeError as e:
        return _fallback("choice", question, default, e)

    # The probabilities show how close the runner-up was; they go in the log, rounded.
    probabilities = {name: round(p, 2) for name, p in answer.probabilities.items()}
    logger.info(
        "jev choice question=%r choice=%s confidence=%.3f probabilities=%s "
        "state_chars=%d model=%s elapsed=%.1fs",
        question,
        answer.choice,
        answer.confidence,
        probabilities,
        len(state),
        response.model,
        time.perf_counter() - started,
    )
    return Decision(
        value=answer.choice,
        confidence=answer.confidence,
        fallback=False,
        model=response.model,
    )
