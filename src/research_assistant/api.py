"""HTTP API for the research assistant.

    POST /runs                     start a run; returns at once with its thread_id
    GET  /runs/{thread_id}         status, plan, sources, decisions, report, ship verdict
    POST /runs/{thread_id}/resume  continue a run that failed or was interrupted
    GET  /health

A run takes a minute or more, so POST /runs only starts it (in a background thread) and
the client polls GET /runs/{thread_id}. State lives in the SQLite checkpoints, which is
also why /resume works, even after the server itself was restarted. Run it with:

    uvicorn research_assistant.api:app
"""

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from research_assistant.schemas.findings import attempts_by_subtask, latest_per_subtask
from research_assistant.schemas.report import ShipVerdict

logger = logging.getLogger(__name__)

# langgraph reads this once, when it is first imported (the same reason main.py imports
# it inside main()). Nothing above imports langgraph; build_graph is imported later.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

Status = Literal["running", "failed", "interrupted", "done"]


# --- request and response shapes -------------------------------------------


class StartRun(BaseModel):
    question: str
    thread_id: str | None = Field(
        default=None, description="Name of the run. Generated if omitted."
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value

    @field_validator("thread_id")
    @classmethod
    def thread_id_is_simple(cls, value: str | None) -> str | None:
        if value is not None and not THREAD_ID_PATTERN.match(value):
            raise ValueError("thread_id may use letters, digits, - and _ (max 64).")
        return value


class Started(BaseModel):
    thread_id: str
    status: Status


class SourceView(BaseModel):
    url: str
    # None = no Jev check ran for this source (no review yet, or it was too short).
    kept: bool | None
    score: float | None


class SubtaskView(BaseModel):
    id: uuid.UUID
    description: str
    status: Literal["pending", "found", "no_results", "failed"]
    attempts: int
    answer: str | None
    addresses_subtask: bool | None
    sources: list[SourceView]
    next_step: str | None = Field(
        description="Jev's latest retry-vs-escalate choice for this subtask, if any."
    )
    needs_human: bool


class RunView(BaseModel):
    thread_id: str
    status: Status
    question: str | None
    subtasks: list[SubtaskView]
    report: str | None
    ship: ShipVerdict | None
    error: str | None


# --- turning saved graph state into a response -------------------------------


def _subtask_views(values: dict) -> list[SubtaskView]:
    plan = values.get("plan")
    if plan is None:
        return []
    findings = values.get("findings", [])
    latest = latest_per_subtask(findings)
    attempts = attempts_by_subtask(findings)
    reviews = {(r.subtask_id, r.attempt): r for r in values.get("reviews", [])}
    decisions = {}
    for decision in values.get("decisions", []):  # later ones come later in the list
        decisions[decision.subtask_id] = decision

    views = []
    for subtask in plan.subtasks:
        finding = latest.get(subtask.id)
        count = attempts.get(subtask.id, 0)
        review = reviews.get((subtask.id, count))
        decision = decisions.get(subtask.id)

        sources = []
        for source in finding.sources if finding else []:
            kept = None if review is None else source.url in review.kept_urls
            score = None if review is None else review.source_scores.get(source.url)
            sources.append(SourceView(url=source.url, kept=kept, score=score))

        views.append(
            SubtaskView(
                id=subtask.id,
                description=subtask.description,
                status=finding.status if finding else "pending",
                attempts=count,
                answer=finding.answer if finding else None,
                addresses_subtask=review.addresses_subtask if review else None,
                sources=sources,
                next_step=decision.action if decision else None,
                needs_human=bool(
                    decision
                    and decision.action == "escalate"
                    and decision.attempt == count
                ),
            )
        )
    return views


# --- the app -----------------------------------------------------------------


def create_app(
    db_path: str | None = None, *, configure_logging: bool = True
) -> FastAPI:
    db_path = db_path or os.getenv("RESEARCH_DB", "checkpoints.db")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Imported here, after the strict-mode flag above has been set.
        from langgraph.checkpoint.sqlite import SqliteSaver

        from research_assistant.graph.build import build_graph
        from research_assistant.logging_config import setup_logging

        if configure_logging:
            setup_logging()
        with SqliteSaver.from_conn_string(db_path) as saver:
            app.state.graph = build_graph(saver)
            # Runs this process is working on, and the error of runs that failed. Both
            # are in memory only: after a restart a half-finished run simply shows as
            # "interrupted" (its checkpoint still says work is left) and can be resumed.
            app.state.running = set()
            app.state.errors = {}
            yield

    app = FastAPI(title="Multi-agent research assistant", lifespan=lifespan)

    def config_for(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def status_of(request: Request, thread_id: str) -> Status | None:
        """None means the thread is unknown."""
        state = request.app.state
        if thread_id in state.running:
            return "running"
        if thread_id in state.errors:
            return "failed"
        snapshot = state.graph.get_state(config_for(thread_id))
        if not snapshot.values:
            return None
        return "done" if not snapshot.next else "interrupted"

    def execute(app: FastAPI, thread_id: str, graph_input: dict | None) -> None:
        """The background job: run the graph, and record how it ended."""
        try:
            app.state.graph.invoke(
                graph_input, config_for(thread_id), durability="sync"
            )
        except Exception as e:
            logger.exception("api run failed thread_id=%s", thread_id)
            app.state.errors[thread_id] = str(e) or type(e).__name__
        finally:
            app.state.running.discard(thread_id)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/runs", status_code=202, response_model=Started)
    def start_run(
        body: StartRun, background: BackgroundTasks, request: Request
    ) -> Started:
        thread_id = body.thread_id or uuid.uuid4().hex
        if status_of(request, thread_id) is not None:
            raise HTTPException(409, f"A run named {thread_id!r} already exists.")
        request.app.state.running.add(thread_id)
        background.add_task(
            execute, request.app, thread_id, {"question": body.question}
        )
        logger.info("api run started thread_id=%s", thread_id)
        return Started(thread_id=thread_id, status="running")

    @app.post("/runs/{thread_id}/resume", status_code=202, response_model=Started)
    def resume_run(
        thread_id: str, background: BackgroundTasks, request: Request
    ) -> Started:
        status = status_of(request, thread_id)
        if status is None:
            raise HTTPException(404, "No such run.")
        if status not in ("failed", "interrupted"):
            raise HTTPException(409, f"A run that is {status} cannot be resumed.")
        request.app.state.errors.pop(thread_id, None)
        request.app.state.running.add(thread_id)
        # None as the input means "continue from the last checkpoint".
        background.add_task(execute, request.app, thread_id, None)
        logger.info("api run resumed thread_id=%s", thread_id)
        return Started(thread_id=thread_id, status="running")

    @app.get("/runs/{thread_id}", response_model=RunView)
    def get_run(thread_id: str, request: Request) -> RunView:
        status = status_of(request, thread_id)
        if status is None:
            raise HTTPException(404, "No such run.")
        values = request.app.state.graph.get_state(config_for(thread_id)).values
        draft = values.get("draft")
        return RunView(
            thread_id=thread_id,
            status=status,
            question=values.get("question"),
            subtasks=_subtask_views(values),
            report=draft.text if draft else None,
            ship=values.get("ship"),
            error=request.app.state.errors.get(thread_id),
        )

    return app


app = create_app()
