"""Streamlit review page. Start it with:

    streamlit run src/research_assistant/review_ui.py

It shows a run (plan, sources with their Jev scores, Jev's ship verdict, the draft) and,
when the run is waiting, lets a person approve it, reject it with feedback, or take over.
It talks to the API only (RESEARCH_API_URL, default http://127.0.0.1:8000).
"""

import os
import time

import streamlit as st

from research_assistant.ui_client import ApiClient, ApiError

API_URL = os.getenv("RESEARCH_API_URL", "http://127.0.0.1:8000")
REFRESH_SECONDS = 2

APPROVE, REJECT, TAKE_OVER = "Approve", "Reject with feedback", "Take over (edit)"
SHIP_REASONS = {
    "jev": "Jev's judgement",
    "unanswered": "a subtask has no usable answer, so Jev was not asked",
    "fallback": "Jev was unreachable, so the safe default was used",
}

client = ApiClient(API_URL)

st.set_page_config(page_title="Research review", layout="wide")
st.title("Research assistant: review")

# --- sidebar: start a run, or open one ----------------------------------------
with st.sidebar:
    st.header("Start a run")
    question = st.text_area("Research question", key="new_question")
    if st.button("Start"):
        try:
            # Set before the "Run id" box below is created, so the box shows it.
            st.session_state["thread_id"] = client.start(question)["thread_id"]
        except ApiError as e:
            st.error(str(e))

    st.header("Open a run")
    st.text_input("Run id", key="thread_id")
    st.caption(f"API: {API_URL}")

thread_id = st.session_state.get("thread_id", "").strip()
if not thread_id:
    st.info("Start a run in the sidebar, or enter the id of an existing one.")
    st.stop()

try:
    run = client.get(thread_id)
except ApiError as e:
    st.error(str(e))
    st.stop()
if run is None:
    st.warning(f"There is no run with the id {thread_id!r}.")
    st.stop()

status = run["status"]
st.subheader(run["question"] or "(starting...)")

# --- status ---------------------------------------------------------------------
if status == "running":
    st.info("Working on it. This page refreshes by itself.")
elif status == "awaiting_review":
    st.warning("The report is drafted and waiting for your decision (below).")
elif status == "done":
    st.success("Done.")
elif status == "failed":
    st.error(f"The run failed: {run['error']}")
else:  # interrupted
    st.warning("The run stopped part-way (for example, the server restarted).")

# --- what the researchers found ------------------------------------------------
for subtask in run["subtasks"]:
    label = f"{subtask['description']}  [{subtask['status']}]"
    with st.expander(label, expanded=subtask["needs_human"]):
        if subtask["attempts"] > 1:
            st.caption(f"{subtask['attempts']} attempts (the newest is shown)")
        if subtask["needs_human"]:
            st.error(
                "Jev could not get a usable answer for this one: it needs a human."
            )
        elif subtask["next_step"]:
            st.caption(f"Jev's choice after a weak attempt: {subtask['next_step']}")
        if subtask["answer"]:
            st.write(subtask["answer"])
        if subtask["sources"]:
            rows = [
                {
                    "verdict": {True: "kept", False: "dropped", None: "not scored"}[
                        source["kept"]
                    ],
                    "score (0-2)": source["score"],
                    "url": source["url"],
                }
                for source in subtask["sources"]
            ]
            st.table(rows)

# --- Jev's ship verdict and the draft -------------------------------------------
ship = run["ship"]
if ship:
    why = SHIP_REASONS[ship["reason"]]
    confidence = f", confidence {ship['confidence']:.2f}" if ship["confidence"] else ""
    if ship["ready"]:
        st.success(f"Jev: the draft looks ready for review ({why}{confidence}).")
    else:
        st.warning(f"Jev: the draft is NOT ready ({why}{confidence}).")

if run["report"]:
    st.markdown("### Draft report")
    with st.container(border=True):
        st.markdown(run["report"])

if run["feedback"]:
    st.markdown("### Earlier rejections")
    for number, note in enumerate(run["feedback"], start=1):
        st.write(f"{number}. {note}")

# --- the decision ---------------------------------------------------------------
if status == "awaiting_review":
    st.markdown("### Your decision")
    with st.form("review"):
        choice = st.radio("Decision", [APPROVE, REJECT, TAKE_OVER])
        feedback = st.text_area("Feedback for the writer (used by 'Reject')")
        edited = st.text_area(
            "Your own final report (used by 'Take over')",
            value=run["report"] or "",
            height=300,
        )
        submitted = st.form_submit_button("Submit decision")
    if submitted:
        try:
            if choice == APPROVE:
                client.review(thread_id, "approve")
            elif choice == REJECT:
                client.review(thread_id, "reject", feedback=feedback)
            else:
                client.review(thread_id, "edit", text=edited)
        except ApiError as e:
            st.error(str(e))
        else:
            st.rerun()

resume_clicked = status in ("failed", "interrupted") and st.button(
    "Resume from the last checkpoint"
)
if resume_clicked:
    try:
        client.resume(thread_id)
    except ApiError as e:
        st.error(str(e))
    else:
        st.rerun()

if status == "done" and run["final"]:
    how = {"approved": "approved as drafted", "edited": "replaced by the reviewer"}
    st.markdown(f"### Delivered report ({how[run['final']['how']]})")
    with st.container(border=True):
        st.markdown(run["final"]["text"])
    st.download_button("Download as Markdown", run["final"]["text"], "report.md")

if status == "running":
    time.sleep(REFRESH_SECONDS)
    st.rerun()
