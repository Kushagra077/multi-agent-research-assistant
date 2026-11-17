# Real run: a run waiting for review survived a restart of the API

Date: 2026-11-17 (times are UTC, from `/app/logs/logfile.log` and `docker inspect`).
Stack: `docker compose` (API and Streamlit page), checkpoints in a named volume. Planner in
demo mode (fixed plan); Tavily, Groq and Jev real.

| Time | Event |
|---|---|
| 01:07:43 | run `9a62311305de45bda4563eed2a38decf` started from the page |
| 01:07:54 | `ship_check ready=True reason=jev confidence=0.81`, then `human_review waiting revisions=0`: the run paused |
| 01:09:39 | the API container was restarted (`docker compose restart api`; `StartedAt` from `docker inspect`) |
| 01:10:04 | `api review ... action=approve`, then `human_review resumed action=approve`: the run was approved and finished |

The API accepts a review only for a run that is waiting (it answers 409 otherwise), so the
approval after the restart shows the pause was still there. The report that was approved was
the one drafted before the restart. No researcher, writer or ship-check line appears between
01:07:54 and 01:10:04 other than the resumed review, so nothing was redone.

This is the brief's check: pause, restart the process, resume, state intact. The same is also
covered offline by tests (a real `SqliteSaver` file and two separate "processes").
