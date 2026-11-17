# Real run: a subtask that lost its subject is rejected, retried, and escalated to a human

Date: 2026-11-17 (UTC times from the API log). Run `e90b79b0d06a440a9efd551c9f7af563`, started
and approved through the Streamlit page against the Docker stack.
Question: "What is the best way for an adult beginner to learn guitar?"

Demo mode (fixed plan) with a **deliberately vague third subtask**, imitating a planner slip seen
earlier the same day: "What are the best warm-up exercises and progression milestones for adult
beginners?" (no "guitar"). Tavily, Groq, Jev and the review were real. This run checked the fix
made that morning: Jev is now shown the original question as well as the subtask.

## What happened (11 Jev calls, 0.5 to 0.6 s each)

| Subtask | Attempt | Source scores | Jev decision |
|---|---|---|---|
| 1. practice routine (guitar) | 1 | 1.90, 1.83 (kept) | answer addresses it (p = 0.97) |
| 2. lessons, teacher or books (guitar) | 1 | 1.53, 1.28 (kept) | answer addresses it (p = 0.98) |
| 3. warm-ups and milestones (vague) | 1 | thebodycoach.com **0.14**, nerdfitness.com **0.01** (dropped) | `retry_different`, confidence **1.00** (retry_same 0.00, escalate 0.00); skip those two sites |
| 3. | 2 | muscleandstrength.com **0.01**, harvardhealth.org **0.02** (dropped) | retry limit reached: **escalate** (`reason=cap`, Jev not asked) |

- Earlier the same day, a gym-exercise page scored **1.42** for the same kind of subtask (before
  the original question was shown to Jev). Here every fitness page scored 0.01 to 0.14.
- The writer wrote the report from subtasks 1 and 2 (`answered=2 unanswered=1`,
  `finish_reason=stop`, 496 completion tokens). The delivered report ends with a
  "Could not be answered" section listing subtask 3, and none of the rejected answer's
  content (push-ups, squats, plyometrics) is in it.
- The ship check did not ask Jev: `ready=False reason=unanswered`.
- Timeline: first Jev call 01:23:06, paused for review 01:23:17 (11 s), approved through the
  page 01:23:54. The reviewer chose to approve the partial report, which says plainly what it
  could not answer.

## What it does not show

- The subtask text was vague on purpose, so this is a stress test of the safety net, not a
  planner failure that happened on its own in this run.
- Jev judged the pages against the original question, which a vague subtask cannot hide. A
  vague subtask that still names a plausible subject would not be caught this way.

Screenshots: `docs/screenshots/02-escalated-subtask.png` (this run) and
`docs/screenshots/01-approved-run.png` (the approved run of the restart test).
