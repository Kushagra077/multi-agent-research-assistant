# Design notes

Why the project is built the way it is, what broke along the way, and what was and was not
checked on real runs. Last updated 2026-11-18.

The README says what the project does. This file is for someone who wants to judge the
engineering: the decisions, the trade-offs I accepted, and the evidence behind each claim.
Numbers link to notes in [docs/evidence/](evidence/).

## Contents

1. [Principles](#principles)
2. [The graph](#the-graph)
3. [The four Jev decisions](#the-four-jev-decisions)
4. [Checkpointing and human review](#checkpointing-and-human-review)
5. [Models, search and rate limits](#models-search-and-rate-limits)
6. [What broke](#what-broke)
7. [What was measured](#what-was-measured)
8. [What was not verified, and known limits](#what-was-not-verified-and-known-limits)
9. [Testing](#testing)

---

## Principles

Four rules that most decisions below follow from.

1. **Failures are data, not exceptions.** Researchers run in parallel and are joined, so one
   raised exception would lose the other branches' results. Every researcher outcome is a
   typed `Finding` (`found`, `no_results`, `failed`), and a `found` finding validates itself
   (it must have an answer and at least one source).
2. **The model never decides what code can decide.** Ids, statuses, source lists, the
   "could not be answered" list and the retry limit are all made by code. A model that is
   asked to produce them will eventually invent one.
3. **Every model call fails soft, with a default chosen for that call.** When Jev is
   unreachable, a source is kept (do not throw research away), an answer is assumed to
   address its subtask (and flagged), a weak subtask escalates to a person (do not spend
   quota on a blind retry), and a draft is "not ready" (a person looks at it). The
   `fallback` flag on every `Decision` separates a real answer from a default.
4. **A claim in the README needs a run behind it.** Where I could not check something, the
   README and this file say so.

## The graph

```
intake -> planner -> researcher (one per subtask, parallel) -> merge -> decide
decide -> researcher (retry)  |  writer
writer -> ship_check -> human_review -> END  (reject -> writer)
```

**Orchestrator-worker with `Send`, as a custom `StateGraph`.** The planner returns a `Plan`;
code (not an LLM) dispatches one `Send("researcher", ...)` per subtask; a reducer collects the
results. Rejected alternative: an LLM supervisor that calls workers as tools. It is more
flexible, but the state is less visible and harder to checkpoint, replay and interrupt, and
the workers here are all the same kind (this is not a router that picks a specialist).

**A researcher sees only its own subtask.** Isolation and a smaller prompt, which matters on a
free token budget. The cost is that a researcher cannot use the original question. The
planner prompt compensates (each subtask must be self-contained), and when that failed (see
[What broke](#what-broke)) the fix was to give the original question to Jev, not to the
researcher.

**`findings`, `reviews` and `decisions` are append-only lists** (`operator.add`). Parallel
branches each return `[finding]`; a retry appends one more. Nothing is overwritten, so the
history of every attempt stays in the checkpoint and on the review page. Two things are
*derived* from that history instead of stored, so they cannot drift: the attempt count
(`attempts_by_subtask(findings)`) and the newest finding of each subtask
(`latest_per_subtask`).

## The four Jev decisions

Jev answers with a typed value and a probability, so there is nothing to parse. A small
wrapper (`decisions/jev.py`) turns each call into a `Decision(value, confidence, fallback,
model)` and is the only place that touches the SDK. The model is pinned to `jev-1.13.0` (not
"latest") so logged results stay comparable. Only `TypeSafeError` is caught: anything else is
my bug and should surface. A 4xx other than 429 is logged at error level, because it will fail
every time.

| # | Decision | Type | Outcome | If Jev is down |
|---|---|---|---|---|
| 1 | Is this source useful for the subtask? | score, 3 levels | drop below 1.0 | keep |
| 2 | Does this answer address its subtask? | boolean | flag in the review | assume yes |
| 3 | Retry, retry differently, or escalate? | choice of 3 | next step for the subtask | escalate |
| 4 | Is the draft ready for a person? | boolean | shown to the reviewer | not ready |

**Decisions 1 and 2 (`merge`).** A page is dropped by a plain rule first if it has under 300
characters (too little to judge, and it saves a call), then scored against the subtask. Jev
is shown both the original question and the subtask, because a page can answer a subtask as
worded and still be about the wrong subject. The score is stored per URL in the `Review`
(`None` when Jev was unreachable, so a default is never shown as a real score). The review
stores URLs, not page text, so text is not duplicated in every checkpoint.

**Decision 3 (`decide`, then a conditional edge).** It is a node plus an edge, not one edge
function, because an edge cannot write state and the reason and confidence are evidence worth
keeping. Jev chooses the action; **code decides when to stop**: at two attempts the subtask
escalates without asking Jev, and the attempt count can only grow, so the loop terminates
regardless of what the model says. "Retry differently" means the same query with the domains
of the rejected sources excluded from the search. I chose that over an LLM query rewrite
because the gate has already told us which sites were useless, so the evidence is free and no
free-tier quota is spent. Its limit: it can avoid bad sites, not fix a bad query.

**Decision 4 (`ship_check`).** The model writes only the body of the report. Code appends the
sources (only URLs that passed the gate) and a "could not be answered" section, so a link
cannot be invented and a gap cannot be hidden. A draft with an unanswered subtask is "not
ready" without a Jev call.

**The cutoff is a guess that was checked, not tuned.** 1.0 sat in a natural gap in the first
runs (rejected at 0.88 or below, kept at 1.04 or above). Raising it to 1.5 would have
separated those runs more cleanly but also dropped a relevant article (1.12), and with 12
sources from two topics that would be fitting the cutoff to the sample. It stayed at 1.0.

## Checkpointing and human review

**SQLite, not Redis or a server database.** An embedded file survives restarts, which is all
the project needs, and swapping the saver is one line in `compile()`, so a fancier backend
would show little. `build_graph(checkpointer)` takes the saver as an argument, so tests use an
in-memory one.

**`durability="sync"`.** LangGraph's default is async: checkpoints are written in the
background, so a hard kill can lose the latest one. The entry point saves each checkpoint
before the next step starts and pays a little speed. (That async can lose one is documented,
not demonstrated here.)

**Strict msgpack mode.** Loading arbitrary classes from a database is an attack surface, so
the project runs with `LANGGRAPH_STRICT_MSGPACK=true`, set before `langgraph` is imported. In
strict mode LangGraph builds an allowlist from the state schema automatically. The tests run
in strict mode too, and a guard test fails if anything imports `langgraph` before the flag is
set.

**`interrupt()` and checkpoints are one mechanism.** A run waiting for review is a checkpoint
whose next node is `human_review` with a pending interrupt. Stopping the process loses
nothing; a new process opens the same database and resumes, and no research or writing is
repeated (tested with a real SQLite file, and demonstrated through the Docker stack).
`human_review` does nothing before `interrupt()` except read state, because the node restarts
from its top on resume. The reviewer can approve, reject with feedback (the writer redrafts
with the feedback and the earlier draft, then the run waits again), or take over with their
own text. The gate is optional (`human_review_gate=True`), because `interrupt()` needs a
checkpointer; the CLI ends at the ship check.

**Crash recovery, measured.** A test starts the real entry point as a subprocess, kills it
with SIGKILL after all three researchers have started, and resumes it. The two finished
subtasks were searched once each and the killed one twice. The checkpoint after the kill was
not what I expected (twice): one researcher task left, not three, and the two finished
findings already present, because LangGraph keeps finished tasks as pending writes.

**Time travel (with fakes).** Replaying from a checkpoint re-executes everything after it
(it is not a cache read); a fork with an edited plan runs only the researchers, with no
planner call. A fork does not generate a plan, so whoever forks supplies it.

**API and UI.** FastAPI starts a run in a background thread and returns a `thread_id` at once
(a run takes a minute or more). Status comes from the checkpoint: `awaiting_review` (a
pending interrupt) versus `interrupted` (stopped part-way; `/resume` continues it). A
Streamlit page talks to the API only and shows each source with its score and whether it was
kept, Jev's ship verdict, the draft, earlier rejections and the decision form. Both ports are
bound to `127.0.0.1`: the API has no login and spends the owner's keys.

## Models, search and rate limits

- **Groq `gpt-oss-20b` for the planner, researchers and writer.** The planner started on
  Gemini and moved to Groq after Gemini's free tier answered 504 at the 30 s timeout on every
  attempt, including from inside Docker. Groq's strict JSON mode needs every field required
  and a `Subtask` has a generated id (a model should not invent ids), so an adapter asks for
  a list of descriptions and builds the `Plan` with ids made in code. Gemini remains a setting.
- **Hard timeouts and one retry on every client.** The first real run took 296.8 s: the
  planner had no cap, and Gemini kept answering 503/504 while the client retried and spent
  quota. The fan-in waits for every researcher, so one hung call stalls the whole run.
- **Tavily reports errors by returning them.** It returns `{"error": ...}` for any failure and
  a plain string when nothing matched. The first version assumed `{"results": [...]}`, so all
  three researchers failed with `KeyError: 'results'` and the real error was hidden. The
  wrapper now handles all three shapes, and two tests use the real `TavilySearch` class with
  only the HTTP layer stubbed, so they fail if the library changes how it reports errors.
- **Source text is the Tavily snippet plus a cleaned slice of the page.** Measured on real
  pages, the first 1,500 characters of the raw page are mostly navigation and sign-up links.
  The cleaner drops images, URLs, link markup and short lines. It is a heuristic: genuine
  short bullets are lost and sentence-like ads survive.
- **A cut-off model reply is an error, not a report.** A redraft once stopped mid-sentence
  and dropped a whole section. The writer now reads `finish_reason` and raises on `length`,
  and sets explicit `max_tokens` and `reasoning_effort`. The cause was a hypothesis (a
  reasoning model's hidden tokens count against the output limit), not confirmed.

## What broke

Failures that taught something, in the order they appeared.

| What happened | Cause | Fix |
|---|---|---|
| All researchers failed with `KeyError: 'results'` | Tavily returns errors instead of raising; the code assumed success | Handle the three return shapes; tests with the real class |
| A run took 296.8 s | No timeout on the planner client; Gemini 503/504 plus client retries | `timeout=30`, `max_retries=1` on every client |
| `error=""` rejected by the validator, inside an `except` block | `TimeoutError()` has an empty message | Error text falls back to the exception class name |
| `.env` value failed with a `latin-1` codec error | A trailing comment on a value; the notebook kernel had loaded the raw line | Comments on their own line; restart kernels after editing `.env` |
| A module-level `InMemorySaver` next to `build_graph` | It was shadowed by the parameter and would have been shared by every graph in the process | Removed |
| A test used `monkeypatch.undo()` and reached the real Jev API | It removed the offline guard in `conftest.py` along with the fake | The test gets its own failing client |
| A report's weakest bullets came from a 403-character Instagram post (score 1.03) | Jev scored a thin page as useful | Plain rule: pages under 300 characters are dropped before Jev |
| A redraft ended mid-sentence | Probably the output-token limit on a reasoning model | `finish_reason` guard, explicit limits |
| A guitar report included gym-exercise advice | The planner dropped the word "guitar" from a subtask; Jev judged the page against the subtask as worded and found it relevant (1.42); the writer then bridged the gap with its own advice | Planner must name the subject; Jev sees the original question; the writer must not fill a gap |
| The test suite failed on one machine | The developer's `.env` (`PLANNER_PROVIDER=fixed`) leaked into tests through `load_dotenv()` | `conftest.py` clears those variables for every test |
| A comparison script hung for 15 minutes | `gpt-oss` answered "1" as text instead of calling the structured-output tool; my retry loop slept 20 s five times per page | Ask for plain text and parse the digit; retry only on 429; test one call before looping |

The guitar case is the one I would point to. Jev was not wrong: it answered the question it
was asked. The bug was in what it was shown, and the repair was upstream (what the planner
writes, what Jev sees), not a different cutoff.

## What was measured

All from real runs unless marked otherwise. The setup, limits and raw numbers are in the
linked notes.

- **Jev latency:** 42 calls made through the Docker stack: median 0.50 s, 95th percentile
  0.60 s, slowest 0.7 s. One request at a time. [Note](evidence/2026-11-17-jev-latency.md).
- **Jev against an LLM judge, on 35 saved pages:** 7 were off-topic (the text never says
  "guitar"). Jev and the project's Groq model, given the same question and level
  descriptions, each dropped all 7. Jev wrongly dropped 1 of 28 on-topic pages (a borderline
  forum thread at 0.90), the LLM judge none. Per call: Jev median 0.57 s (95th percentile
  0.95), the LLM judge 3.0 s (7.9). Small sample, one topic, wrong-subject detection only, and
  the LLM is a reasoning model on a rate-limited free tier, so this is not a general speed
  claim. [Note](evidence/2026-11-17-jev-vs-llm-judge.md).
- **A bad source, filtered and retried:** two quora.com pages about the wrong topic scored
  0.01 (confidence 0.98); Jev chose `retry_different` (0.97 against 0.00 for `escalate`); the
  retry excluded quora.com and passed. A second run reproduced it within 0.01 to 0.07; a
  later day returned different pages for the same plan. [Note](evidence/2026-11-16-retry-loop-real-run.md).
- **Off-topic pages, caught and escalated:** with a deliberately vague subtask, fitness pages
  scored 0.01 to 0.14, Jev chose `retry_different` (1.00), the retry limit sent the subtask to
  a person, the report listed it as not answered, and the ship check said not ready. The same
  kind of page had scored 1.42 before the fix. [Note](evidence/2026-11-17-vague-subtask-escalation.md).
- **A pause that survives a restart:** a run waiting for review was approved after the API
  container was restarted, with no search, Jev or writer call afterwards.
  [Note](evidence/2026-11-17-pause-survives-restart.md).
- **The default (Groq) planner, one run:** a model-written 3-subtask plan, every subtask
  naming the main subject; all found; 10 Jev calls, none fell back.
- **Tracing, one run:** all 38 runs of the graph appeared in LangSmith, the 10 Jev calls as
  spans with the page text replaced by its length (the Groq and Tavily spans do contain page
  text).

## What was not verified, and known limits

- **No claim is checked against its source.** The researcher answers from the pages and the
  writer sees only the answers, so the ship check judges completeness and coherence, not
  whether a statement is supported. In one run a table said an in-person teacher costs
  "similar or higher" than online lessons, and I could not tie that line to a cited page. A
  check that shows Jev a claim together with its sources is the most valuable thing not built.
- **A vague subtask that still names a plausible but wrong subject** would not be caught by
  giving Jev the original question. The stress test used a subtask with the subject missing.
- **The planner's prompt rule is a request, not a guarantee.** A code check (every subtask
  shares a content word with the question) was considered and not built.
- **The 300-character rule has not met a real short page.** None in the runs was that short,
  and a result that is only a short snippet is now dropped.
- **The source-quality cutoff and the level wording came from a handful of runs.** The
  comparison above is small and one-sided. Plan quality from `gpt-oss-20b` against Gemini was
  never compared.
- **Retries are capped at one, and cannot fix a bad query.** There is no `RetryPolicy`: the
  researcher already turns every exception into a `failed` finding, so a node-level policy
  would never fire, and a retry would repeat the search after an LLM failure.
- **Sequential Jev calls.** One request per source and per question; batching is an
  unfinished optimisation.
- **One reviewer, one process, no login.** Two people reviewing the same run is not handled,
  the API keeps which runs it is working on in memory, and a rejection overwrites `draft` (the
  earlier one survives in the prompt and the checkpoint history).
- **Docker on a clean machine** has not been tried; the stack was built and run on one machine.
- **Jev access.** The author's Jev credits end on 2026-11-19. After that, every Jev call fails
  and the run continues on the defaults in the table above (keep every source, assume every
  answer addresses its subtask, escalate every weak subtask, mark every draft not ready), so
  it still runs end to end but the Jev decisions are not exercised.

## Testing

286 tests, none of which touches the network: the planner, researcher and writer models,
Tavily and Jev are all faked at the module names the nodes use. A guard in `tests/conftest.py`
replaces the Jev client and the writer model in every test with one that fails like an outage
or returns a canned value, forces tracing off, and clears the environment variables that
change behaviour, so a forgotten fake cannot spend quota or send data out. A real call is
made only for questions no fake can answer (does Groq accept this schema, do the answers read
well). The crash-recovery test starts the real entry point in a separate process and kills it.
It waits a fixed 2 s after the researchers start, so it depends on timing (stable over
repeated runs, about 8 s); if it turns flaky, replace the wait with a poll on the database.
CI runs lint and the whole suite on every push, with no API keys.
