# Jev latency, measured from the logs of real runs

Date: 2026-11-17. Source: `/app/logs/logfile.log` in the Docker volume, which holds every
run made through the review page since the Groq planner went in (the planner, researchers,
writer, Tavily and Jev were all real). Method: take the `elapsed=` of each
`research_assistant.decisions.jev` log line. Each line times one request, made one after
another.

| | calls | mean | median | slowest |
|---|---|---|---|---|
| all | 42 | 0.50 s | 0.50 s | 0.7 s (p95 0.6 s) |
| source score | 24 | 0.50 s | | 0.6 s |
| yes/no check | 18 | 0.52 s | | 0.7 s |

The earlier notebook run (docs/evidence/2026-11-16-full-pipeline-real-run.md) had 13 calls,
mean about 0.7 s, range 0.5 to 1.8 s. The logs round to 0.1 s.

In these page runs the 24 scored sources ranged 1.03 to 1.96 and none fell below the 1.0
cutoff, so the gate dropped nothing there; its rejections (quora.com pages at 0.01) come from
the notebook runs. The 6 ship checks were 3 ready (0.69, 0.76, 0.77) and 3 not ready (0.80
for a cut-off redraft, 0.75, 0.67 for a draft padded with off-topic findings). All 3 writer
drafts that logged a finish reason ended with `stop`.

No LLM call was timed, so this says nothing about Jev versus an LLM.
