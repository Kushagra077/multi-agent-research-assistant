# Real run: source gate rejects off-topic pages, Jev picks `retry_different`, retry succeeds

Date: 2026-11-16. Model: `jev-1.13.0`. Question: "What is the best way for an adult beginner
to learn guitar?" Planner faked with a fixed 3-subtask plan (Gemini was returning 504s);
Tavily, Groq and Jev were real. Run from the notebook, 12 Jev calls, 0.4 to 1.1 s each.

## Result per subtask

| Subtask | Attempt 1 | Decision | Attempt 2 |
|---|---|---|---|
| 1. Practice routine | kept 1.82 (dietzemusic.com), kept 1.52 (youtube.com) | none | |
| 2. Lessons vs teacher vs books | kept 1.12 (guitarworld.com), kept 1.22 (theguitarlesson.com) | none | |
| 3. Common mistakes | dropped 0.01 and 0.01 (two quora.com threads about Python and drawing), answer does not address the subtask | `retry_different`, confidence 0.94 (probabilities: retry_same 0.04, retry_different 0.96, escalate 0.00), skip `quora.com` | kept 1.04 (instagram.com post, 403 chars of text), kept 1.06 (guitarprinciples.com), answer addresses the subtask |

## What it shows

- The gate rejects clearly off-topic pages with high confidence (0.01, confidence 0.98).
- `ask_choice` picks a sensible action from the evidence (0.96 for `retry_different`).
- Only the retried subtask is reviewed again (2 scores and 1 yes/no check in round 2).
- Weak spot: the retry passed with scores of 1.04 and 1.06, and one of its two sources is a
  403-character Instagram post. A minimum-text-length filter is an open idea.

## Jev log lines (round 2 and the decision)

```
jev choice question='A research subtask did not produce a usable answer. What should be done next?' choice=retry_different confidence=0.940 probabilities={'retry_same': 0.04, 'retry_different': 0.96, 'escalate': 0.0} state_chars=608 model=jev-1.13.0 elapsed=0.5s
jev score ... score=1.04 levels=3 confidence=0.880 state_chars=403 model=jev-1.13.0 elapsed=0.5s
jev score ... score=1.06 levels=3 confidence=0.900 state_chars=3150 model=jev-1.13.0 elapsed=0.5s
jev boolean ... answer=True p=0.970 threshold=0.50 confidence=0.970 state_chars=747 model=jev-1.13.0 elapsed=0.4s
```
