# Jev vs an LLM judge on saved pages (2026-11-17)

A small comparison, to give the README's "Jev instead of an LLM call" claim some evidence.
It is small and it tests one thing, so read the limits below before quoting a number.

## Setup

- **Pages:** the 44 sources saved in the Docker volume from 7 real runs of the guitar
  question; 35 unique (url, subtask) pairs. No new searches were made.
- **Judges, same input for both:** the text `merge.py` builds (original question, subtask,
  title, URL, page text), the same question and the same three level descriptions.
  - **Jev:** `ask_score` on the current code (`jev-1.13.0`), kept if the score is at least 1.0.
  - **LLM judge:** the project's researcher model (`openai/gpt-oss-20b` on Groq), asked to
    reply with a level 0, 1 or 2, kept if the level is at least 1.
- **Label (mechanical, no hand labelling):** a page whose text never contains the word
  "guitar" is off-topic for a guitar question. 7 of the 35 pages are (fitness warm-up pages,
  a general exercise guide, a strength-training video, a German-learning page).
- Script: not committed (it lived in a scratch folder); it is about 90 lines.

## Result

| | Jev | Groq LLM judge |
|---|---|---|
| Off-topic pages dropped | 7 of 7 | 7 of 7 |
| On-topic pages wrongly dropped | 1 of 28 | 0 of 28 |
| Time per call, median | 0.57 s | 3.03 s |
| Time per call, 95th percentile | 0.95 s | 7.9 s |
| Slowest call | 1.2 s | 8.7 s |
| Calls that fell back or failed | 0 | 0 (3 rate-limit retries, not timed) |

- The one on-topic page Jev dropped was a forum thread asking whether online lessons are
  worth it (Jev 0.90, just under the 1.0 cutoff). The LLM judge kept it. The level-0 text
  names "a forum question that gives no answer", so Jev's call is defensible.
- Two pages Jev had **kept** in earlier runs (an exercise guide at 1.42 and a
  strength-training video at 1.06, scored before the original question was added to what Jev
  sees) now score 0.00 with the current prompt. This is the fix from design entry 42,
  confirmed on the pages that exposed the problem.

## Limits

- **Small and one-sided:** 35 pages, 7 negatives, one question topic. It shows both judges
  catch a page about the wrong subject. It does not show how they compare on grading
  quality between "partly" and "directly" useful.
- **The label is crude:** "never says guitar" finds wrong-subject pages only.
- **Not like for like on speed:** the Groq model is a reasoning model on a rate-limited free
  tier, with calls made one at a time, and the 3 rate-limited calls are not in its timings.
  A different model or a paid tier could be faster. The comparison says what *this project's
  LLM setup* costs, not what an LLM judge must cost.
- **Jev adds a probability the LLM judge here did not give:** a score between levels and a
  confidence. That was not evaluated.
- Jev's calls were made once each; repeat runs earlier moved scores by 0.01 to 0.07.
