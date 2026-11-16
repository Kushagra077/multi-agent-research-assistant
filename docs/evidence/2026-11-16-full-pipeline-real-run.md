# Real run: question to drafted report, with all four Jev decisions

Date: 2026-11-16. Model: `jev-1.13.0`. Question: "What is the best way for an adult beginner
to learn guitar?" Planner faked with a fixed 3-subtask plan (Gemini was returning 504s);
Tavily, Groq (researchers and writer) and Jev were real. Run from the notebook.

## Jev calls (13 in total, run one after another)

- Latency per call: 0.5 s to 1.8 s, mean about 0.7 s (the log rounds to 0.1 s).
- Source gate (score): 8 calls. Kept 1.82, 1.56, 1.09, 1.29, 1.03, 1.06; dropped 0.01, 0.01.
- Answer check (boolean): 4 calls, all yes (p = 0.95, 0.97, 0.97; the ship check 0.75).
- Retry-vs-escalate (choice): 1 call, `retry_different` 0.97 (retry_same 0.03, escalate 0.00).
- Ship check (boolean) on the 3,389-character draft: ready, p = 0.75.

## The retry

Subtask 3 ("most common mistakes") first returned two quora.com threads about Python and
drawing. Both scored 0.01 (confidence 0.98) and were dropped, the answer check said no, and
Jev chose `retry_different`. The retry skipped quora.com and passed with an instagram.com post
(1.03) and guitarprinciples.com (1.06). Only that subtask was reviewed again.

This is the same sequence as the earlier run (docs/evidence/2026-11-16-retry-loop-real-run.md):
Tavily returned the same pages and Jev scored them within 0.01 to 0.07 of the earlier scores,
so the demo is reproducible.

## Caveats found in the output (not yet fixed)

- Two of the five "common mistakes" in the report read oddly (a "long stretches without
  breaks" item and "starting finger training at the first fret"). Subtask 3 was written from
  the weakest sources of the run (the 403-character Instagram post and one blog). The claims
  were not checked against the pages.
- The ship check passed the draft at p = 0.75. It only sees the draft, so it judges
  completeness and coherence, not whether each claim is supported by a source.
- Citations are per section (sources are listed under each subtask), not per sentence.

## The report (verbatim)

```
**Summary**
An adult beginner should start with a short, focused daily practice routine, about 15 minutes, 4-6 days a week, targeting one specific skill per session. Learning can be done through online lessons, a private teacher, or books, each with distinct advantages and drawbacks, but consistent practice remains the most critical factor. Avoid common pitfalls such as excessive string pressure, neglecting the tuner, and improper pacing to ensure steady progress.

### Practice Routine and Daily Schedule
- Practice 15 minutes a day, 4-6 days per week.
- Each session focuses on a single goal (e.g., a chord transition, a strumming pattern, or a rhythm exercise).
- Allocate roughly 5 minutes to that skill.
- On low-energy days keep it simple: tune, practice one easy chord change, and play a familiar song.
- When more time or energy is available, add extra work such as additional chord shapes or a new song.
- This short, focused routine builds muscle memory, keeps progress measurable, and prevents burnout.

### Learning Method: Online Lessons, Teacher, or Books
- Online lessons (free or paid) are inexpensive, flexible, and allow repeated viewing, making them a good value for self-paced learning.
- Private teacher offers a structured roadmap, accountability, and personalized feedback, which can accelerate progress and maintain motivation.
- Books are generally the least effective option because they lack practical demonstration and real-time correction.
- Regardless of the chosen method, consistent practice is the single most important factor for advancement.

### Common Mistakes and How to Avoid Them
Adult beginners often make five key mistakes:
1. Squeezing the strings too hard - causes hand pain; avoid by using a relaxed grip.
2. Staring at the fretting hand instead of the fretboard - reduce tension by focusing on the fretboard.
3. Skipping the tuner - chords sound off; always tune before each session.
4. Long stretches of time without proper practice breaks - prevent fatigue by pacing practice with short, focused sessions.
5. Starting finger training at the first fret - builds unnecessary tension; begin finger exercises away from the first fret to reduce tension.

By following the recommended practice routine, choosing an appropriate learning method, and steering clear of these common mistakes, an adult beginner can establish a solid foundation and progress efficiently on the guitar.

## Sources
(3 subtasks, 2 kept sources each: dietzemusic.com, youtube.com, guitarworld.com,
theguitarlesson.com, instagram.com, guitarprinciples.com. quora.com was dropped.)
```
