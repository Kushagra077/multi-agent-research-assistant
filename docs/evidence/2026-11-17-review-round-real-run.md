# Real run through the review page: reject and redraft, and a subtask that lost its subject

Date: 2026-11-17. Question: "What is the best way for an adult beginner to learn guitar?"
Planner on Groq, researchers and writer on Groq, Tavily and Jev real, run from the Streamlit
review page against the Docker stack.

## The review gate on real data

- First draft 4,043 characters, Jev ship check: ready, 0.76. The reviewer rejected it with a
  note. `human_review` resumed, the writer ran again, and the run waited again (`revisions=1`).
- The redraft was 2,551 characters and stopped mid-sentence, with a whole section missing.
  Jev's ship check rated it **not ready, 0.80**. The cause was not confirmed (the log did not
  record why the model stopped); the writer now logs `finish_reason` and `completion_tokens`
  and refuses a reply cut off at the output limit.
- With the explicit limits (`max_tokens=4096`, `reasoning_effort=low`), later drafts were
  4,035 and 4,816 characters, `finish_reason=stop`, 797 and 958 completion tokens.

## A subtask that lost its subject

The Groq planner wrote a third subtask without the word "guitar" ("Outline an effective daily
practice routine for adult beginners that maximizes skill acquisition, including recommended
duration, warm-up exercises, and progression milestones"). Tavily returned a Healthline "how
to start exercising" page (scored 1.42) and a YouTube video (1.06); the answer was about leg
kicks and push-ups, and the answer check passed it. Jev's judgements were consistent with the
subtask as worded; the planner broke its own rule to name the key entities.

The writer then added its own guitar advice to bridge the gap, against the "use only the
findings" rule. The ship check rated the draft **not ready, 0.67**.

Changes made after this run: the planner prompt requires every subtask to name the main
subject, Jev is shown the original question with every page and answer, and the writer is told
not to fill a gap from its own knowledge. These changes alter the wording Jev is asked, so
scores from earlier runs are not exactly comparable; not yet verified on a real run.
