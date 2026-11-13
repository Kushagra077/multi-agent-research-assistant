"""Kill-and-resume: start the real entry point as a separate process, kill it mid-run,
start it again with --resume, and check that the finished work was not repeated.

Everything external is faked in crash_runner.py, so no network and no API quota.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from research_assistant.graph.build import build_graph

RUNNER = Path(__file__).parent / "crash_runner.py"
THREAD_ID = "crash-test"
FAST = ["fast question 1", "fast question 2"]
SLOW = "slow question"


def _start(tmp_path: Path, *args: str) -> subprocess.Popen:
    env = {**os.environ, "CRASH_TEST_CALLS_FILE": str(tmp_path / "calls.txt")}
    return subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            *args,
            "--thread-id",
            THREAD_ID,
            "--db",
            str(tmp_path / "checkpoints.db"),
        ],
        cwd=tmp_path,  # setup_logging() writes logs/ here, not into the project
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def _wait_until(condition, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.1)
    pytest.fail("timed out waiting for the child process")


def _saved_state(tmp_path: Path):
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
        return build_graph(saver).get_state({"configurable": {"thread_id": THREAD_ID}})


def test_killed_run_resumes_and_only_the_unfinished_researcher_runs_again(tmp_path):
    # --- run 1: start, wait until all three researchers are in flight, then kill -9 ---
    first = _start(tmp_path, "What is X?")
    _wait_until(lambda: {*FAST, SLOW} <= set(_calls(tmp_path)))
    time.sleep(2)  # let the two fast researchers finish and their results be saved
    first.kill()  # SIGKILL: no cleanup, no chance to save anything
    _, first_err = first.communicate()
    assert first.returncode != 0

    # --- what is on disk after the crash ---
    crashed = _saved_state(tmp_path)
    # Three researchers started, but `next` lists only ONE still to run: the two that
    # finished were saved individually as pending writes, so they count as done.
    assert crashed.next == ("researcher",)
    assert "plan" in crashed.values
    # Their results are already readable from disk: the two fast subtasks have findings,
    # and the slow one (the one that was killed) does not.
    by_description = {s.id: s.description for s in crashed.values["plan"].subtasks}
    saved = {by_description[f.subtask_id] for f in crashed.values["findings"]}
    assert saved == set(FAST)

    # --- run 2: resume with the same thread id ---
    second = _start(tmp_path, "--resume")
    out, err = second.communicate(timeout=60)
    assert second.returncode == 0, err

    # all three subtasks ended up with a finding
    assert out.count("[found]") == 3
    assert "[failed]" not in out

    # the finished researchers were NOT searched again; only the killed one was
    calls = _calls(tmp_path)
    assert calls.count(FAST[0]) == 1
    assert calls.count(FAST[1]) == 1
    assert calls.count(SLOW) == 2

    # the saved run is now complete
    final = _saved_state(tmp_path)
    assert final.next == ()
    assert len(final.values["findings"]) == 3

    # strict msgpack was on for both processes, so nothing was unregistered or blocked
    for stderr in (first_err, err):
        assert "unregistered type" not in stderr
        assert "Blocked deserialization" not in stderr


def test_resuming_a_thread_that_never_ran_exits_with_an_error(tmp_path):
    proc = _start(tmp_path, "--resume")
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 1
    assert "No saved state" in err


def test_starting_a_new_question_on_an_existing_thread_is_refused(tmp_path):
    first = _start(tmp_path, "What is X?")
    # let it get far enough to save something, then stop it
    _wait_until(lambda: SLOW in _calls(tmp_path))
    time.sleep(2)
    first.kill()
    first.communicate()

    again = _start(tmp_path, "A different question")
    _, err = again.communicate(timeout=60)
    assert again.returncode == 1
    assert "saved state exists" in err
