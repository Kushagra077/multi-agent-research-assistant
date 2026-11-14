import argparse
import os
import uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent research assistant")
    parser.add_argument(
        "question", nargs="?", help="Research question (not needed with --resume)"
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Name of the run. A new one is generated and printed if omitted.",
    )
    parser.add_argument(
        "--db", default="checkpoints.db", help="SQLite file that stores checkpoints"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an unfinished run instead of starting a new one",
    )
    return parser.parse_args()


def print_summary(result: dict) -> None:
    """Print the plan, then one line per subtask with its finding's status, plus what
    the Jev checks in merge decided about it (if a review exists)."""
    subtasks = result["plan"].subtasks
    # Findings arrive in completion order (the researchers run in parallel), so they
    # are matched to subtasks by id, not by position.
    findings_by_id = {f.subtask_id: f for f in result["findings"]}
    # A run saved before the Jev checks existed has no "reviews" key.
    reviews_by_id = {r.subtask_id: r for r in result.get("reviews", [])}

    print("\nPlan:")
    for number, subtask in enumerate(subtasks, start=1):
        print(f"  {number}. {subtask.description}")

    print("\nFindings:")
    for subtask in subtasks:
        finding = findings_by_id.get(subtask.id)
        if finding is None:
            print(f"  [missing] {subtask.description}")
            continue
        print(f"  [{finding.status}] {subtask.description}")
        if finding.status == "failed":
            print(f"      error: {finding.error}")
        review = reviews_by_id.get(subtask.id)
        if review is not None:
            answer = "yes" if review.addresses_subtask else "NO"
            print(
                f"      sources: {len(review.kept_urls)} kept, "
                f"{len(review.dropped_urls)} dropped; answers the subtask: {answer}"
            )
            for url in review.dropped_urls:
                print(f"      dropped: {url}")
            if review.used_fallback:
                print("      note: Jev was unreachable, defaults were used")


def main() -> None:
    # langgraph reads LANGGRAPH_STRICT_MSGPACK when it is first imported, so the flag
    # is set first and the imports that pull in langgraph sit inside main(), after it.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    from langgraph.checkpoint.sqlite import SqliteSaver

    from research_assistant.graph.build import build_graph
    from research_assistant.logging_config import setup_logging

    args = parse_args()
    setup_logging()

    # The thread id is the only key to a saved run, so it is always printed.
    thread_id = args.thread_id or str(uuid.uuid4())
    print(f"Thread ID: {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(args.db) as saver:
        graph = build_graph(saver)

        saved = graph.get_state(config)
        has_saved_state = bool(saved.values)  # an unknown thread has empty values

        # Guards. SystemExit(message) prints the message to stderr and exits with
        # code 1, so scripts and tests can tell that the run did not happen.
        if not args.resume and has_saved_state:
            raise SystemExit(
                "Error: A saved state exists for this thread. Use --resume to continue."
            )
        if args.resume and not has_saved_state:
            raise SystemExit("Error: No saved state exists for this thread.")
        if not args.resume and not args.question:
            raise SystemExit("Error: Give a question, or use --resume.")

        # durability="sync" saves each checkpoint before the next step starts. The
        # default ("async") saves in the background, so a hard kill could lose the
        # latest checkpoint, which defeats the purpose of checkpointing.
        # None as the input means "continue from the last checkpoint".
        if args.resume:
            result = graph.invoke(None, config, durability="sync")
        else:
            result = graph.invoke(
                {"question": args.question}, config, durability="sync"
            )

        print_summary(result)


if __name__ == "__main__":
    main()
