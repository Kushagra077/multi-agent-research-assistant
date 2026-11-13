from langgraph.checkpoint.serde import _msgpack


def test_the_test_run_is_in_strict_msgpack_mode():
    # Fails if something imports langgraph before tests/conftest.py sets the flag.
    assert _msgpack.STRICT_MSGPACK_ENABLED is True
