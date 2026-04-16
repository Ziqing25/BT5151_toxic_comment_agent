# tests/test_state.py
from pathlib import Path


def test_build_state_exists():
    from pipeline.state import BuildState
    assert BuildState is not None


def test_runtime_state_exists():
    from pipeline.state import RuntimeState
    assert RuntimeState is not None


def test_agent_state_removed():
    import pipeline.state as m
    assert not hasattr(m, "AgentState"), "AgentState must be removed — replaced by RuntimeState"


def test_build_initial_build_state_defaults():
    from pipeline.state import build_initial_build_state
    state = build_initial_build_state("/tmp/project")
    assert state["project_root"] == str(Path("/tmp/project").resolve())
    assert state["raw_train_path"].endswith("raw_data/train.csv")
    assert state["raw_test_path"].endswith("raw_data/test.csv")


def test_build_initial_build_state_custom_paths():
    from pipeline.state import build_initial_build_state
    state = build_initial_build_state(
        "/tmp/project",
        raw_train_path="/tmp/train.csv",
        raw_test_path="/tmp/test.csv",
    )
    assert state["raw_train_path"] == str(Path("/tmp/train.csv").resolve())
    assert state["raw_test_path"] == str(Path("/tmp/test.csv").resolve())


def test_build_initial_runtime_state_basic():
    from pipeline.state import build_initial_runtime_state
    state = build_initial_runtime_state("hello world")
    assert state["comment_text"] == "hello world"
    assert "project_root" not in state


def test_build_initial_runtime_state_with_root():
    from pipeline.state import build_initial_runtime_state
    state = build_initial_runtime_state("test", project_root="/tmp/project")
    assert state["project_root"] == str(Path("/tmp/project").resolve())


def test_old_build_initial_state_removed():
    import pipeline.state as m
    assert not hasattr(m, "build_initial_state"), (
        "build_initial_state must be removed — split into "
        "build_initial_build_state and build_initial_runtime_state"
    )
