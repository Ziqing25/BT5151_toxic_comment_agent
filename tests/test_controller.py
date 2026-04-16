from unittest.mock import MagicMock, patch

import pytest


def test_run_build_mode_calls_build_graph(tmp_path):
    from pipeline.state import build_initial_build_state

    mock_result = {"selected_model_id": "minilm_ft"}
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = mock_result

    with patch("pipeline.controller.compile_build_graph", return_value=mock_graph):
        from pipeline.controller import run
        state = build_initial_build_state(str(tmp_path))
        result = run("build", state)

    mock_graph.invoke.assert_called_once_with(state)
    assert result == mock_result


def test_run_moderate_mode_calls_runtime_graph():
    from pipeline.state import build_initial_runtime_state

    mock_result = {"warning_message": "Please follow guidelines."}
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = mock_result

    with patch("pipeline.controller.compile_runtime_graph", return_value=mock_graph):
        from pipeline.controller import run
        state = build_initial_runtime_state("hello world")
        result = run("moderate", state)

    mock_graph.invoke.assert_called_once_with(state)
    assert result == mock_result


def test_run_invalid_mode_raises():
    from pipeline.state import build_initial_runtime_state
    from pipeline.controller import run

    with pytest.raises(ValueError, match="mode"):
        run("invalid_mode", build_initial_runtime_state("test"))
