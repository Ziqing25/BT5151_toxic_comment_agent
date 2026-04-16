from unittest.mock import MagicMock, patch

import pytest


def test_get_openai_key_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    from pipeline.graph import _get_openai_key
    assert _get_openai_key() == "sk-test-key"


def test_get_openai_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from pipeline.graph import _get_openai_key
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _get_openai_key()


def test_fallback_warning_covers_all_action_codes():
    from pipeline.graph import _fallback_warning
    action_codes = ["soft_warn", "review_and_warn", "hide_and_review", "remove_and_escalate"]
    for code in action_codes:
        state = {"action_code": code, "severity_label": "medium", "action_label": "Test"}
        msg = _fallback_warning(state)
        assert isinstance(msg, str) and len(msg) > 10, f"Empty fallback for {code}"


def test_build_warning_prompt_excludes_comment_text():
    from pipeline.graph import _build_warning_prompt
    state = {
        "action_code": "soft_warn",
        "severity_label": "low",
        "action_label": "Soft Warning",
        "comment_text": "this is a secret sensitive comment",
    }
    prompt = _build_warning_prompt(state)
    assert "secret sensitive comment" not in prompt, "Raw comment text must not appear in prompt"
    assert "soft_warn" in prompt or "Soft Warning" in prompt or "low" in prompt


def test_draft_warning_node_skips_for_allow():
    from pipeline.graph import draft_warning_node
    state = {"action_code": "allow", "severity_label": "clean"}
    result = draft_warning_node(state)
    assert result["warning_skipped"] is True
    assert result["warning_message"] == ""


def test_draft_warning_node_skips_for_allow_with_monitoring():
    from pipeline.graph import draft_warning_node
    state = {"action_code": "allow_with_monitoring", "severity_label": "borderline"}
    result = draft_warning_node(state)
    assert result["warning_skipped"] is True


def test_draft_warning_node_uses_fallback_on_openai_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    from pipeline.graph import draft_warning_node
    with patch("pipeline.graph.OpenAI", side_effect=Exception("API down")):
        state = {
            "action_code": "soft_warn",
            "severity_label": "low",
            "action_label": "Soft Warning",
        }
        result = draft_warning_node(state)
    assert result["warning_skipped"] is False
    assert isinstance(result["warning_message"], str)
    assert len(result["warning_message"]) > 0


def test_draft_warning_node_calls_openai_for_toxic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Please be respectful in future comments."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("pipeline.graph.OpenAI", return_value=mock_client):
        from pipeline.graph import draft_warning_node
        state = {
            "action_code": "review_and_warn",
            "severity_label": "medium",
            "action_label": "Review and Warn",
        }
        result = draft_warning_node(state)

    assert result["warning_message"] == "Please be respectful in future comments."
    assert result["warning_skipped"] is False
    assert "warning_generated_at_utc" in result
