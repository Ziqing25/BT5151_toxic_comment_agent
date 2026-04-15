from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict


SeverityLabel = Literal["clean", "borderline", "low", "medium", "high", "critical"]
ReviewPriority = Literal["none", "low", "medium", "high", "urgent"]
ActionCode = Literal[
    "allow",
    "allow_with_monitoring",
    "soft_warn",
    "review_and_warn",
    "hide_and_review",
    "remove_and_escalate",
]
UserNotification = Literal["none", "gentle_warning", "warning", "final_warning"]


class AgentState(TypedDict, total=False):
    # Runtime input
    comment_text: str
    project_root: str
    select_model_output_path: str
    train_metadata_path: str
    run_inference_output_path: str
    assess_severity_output_path: str
    moderation_action_output_path: str

    # Model selection context
    selected_model_id: str
    selected_model_label: str
    model_type: str
    threshold_used: float
    selection_justification: str
    bias_assessment: str
    source_artifact: dict[str, Any]

    # Inference output
    predicted_label: Literal["toxic", "non-toxic"]
    predicted_class_id: int
    is_toxic: bool
    toxicity_probability: float
    non_toxic_probability: float
    confidence: float
    raw_score: float
    inference_timestamp_utc: str

    # Severity output
    severity_label: SeverityLabel
    severity_rank: int
    review_priority: ReviewPriority
    escalation_required: bool
    severity_explanation: str
    summary_for_moderator: str
    severity_assessed_at_utc: str

    # Moderation action output
    action_code: ActionCode
    action_label: str
    action_priority: ReviewPriority
    human_review_required: bool
    user_notification: UserNotification
    moderator_rationale: str
    business_message: str
    final_recommendation: str
    ui_explanation: str
    action_recommended_at_utc: str


def build_initial_state(comment_text: str, project_root: str | Path | None = None) -> AgentState:
    state: AgentState = {"comment_text": comment_text}
    if project_root is not None:
        state["project_root"] = str(Path(project_root).resolve())
    return state
