# pipeline/state.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

# ── Shared literal types ───────────────────────────────────────────────────────

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


# ── Build layer state (offline ML pipeline) ───────────────────────────────────

class BuildState(TypedDict, total=False):
    # Inputs
    project_root: str
    raw_train_path: str
    raw_test_path: str

    # preprocess-data outputs
    train_processed_path: str      # experiments/processed_data/train_set.csv
    val_processed_path: str        # experiments/processed_data/val_set.csv
    test_processed_path: str       # experiments/processed_data/test_set.csv
    preprocessing_summary: dict[str, Any]

    # train-models outputs
    train_metadata_path: str       # models/selected_model_metadata.json
    candidate_model_ids: list[str]

    # evaluate-models outputs
    evaluation_report_path: str    # models/evaluation_report.json
    bias_audit_path: str           # models/bias_audit_summary.json

    # select-model outputs
    select_model_output_path: str  # select_model_output.json (project root)
    selected_model_id: str
    selection_justification: str


# ── Runtime layer state (online moderation pipeline) ─────────────────────────

class RuntimeState(TypedDict, total=False):
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

    # Draft warning output (new)
    warning_message: str
    warning_skipped: bool
    warning_generated_at_utc: str


# ── Constructor helpers ────────────────────────────────────────────────────────

def build_initial_build_state(
    project_root: str | Path,
    raw_train_path: str | Path | None = None,
    raw_test_path: str | Path | None = None,
) -> BuildState:
    root = Path(project_root).resolve()
    state: BuildState = {"project_root": str(root)}
    state["raw_train_path"] = (
        str(Path(raw_train_path).resolve())
        if raw_train_path is not None
        else str(root / "raw_data" / "train.csv")
    )
    state["raw_test_path"] = (
        str(Path(raw_test_path).resolve())
        if raw_test_path is not None
        else str(root / "raw_data" / "test.csv")
    )
    return state


def build_initial_runtime_state(
    comment_text: str, project_root: str | Path | None = None
) -> RuntimeState:
    state: RuntimeState = {"comment_text": comment_text}
    if project_root is not None:
        state["project_root"] = str(Path(project_root).resolve())
    return state
