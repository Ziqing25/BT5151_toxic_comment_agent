---
name: recommend-moderation-action
description: >
  Converts the severity assessment into a concrete moderation recommendation,
  including platform action, review priority, user notification policy, and a
  business-facing explanation suitable for display in the Gradio interface.
applyTo: "recommend-moderation-action"
---

## When to Use

Invoke this skill after `assess-severity` has assigned a severity label and review priority. This skill should be the final downstream business-facing stage before results are shown in the Gradio interface.

---

## How to Execute

1. Load `assess_severity_output.json` from the upstream `assess-severity` stage.
2. Read the severity decision fields: `severity_label`, `review_priority`, `escalation_required`, `toxicity_probability`, and `confidence`.
3. Apply rule-based moderation policy to map the severity level to an operational platform action.
4. Decide whether human review is required and what user notification level is appropriate.
5. Produce a short business-facing summary and UI explanation for a non-technical moderator or platform operator.
6. Write `moderation_action_output.json` to agent state for direct use in the Gradio interface.

---

## Inputs from Agent State

| Key | Source stage | Description |
|---|---|---|
| `assess_severity_output.json` | `assess-severity` | Severity label, review priority, escalation flag, confidence, and supporting traceability fields |

---

## Outputs to Agent State

| Key | Type | Description |
|---|---|---|
| `comment_text` | `string` | Original comment passed through from upstream |
| `selected_model_id` | `string` | Internal model identifier |
| `selected_model_label` | `string` | Human-readable model name |
| `predicted_label` | `string` | Upstream binary prediction |
| `predicted_class_id` | `int` | Binary class ID |
| `is_toxic` | `bool` | Boolean toxic flag |
| `toxicity_probability` | `float` | Toxic-class probability from the model |
| `non_toxic_probability` | `float` | Complementary non-toxic probability |
| `confidence` | `float` | Distance from the decision boundary |
| `threshold_used` | `float` | Threshold applied upstream |
| `severity_label` | `string` | Upstream severity band |
| `severity_rank` | `int` | Numeric severity level |
| `review_priority` | `string` | Moderator queue priority |
| `escalation_required` | `bool` | Whether stronger action is required |
| `action_code` | `string` | Internal action identifier |
| `action_label` | `string` | Human-readable recommended action |
| `action_priority` | `string` | Priority attached to the moderation action |
| `human_review_required` | `bool` | Whether a human moderator must review the case |
| `user_notification` | `string` | User-facing notification type (`none`, `gentle_warning`, `warning`, `final_warning`) |
| `moderator_rationale` | `string` | Explanation of why this action is recommended |
| `business_message` | `string` | Short business-facing summary for the platform team |
| `final_recommendation` | `string` | One-line recommendation for the UI |
| `ui_explanation` | `string` | Plain-language explanation suitable for a non-technical user |
| `summary_for_moderator` | `string` | Upstream summary from the severity stage |
| `severity_explanation` | `string` | Upstream explanation of the severity decision |
| `selection_justification` | `string` | Upstream model selection rationale |
| `bias_assessment` | `string` | Upstream fairness note |
| `inference_timestamp_utc` | `string` | Original inference timestamp |
| `severity_assessed_at_utc` | `string` | Severity assessment timestamp |
| `action_recommended_at_utc` | `string` | Action recommendation timestamp |

---

## Output Format

The primary output is `moderation_action_output.json`. Example structure:

```json
{
    "comment_text": "You are an absolute idiot and nobody wants you here.",
    "selected_model_id": "minilm_ft",
    "selected_model_label": "Fine-tuned MiniLM",
    "predicted_label": "toxic",
    "predicted_class_id": 1,
    "is_toxic": true,
    "toxicity_probability": 0.9412,
    "non_toxic_probability": 0.0588,
    "confidence": 0.8824,
    "threshold_used": 0.5,
    "severity_label": "critical",
    "severity_rank": 4,
    "review_priority": "urgent",
    "escalation_required": true,
    "action_code": "remove_and_escalate",
    "action_label": "Remove and Escalate",
    "action_priority": "urgent",
    "human_review_required": true,
    "user_notification": "final_warning",
    "moderator_rationale": "The comment shows a very strong toxic signal and should be removed immediately with escalation to urgent moderator attention.",
    "business_message": "Remove the comment immediately and escalate the case to the urgent moderation queue.",
    "final_recommendation": "Recommended action: Remove and Escalate. Severity is critical with toxicity probability 0.9412 and confidence 0.8824.",
    "ui_explanation": "This comment was assessed as critical. Remove the comment immediately and escalate the case to the urgent moderation queue.",
    "summary_for_moderator": "Predicted toxic with toxicity probability 0.9412 and confidence 0.8824. Assigned severity: critical.",
    "severity_explanation": "The comment is classified as toxic with both very high probability and high confidence, suggesting a severe moderation risk that should be escalated immediately.",
    "selection_justification": "Fine-tuned MiniLM achieves the highest AUC-ROC ...",
    "bias_assessment": "False positives are mainly caused by dataset label errors ...",
    "inference_timestamp_utc": "2026-04-12T08:00:00+00:00",
    "severity_assessed_at_utc": "2026-04-12T08:01:00+00:00",
    "action_recommended_at_utc": "2026-04-12T08:02:00+00:00"
}
```

---

## Notes

- This stage is intentionally rule-based so the policy logic is easy to explain and audit in the technical report.
- The Gradio interface should display the action recommendation and business explanation, not just the raw model output.
- `allow_with_monitoring` is designed for borderline cases where the system should remain cautious without over-moderating benign content.
- `remove_and_escalate` should be reserved for the most severe cases, especially when severity is `critical`.
- This stage should not reload any model or recompute severity; it only interprets the upstream policy state.
