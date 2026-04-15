---
name: assess-severity
description: >
  Translates the raw output of the run-inference stage into a business-facing
  moderation severity level, review priority, and escalation signal so that
  downstream policy skills can recommend an appropriate action.
applyTo: "assess-severity"
---

## When to Use

Invoke this skill immediately after `run-inference` has produced a binary prediction and class probability for a new comment. This skill should run before `recommend-moderation-action` so the next stage can use a stable severity label rather than raw model scores.

---

## How to Execute

1. Load `run_inference_output.json` from the upstream `run-inference` stage.
2. Read the model decision fields: `predicted_label`, `is_toxic`, `toxicity_probability`, `confidence`, and `threshold_used`.
3. Apply rule-based severity mapping to convert the model output into a business-facing severity band.
4. Assign a review priority and an escalation flag based on the severity band.
5. Produce a short explanation for moderators describing why the severity was assigned.
6. Write `assess_severity_output.json` to agent state for downstream policy or UI stages.

---

## Inputs from Agent State

| Key | Source stage | Description |
|---|---|---|
| `run_inference_output.json` | `run-inference` | Binary decision, class probabilities, confidence, model metadata, and traceability fields |

---

## Outputs to Agent State

| Key | Type | Description |
|---|---|---|
| `comment_text` | `string` | Original comment passed through from upstream |
| `selected_model_id` | `string` | Internal model identifier |
| `selected_model_label` | `string` | Human-readable model name |
| `predicted_label` | `string` | Upstream binary decision (`toxic` / `non-toxic`) |
| `predicted_class_id` | `int` | Binary class ID |
| `is_toxic` | `bool` | Boolean toxic flag |
| `toxicity_probability` | `float` | Toxic-class probability from the model |
| `non_toxic_probability` | `float` | Complementary non-toxic probability |
| `confidence` | `float` | Distance from the decision boundary |
| `threshold_used` | `float` | Threshold applied upstream |
| `severity_label` | `string` | Business-facing severity band (`clean`, `borderline`, `low`, `medium`, `high`, `critical`) |
| `severity_rank` | `int` | Numeric encoding of severity for downstream rules |
| `review_priority` | `string` | Human-review priority (`none`, `low`, `medium`, `high`, `urgent`) |
| `escalation_required` | `bool` | Whether stronger moderation action is likely required |
| `severity_explanation` | `string` | Plain-language explanation of the severity decision |
| `summary_for_moderator` | `string` | One-line moderation summary suitable for the UI |
| `selection_justification` | `string` | Upstream selection rationale for traceability |
| `bias_assessment` | `string` | Upstream fairness note for transparency |
| `inference_timestamp_utc` | `string` | Original inference timestamp |
| `severity_assessed_at_utc` | `string` | Severity assessment timestamp |

---

## Output Format

The primary output is `assess_severity_output.json`. Example structure:

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
    "severity_explanation": "The comment is classified as toxic with both very high probability and high confidence, suggesting a severe moderation risk that should be escalated immediately.",
    "summary_for_moderator": "Predicted toxic with toxicity probability 0.9412 and confidence 0.8824. Assigned severity: critical.",
    "selection_justification": "Fine-tuned MiniLM achieves the highest AUC-ROC ...",
    "bias_assessment": "False positives are mainly caused by dataset label errors ...",
    "inference_timestamp_utc": "2026-04-12T08:00:00+00:00",
    "severity_assessed_at_utc": "2026-04-12T08:01:00+00:00"
}
```

---

## Notes

- This skill is deliberately rule-based so that severity policy remains interpretable and easy to explain in the technical report.
- `borderline` is reserved for comments that remain below the toxic threshold but are close enough to it that a downstream review rule may still want to flag them.
- `critical` should be used sparingly for toxic comments that are both highly probable and far from the decision boundary.
- This stage should not reload any model or recompute probabilities; it only interprets the upstream output.
