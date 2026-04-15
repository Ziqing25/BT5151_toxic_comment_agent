---
name: run-inference
description: >
  Loads the model selected by the upstream select-model stage, applies the
  correct inference path for the chosen model family, predicts whether a new
  comment is toxic, and writes a unified inference payload to agent state for
  downstream business-facing moderation skills.
applyTo: "run-inference"
---

## When to Use

Invoke this skill after `select-model` has written the final deployment decision to agent state and when a new user comment needs to be classified. This skill should be executed before downstream skills such as `assess-severity` and `recommend-moderation-action`.

---

## How to Execute

1. Load `select_model_output.json` from the upstream `select-model` stage.
2. Load `selected_model_metadata.json` from the `train-models` outputs to recover fallback artifact paths and model names.
3. Read the incoming user comment from agent state.
4. Inspect `selected_model_id` and `artifact.type` to determine which inference path to use.
5. Load the selected model and any required auxiliary artefacts:
   - TF-IDF vectorizer + sklearn classifier
   - frozen transformer encoder + logistic regression classifier
   - fine-tuned transformer classifier directory
6. Run inference on the new comment and compute the toxic-class probability.
7. Apply `inference_threshold` to convert the probability into a binary toxic / non-toxic decision.
8. Write `run_inference_output.json` to agent state so downstream skills can turn the prediction into a business-facing moderation output.

---

## Inputs from Agent State

| Key | Source stage | Description |
|---|---|---|
| `select_model_output.json` | `select-model` | Selected model ID, label, artifact metadata, threshold, and selection rationale |
| `selected_model_metadata.json` | `train-models` | Backup artifact paths, base model names, and shared threshold metadata |
| `comment_text` | user input / Gradio | Raw comment text to classify |

---

## Outputs to Agent State

| Key | Type | Description |
|---|---|---|
| `comment_text` | `string` | User comment after basic validation |
| `selected_model_id` | `string` | Internal ID of the model used for inference |
| `selected_model_label` | `string` | Human-readable name of the selected model |
| `model_type` | `string` | Inference loader path used (`tfidf_sklearn`, `bert_embedding_lr`, or `sentence_transformer_finetuned`) |
| `threshold_used` | `float` | Threshold used to convert probability into the binary decision |
| `predicted_label` | `string` | `toxic` or `non-toxic` |
| `predicted_class_id` | `int` | Binary class ID (`1` for toxic, `0` for non-toxic) |
| `is_toxic` | `bool` | Boolean version of the final prediction |
| `toxicity_probability` | `float` | Model probability assigned to the toxic class |
| `non_toxic_probability` | `float` | Complementary probability assigned to the non-toxic class |
| `confidence` | `float` | Distance from the decision boundary, scaled to `[0, 1]` |
| `raw_score` | `float` | Underlying score before downstream interpretation |
| `selection_justification` | `string` | Selection rationale propagated from upstream for traceability |
| `bias_assessment` | `string` | Bias note propagated from upstream for transparency |
| `source_artifact` | `object` | Exact model and loader configuration used for this prediction |
| `inference_timestamp_utc` | `string` | Audit timestamp for the inference run |

---

## Output Format

The primary output is `run_inference_output.json`. Example structure:

```json
{
    "comment_text": "You are an absolute idiot and nobody wants you here.",
    "selected_model_id": "minilm_ft",
    "selected_model_label": "Fine-tuned MiniLM",
    "model_type": "sentence_transformer_finetuned",
    "threshold_used": 0.5,
    "predicted_label": "toxic",
    "predicted_class_id": 1,
    "is_toxic": true,
    "toxicity_probability": 0.9412,
    "non_toxic_probability": 0.0588,
    "confidence": 0.8824,
    "raw_score": 2.7719,
    "selection_justification": "Fine-tuned MiniLM achieves the highest AUC-ROC ...",
    "bias_assessment": "False positives are mainly caused by dataset label errors ...",
    "source_artifact": {
        "model_path": "models/minilm_finetuned/",
        "base_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence_transformer_finetuned"
    },
    "inference_timestamp_utc": "2026-04-12T08:00:00+00:00"
}
```

---

## Notes

- The skill supports three model families so the pipeline can remain reusable even if the selected model changes.
- `confidence` is defined as the distance from the `0.5` decision boundary, scaled to the range `[0, 1]`. This is intended for downstream business logic, not as a calibrated uncertainty estimate.
- If `artifact.type` is missing from the upstream state, the skill falls back to `selected_model_id` and `selected_model_metadata.json` to infer the correct loading path.
- Downstream skills should not reload the model; they should consume the structured output written by this stage.
- For Google Colab runs, paths may need to be updated to mounted Drive locations.
