---
name: select-model
description: >
  Applies a multi-criteria weighted decision framework to compare all trained and
  evaluated toxic comment classification models, selects the best candidate for
  production deployment, and records the justified decision in agent state for the
  downstream run-inference stage.
applyTo: "select-model"
---

## When to Use

Invoke this skill **after** `evaluate-models` has computed per-model metrics on the held-out test set and **before** `classify-violation` (run-inference) loads a model for inference. This skill does not train or re-evaluate any model. Its sole purpose is to read existing evaluation artefacts, reason over them with explicit business-context criteria, and record a final, auditable model selection decision.

---

## How to Execute

1. Load `evaluation_report_path` from the `evaluate-models` output — this contains test-set Precision, Recall, F1, and AUC-ROC for all four candidate models.
2. Load `train_metadata_path` from the `train-models` output — this provides validation-set metrics, best hyperparameters, and file paths to all model artefacts.
3. Load `bias_audit_path` from the `evaluate-models` output — this contains false-positive demographic analysis.
4. Build a unified comparison table covering both the validation and test sets.
5. Apply the weighted scoring formula (see Scoring Criteria below) to rank candidates.
6. Cross-check that the ranking is consistent between validation and test sets.
7. Document the selection rationale — including why the chosen model wins on business-relevant criteria and why alternatives were rejected.
8. Write `select_model_output.json` to agent state and save `model_comparison.png` and `bias_audit_chart.png` as supporting visualisations.

---

## Inputs from Agent State (BuildState)

| Key | Source stage | Description |
|---|---|---|
| `evaluation_report_path` | `evaluate-models` | Path to per-model test-set metrics: F1, AUC-ROC, Precision, Recall for all four candidates |
| `bias_audit_path` | `evaluate-models` | Path to false-positive count, demographic keyword frequencies, and bias conclusion |
| `train_metadata_path` | `train-models` | Path to validation-set metrics, hyperparameters, dataset split sizes, and artefact file paths |

---

## Outputs to Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `select_model_output_path` | `str` | Path to `select_model_output.json` (project root) |
| `selected_model_id` | `str` | e.g. `"minilm_ft"` |
| `selection_justification` | `str` | Plain-language rationale |

---

## Output Format

The primary output is `select_model_output.json`. Example structure:

```json
{
    "selected_model_id": "minilm_ft",
    "selected_model_label": "Fine-tuned MiniLM",
    "selection_criteria": {
        "weights": { "AUC-ROC": 0.35, "Recall": 0.30, "F1": 0.20, "Precision": 0.15 },
        "primary_metric": "AUC-ROC",
        "business_priority": "Maximise recall to minimise undetected toxic content"
    },
    "weighted_score": 1.0,
    "test_metrics":       { "auc_roc": 0.9735, "f1": 0.6590, "precision": 0.5097, "recall": 0.9319 },
    "validation_metrics": { "auc_roc": 0.9859, "f1": 0.8332, "precision": 0.8230, "recall": 0.8438 },
    "artifact": {
        "model_path": "models/minilm_finetuned/",
        "base_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence_transformer_finetuned"
    },
    "inference_threshold": 0.5,
    "selection_justification": "Fine-tuned MiniLM achieves the highest AUC-ROC ...",
    "bias_assessment": "False positives are mainly caused by dataset label errors ...",
    "all_candidates": [ ... ]
}
```

---

## Scoring Criteria

The weighted score is computed on **test-set metrics** using min-max normalisation across candidates:

| Metric | Weight | Business Rationale |
|---|---|---|
| **AUC-ROC** | 35% | Threshold-independent; measures overall discriminative ability across all operating points |
| **Recall** | 30% | Missing toxic content (false negative) exposes users to harm — the primary operational risk |
| **F1-score** | 20% | Balances precision and recall; penalises models that achieve high recall by over-flagging |
| **Precision** | 15% | Excessive false positives increase moderator workload and erode user trust, but this is secondary to safety |

Formula for each candidate $m$:

$$\text{score}(m) = 0.35 \cdot \hat{\text{AUC}}(m) + 0.30 \cdot \hat{\text{Recall}}(m) + 0.20 \cdot \hat{\text{F1}}(m) + 0.15 \cdot \hat{\text{Precision}}(m)$$

where $\hat{\cdot}$ denotes min-max normalisation across the candidate set.

---

## Decision Logic Summary

**Selected model: Fine-tuned MiniLM (`minilm_ft`)**

MiniLM is selected because:

1. **Highest AUC-ROC on both evaluation sets** — test 0.9735 (vs next best 0.9644), validation 0.9859 (vs next best 0.9771). AUC is the most reliable single indicator of model quality as it is threshold-independent.
2. **High recall on the test set** (0.932) — MiniLM achieves the second-highest recall among all candidates. ToxiGen+LR has the highest raw recall (0.957) but the lowest precision (0.399), making it operationally unusable due to excessive false alarms. LR similarly suffers from precision collapse (0.45), leaving MiniLM as the only model that combines high recall with acceptable precision.
3. **Clearly best on the validation set across all metrics** — F1 0.833 vs LinearSVC 0.797 and recall 0.844 vs LinearSVC 0.721.
4. **Consistent ranking across both datasets** — the selection is robust; MiniLM ranks first on the weighted score whether evaluated on the in-distribution validation split or the held-out test set.

**Why LinearSVC was not selected:**  
LinearSVC achieves a marginally higher test F1 (0.683 vs 0.659) due to better precision (0.595 vs 0.510), but its recall is 0.802 — meaning approximately 20% of toxic comments go undetected. In a content moderation context, this is an unacceptable miss rate. MiniLM's +13 recall points and higher AUC on both sets outweigh this narrow F1 advantage.

**Why TF-IDF + LR and ToxiGen+LR were not selected:**  
LR achieves high recall but precision of 0.45 — nearly one in two flagged comments would be a false alarm. ToxiGen+LR shows the lowest F1 (0.563) and precision (0.399) of all candidates, indicating that frozen hate-speech-specific embeddings do not transfer well to the broader Jigsaw toxicity taxonomy.

---

## Notes

- **Threshold**: The default inference threshold is `0.5`. The downstream `assess-severity` stage may apply a lower threshold (e.g. `0.3`) for high-priority content categories where recall is paramount.
- **Bias**: The bias audit confirms that demographic keywords appear in approximately 8.8% of false positives (492 of 5,596 FPs), and no identity group is disproportionately affected relative to others. This share does not indicate systematic bias but should be monitored in production.
- **Reproducibility**: All metrics used in this decision are sourced from deterministic evaluation runs over fixed held-out splits (`random_state=42`). The selection is fully reproducible.
- **Agent state contract**: The `artifact.type` field tells the inference stage which loading path to use (`sklearn_pipeline`, `bert_embedding_lr`, or `sentence_transformer_finetuned`). The inference stage must not hard-code a model path; it must read `artifact.model_path` from this output.
