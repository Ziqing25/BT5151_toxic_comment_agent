---
name: toxic-comment-model-evaluation
description: Use when implementing the model evaluation node in the BT5151 toxic comment LangGraph pipeline. Node sits between model-training and model-selection. Evaluates TF-IDF+LR, TF-IDF+LinearSVC, ToxiGen-RoBERTa+LR, and fine-tuned MiniLM on the test set; conducts error analysis and bias audit; saves JSON artifacts consumed by model-selection.
---

# Toxic Comment Model Evaluation Node

## Pipeline Position

```
[model-training] → [model-evaluation] → [model-selection]
```

## Node Contract

## Inputs from Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `project_root` | `str` | Absolute path to project root |
| `test_processed_path` | `str` | Path to `experiments/processed_data/test_set.csv` |
| `train_metadata_path` | `str` | Path to `models/selected_model_metadata.json` |

## Outputs to Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `evaluation_report_path` | `str` | Path to `models/evaluation_report.json` |
| `bias_audit_path` | `str` | Path to `models/bias_audit_summary.json` |

**Input** — artifacts written by `model-training`:

| Source | Files |
|--------|-------|
| `models/` | `tfidf_vectorizer.pkl`, `model_lr.pkl`, `model_linearsvc.pkl`, `model_toxigen_lr.pkl`, `toxigen_test_emb.npy`, `minilm_finetuned/` |
| `processed_data/` | `test_set.csv` (columns: `id`, `comment_text_clean`, `toxic_label`) |

**Output** — JSON files consumed by `model-selection`:

| File | Contents |
|------|----------|
| `evaluation_report.json` | `metrics_per_model` + `selection_candidates` |
| `bias_audit_summary.json` | FP count, keyword counts, conclusion |

## Constants

```python
MODEL_IDS      = ["logistic_regression", "linear_svc", "toxigen_bert_lr", "minilm_ft"]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
BIAS_KEYWORDS  = ["gay", "black", "white", "racial", "race",
                  "muslim", "christian", "jewish", "asian"]
MINILM_BATCH   = 128
```

## Shared Metric Helper

Reuse the same helper from model-training:

```python
def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_score))
                     if len(np.unique(y_true)) > 1 else float("nan"),
    }
```

## Step 1 — Load Models & Test Data

```python
# TF-IDF + sklearn models
tfidf_union  = pickle.load(open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb"))
best_lr      = pickle.load(open(MODELS_DIR / "model_lr.pkl",         "rb"))
best_svc     = pickle.load(open(MODELS_DIR / "model_linearsvc.pkl",  "rb"))

# ToxiGen: pre-computed test embeddings + LR head
X_test_toxigen = np.load(MODELS_DIR / "toxigen_test_emb.npy")
toxigen_lr     = pickle.load(open(MODELS_DIR / "model_toxigen_lr.pkl", "rb"))

# Fine-tuned MiniLM via HuggingFace pipeline
minilm_tokenizer   = AutoTokenizer.from_pretrained(str(MODELS_DIR / "minilm_finetuned"))
minilm_classifier  = pipeline(
    task="text-classification",
    model=str(MODELS_DIR / "minilm_finetuned"),
    tokenizer=minilm_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=False,
    truncation=True, max_length=512, padding="max_length",
)

test_df    = pd.read_csv(PROCESSED_DIR / "test_set.csv")
y_test     = test_df["toxic_label"].values
test_texts = test_df["comment_text_clean"].tolist()
```

## Step 2 — Generate Predictions

```python
# TF-IDF models
X_test_tfidf  = tfidf_union.transform(test_texts)
y_pred_lr     = best_lr.predict(X_test_tfidf)
y_score_lr    = best_lr.predict_proba(X_test_tfidf)[:, 1]
y_pred_svc    = best_svc.predict(X_test_tfidf)
y_score_svc   = best_svc.predict_proba(X_test_tfidf)[:, 1]

# ToxiGen (embeddings already computed during training)
y_pred_toxigen  = toxigen_lr.predict(X_test_toxigen)
y_score_toxigen = toxigen_lr.predict_proba(X_test_toxigen)[:, 1]

# MiniLM — decode LABEL_1 = toxic
results        = minilm_classifier(test_texts, batch_size=MINILM_BATCH)
y_pred_minilm  = np.array([1 if r["label"] == "LABEL_1" else 0 for r in results])
y_score_minilm = np.array([
    r["score"] if r["label"] == "LABEL_1" else 1 - r["score"] for r in results
])
```

## Step 3 — Compute Metrics

```python
metrics = {
    "logistic_regression": compute_metrics(y_test, y_pred_lr,      y_score_lr),
    "linear_svc":          compute_metrics(y_test, y_pred_svc,     y_score_svc),
    "toxigen_bert_lr":     compute_metrics(y_test, y_pred_toxigen, y_score_toxigen),
    "minilm_ft":           compute_metrics(y_test, y_pred_minilm,  y_score_minilm),
}
```

## Step 4 — Visualisations (optional, for reports)

```python
# ROC curves — all four models on one figure
for name, score in [("LR", y_score_lr), ("SVC", y_score_svc),
                    ("ToxiGen+LR", y_score_toxigen), ("MiniLM", y_score_minilm)]:
    fpr, tpr, _ = roc_curve(y_test, score)
    plt.plot(fpr, tpr, label=f"{name} | AUC={metrics[...]['roc_auc']:.4f}")

# Confusion matrices — MiniLM + LR for comparison
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_minilm,
    display_labels=["Non-Toxic", "Toxic"], cmap="Blues"
)
```

## Step 5 — Error Analysis (False Positives + Bias Audit)

```python
error_df = test_df.copy()
error_df["y_pred"]  = y_pred_minilm
error_df["y_score"] = y_score_minilm

false_positives = error_df[(error_df["toxic_label"] == 0) & (error_df["y_pred"] == 1)]

bias_counts = {
    kw: int(false_positives["comment_text_clean"]
            .str.contains(kw, case=False, na=False).sum())
    for kw in BIAS_KEYWORDS
}
```

Run this analysis on **MiniLM** (the expected best model). Flag if any keyword accounts for > 5 % of total FPs.

## Step 6 — Save Artifacts

```python
evaluation_report = {
    "metrics_per_model": {
        mid: {"f1": m["f1"], "auc": m["roc_auc"],
              "precision": m["precision"], "recall": m["recall"]}
        for mid, m in metrics.items()
    },
    "selection_candidates": {
        "logistic_regression": "model_lr.pkl",
        "linear_svc":          "model_linearsvc.pkl",
        "toxigen_bert_lr":     "model_toxigen_lr.pkl",
        "minilm_ft":           "minilm_finetuned",
    },
}

best_id = max(metrics, key=lambda k: metrics[k]["f1"])

selected_model_metadata = {
    "best_model_id":    best_id,
    "selection_reason": "Highest test F1 among evaluated models",
}

bias_audit_summary = {
    "total_false_positives": len(false_positives),
    "bias_keyword_counts":   bias_counts,
    "conclusion":            "...",   # write a 1-sentence interpretation
}

for fname, obj in [
    ("evaluation_report.json",    evaluation_report),
    ("selected_model_metadata.json", selected_model_metadata),
    ("bias_audit_summary.json",   bias_audit_summary),
]:
    with open(EVAL_OUTPUT / fname, "w") as f:
        json.dump(obj, f, indent=4)
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Calling `minilm_classifier` without `truncation=True` | Long comments crash the pipeline; always set `truncation=True, max_length=512` |
| Treating `LABEL_0`/`LABEL_1` as fixed ordinals | Always branch on `r["label"] == "LABEL_1"` for the toxic score; labels depend on training label order |
| Running ToxiGen embeddings through `tfidf_union.transform` | ToxiGen test embeddings are pre-computed `.npy` arrays — load directly, never re-encode |
| Saving figures but not JSON | `model-selection` reads JSON only; ensure all three JSON artifacts are written before the node exits |
| Bias conclusion without normalisation | Report keyword share of total FPs (%), not raw counts, to avoid misleading findings |
