---
name: toxic-comment-model-training
description: Use when implementing the model training node in the BT5151 toxic comment LangGraph pipeline. Node sits between data-preprocessing and model-evaluation. Trains TF-IDF+LR, TF-IDF+LinearSVC, ToxiGen-RoBERTa+LR, and fine-tuned MiniLM; selects best by val F1; saves artifacts to models/.
---

# Toxic Comment Model Training Node

## Pipeline Position

```
[data-preprocessing] → [model-training] → [model-evaluation]
```

## Node Contract

**Input** (from `processed_data/`):

| File | Required columns |
|------|-----------------|
| `train_set.csv` | `id`, `comment_text_clean`, `toxic_label` |
| `val_set.csv`   | same |
| `test_set.csv`  | same |

`toxic_label` is binary (0/1). `comment_text_clean` has no nulls.

**Output** (to `models/`):

| Artifact | Description |
|----------|-------------|
| `tfidf_vectorizer.pkl` | Fitted FeatureUnion (word + char), refitted on train+val |
| `model_lr.pkl` | Best TF-IDF+LR |
| `model_linearsvc.pkl` | Best TF-IDF+LinearSVC (CalibratedClassifierCV) |
| `model_toxigen_lr.pkl` | Best ToxiGen-RoBERTa+LR |
| `minilm_finetuned/` | Fine-tuned MiniLM HuggingFace directory |
| `selected_model_metadata.json` | Best model ID, metrics, hyperparams, artifact paths |

## Constants

```python
RANDOM_STATE  = 42
THRESHOLD     = 0.5
TFIDF_WORD_PARAMS = dict(analyzer="word",    ngram_range=(1,2), max_features=50_000)
TFIDF_CHAR_PARAMS = dict(analyzer="char_wb", ngram_range=(3,5), max_features=30_000)
LR_C_GRID  = [0.1, 1.0, 5.0]
SVC_C_GRID = [0.01, 0.1, 1.0]

TOXIGEN_MODEL_NAME = "tomh/toxigen_roberta"
MINILM_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_MAX_LEN     = 128
METRIC_COLUMNS     = ["accuracy", "precision", "recall", "f1", "roc_auc"]
```

## Shared Metric Helper

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

## Model 1 & 2: TF-IDF FeatureUnion + sklearn

Build once, share across LR and LinearSVC:

```python
tfidf_union = FeatureUnion([
    ("word", TfidfVectorizer(**TFIDF_WORD_PARAMS)),
    ("char", TfidfVectorizer(**TFIDF_CHAR_PARAMS)),
])
X_train_tfidf = tfidf_union.fit_transform(train_texts)
X_val_tfidf   = tfidf_union.transform(val_texts)
```

**Why char n-grams:** captures deliberate misspellings (`a$$`, `f*ck`, `st0p`).

**LR tuning** — pick best C by val F1:

```python
for c_val in LR_C_GRID:
    lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_tfidf, y_train)
    ...
```

**LinearSVC** — wrap with `CalibratedClassifierCV` for probabilities:

```python
CalibratedClassifierCV(
    LinearSVC(C=c_val, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    cv=3
)
```

## Model 3: ToxiGen-RoBERTa (frozen) + LR

Extract mean-pooled embeddings once; cache to avoid re-encoding during C-grid search:

```python
def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state        # (B, T, H)
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)

def encode_texts(texts, tokenizer, model, batch_size=64, max_length=128):
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i+batch_size], truncation=True, padding=True,
                            max_length=max_length, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            yield mean_pool(out, enc["attention_mask"]).cpu().numpy()
```

Then tune LR on top of cached numpy arrays exactly as Model 1.

## Model 4: Fine-tuned MiniLM

`AutoModelForSequenceClassification` with 2-class head; use `Trainer` with val F1 as checkpoint criterion:

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=(DEVICE == "cuda"),
)
```

Use `DataCollatorWithPadding` for dynamic padding (set `padding=False` in tokenizer).

## Refit on train+val Before Test

After selecting best hyperparameters per model on val F1, refit all sklearn models on `train + val` combined, then run one-time test evaluation:

```python
train_val_texts = pd.concat([train_set, val_set])["comment_text_clean"].tolist()
tfidf_union_final = FeatureUnion([...])  # new instance
X_train_val_tfidf = tfidf_union_final.fit_transform(train_val_texts)
```

MiniLM: the best checkpoint (selected by val F1 via `load_best_model_at_end=True`) is already correct — no refit needed.

## Metadata Artifact

Save `models/selected_model_metadata.json` with:

```python
{
  "selected_model":    "...",        # human name of best model
  "selected_model_id": "...",        # one of: lr, linearsvc, toxigen_lr, minilm_ft
  "selection_reason":  "...",
  "test_metrics":      { model_id: { accuracy, precision, recall, f1, roc_auc } },
  "best_hyperparameters": { model_id: { C or epochs/... } },
  "artifact_paths":    { artifact_name: relative_path },
  "threshold":         0.5
}
```

Selection criterion: **highest test F1** (balances precision/recall for content moderation).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Fitting `tfidf_union` twice (once for train, again for train+val) | Always fit exactly once for tuning; create a fresh instance for final refit |
| Forgetting `CalibratedClassifierCV` on LinearSVC | LinearSVC has no `predict_proba`; calibration wrapper is required |
| Dynamic padding set in tokenizer | Set `padding=False` in tokenize_fn; let `DataCollatorWithPadding` handle it |
| Using the trainer's checkpoint path for inference | Call `trainer.save_model()` + `tokenizer.save_pretrained()` explicitly |
