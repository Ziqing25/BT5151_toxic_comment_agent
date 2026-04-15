# Model Training V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `experiments/02_model_training.ipynb` to replace XGBoost with LinearSVC, add char n-gram FeatureUnion to TF-IDF, add frozen ToxiGen-RoBERTa+LR as a 4th model, and replace frozen MiniLM+LR with a fine-tuned MiniLM classifier — all running on Colab GPU.

**Architecture:** Single notebook replacing the existing one. Reads `processed_data/{train,val,test}_set.csv`. Runs four models: TF-IDF+LR, TF-IDF+LinearSVC (both using FeatureUnion word+char), ToxiGen-RoBERTa frozen+LR, fine-tuned MiniLM. Refits sklearn models on train+val before one-time test evaluation. Saves artifacts to `models/` and figures to `figures/`.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, torch, transformers, datasets, evaluate, accelerate, matplotlib, seaborn, Colab GPU, Google Drive

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Modify | `experiments/requirements-experiments.txt` | Remove xgboost; add torch, transformers, datasets, evaluate, accelerate |
| Rewrite | `experiments/02_model_training.ipynb` | Full notebook rewrite — all 4 models |

---

## Task 0: Update requirements-experiments.txt

**Files:**
- Modify: `experiments/requirements-experiments.txt`

- [ ] **Step 1: Replace file contents**

```
pandas
numpy
scikit-learn
torch
transformers
datasets
evaluate
accelerate
sentence-transformers
matplotlib
seaborn
jupyterlab
```

- [ ] **Step 2: Verify diff**

Check that `xgboost` is removed and the five new packages are present.

---

## Task 1: Colab Setup + Imports + Constants Cell

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cell 0 (markdown title), Cell 1 (imports/config)

- [ ] **Step 1: Replace Cell 0 markdown**

```markdown
# 02 Model Training (V2)
Train and compare four models: TF-IDF+LR, TF-IDF+LinearSVC (word+char FeatureUnion), ToxiGen-RoBERTa (frozen)+LR, and fine-tuned MiniLM. All run on Colab GPU.
```

- [ ] **Step 2: Replace Cell 1 with Colab setup + imports + constants**

```python
# --- Colab setup ---
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive/BT5151_toxic_comment_agent/experiments')

# --- Install dependencies ---
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "accelerate",
    "sentence-transformers", "scikit-learn", "seaborn"])

# --- Imports ---
import json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, roc_curve,
)

import transformers
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset
import evaluate as hf_evaluate

warnings.filterwarnings("ignore")

# --- Constants ---
RANDOM_STATE  = 42
PROCESSED_DIR = Path("processed_data")
MODELS_DIR    = Path("models")
FIGURES_DIR   = Path("figures")
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

TFIDF_WORD_PARAMS = dict(analyzer="word", ngram_range=(1, 2), max_features=50_000)
TFIDF_CHAR_PARAMS = dict(analyzer="char_wb", ngram_range=(3, 5), max_features=30_000)
LR_C_GRID         = [0.1, 1.0, 5.0]
SVC_C_GRID        = [0.01, 0.1, 1.0]
THRESHOLD         = 0.5

TOXIGEN_MODEL_NAME = "tomh/toxigen_roberta"
MINILM_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_MAX_LEN     = 128
METRIC_COLUMNS     = ["accuracy", "precision", "recall", "f1", "roc_auc"]
```

- [ ] **Step 3: Verify**

Run the cell. Expected output: `Device: cuda` (on Colab with GPU runtime).

---

## Task 2: Data Load Cell (keep structure, verify columns)

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cells 2-3 (markdown + data load)

- [ ] **Step 1: Replace data load cell**

```python
EXPECTED_COLUMNS = ["id", "comment_text_clean", "toxic_label"]

splits = {}
for split_name in ("train", "val", "test"):
    path = PROCESSED_DIR / f"{split_name}_set.csv"
    df   = pd.read_csv(path)
    assert list(df.columns[:3]) == EXPECTED_COLUMNS, f"{split_name}: unexpected columns {df.columns.tolist()}"
    assert df["comment_text_clean"].notna().all(), f"{split_name}: nulls in comment_text_clean"
    assert df["toxic_label"].isin([0, 1]).all(), f"{split_name}: non-binary toxic_label"
    splits[split_name] = df
    print(f"{split_name}: {len(df):,} rows | toxic rate: {df['toxic_label'].mean():.3f}")

train_set, val_set, test_set = splits["train"], splits["val"], splits["test"]

y_train = train_set["toxic_label"].values
y_val   = val_set["toxic_label"].values
y_test  = test_set["toxic_label"].values

train_texts = train_set["comment_text_clean"].tolist()
val_texts   = val_set["comment_text_clean"].tolist()
test_texts  = test_set["comment_text_clean"].tolist()
```

- [ ] **Step 2: Verify**

Expected output:
```
train: 127,656 rows | toxic rate: 0.102
val:    31,915 rows | toxic rate: 0.102
test:   63,978 rows | toxic rate: 0.101
```

---

## Task 3: Shared Metric Helpers (keep, minor update)

**Files:**
- Keep: `experiments/02_model_training.ipynb` Cells 4-5 (metric helpers)

- [ ] **Step 1: Keep existing `safe_roc_auc` and `compute_metrics` helpers as-is**

Existing helper signature expected:
```python
def compute_metrics(y_true, y_pred, y_score) -> dict:
    # returns dict with keys: accuracy, precision, recall, f1, roc_auc
```

Verify it returns all five keys in `METRIC_COLUMNS`. No changes needed if it does.

---

## Task 4: TF-IDF FeatureUnion Build

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cells 6-7 (TF-IDF feature build)

- [ ] **Step 1: Replace TF-IDF markdown cell**

```markdown
## TF-IDF Feature Build (word + char FeatureUnion)
Combines word n-grams (1,2) and char_wb n-grams (3,5) to capture both normal vocabulary and deliberate misspellings (a$$, f*ck, st0p).
```

- [ ] **Step 2: Replace TF-IDF build cell**

```python
tfidf_union = FeatureUnion([
    ("word", TfidfVectorizer(**TFIDF_WORD_PARAMS)),
    ("char", TfidfVectorizer(**TFIDF_CHAR_PARAMS)),
])

X_train_tfidf = tfidf_union.fit_transform(train_texts)
X_val_tfidf   = tfidf_union.transform(val_texts)
X_test_tfidf  = tfidf_union.transform(test_texts)

print(f"TF-IDF feature matrix shape: train={X_train_tfidf.shape}, val={X_val_tfidf.shape}, test={X_test_tfidf.shape}")
# Expected: train=(127656, ~80000), val=(31915, ~80000), test=(63978, ~80000)
```

- [ ] **Step 3: Verify**

Shape should be `(127656, N)` where N is between 60K–80K. No errors.

---

## Task 5: TF-IDF + LR Tuning

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cells 8-9 (LR tuning)

- [ ] **Step 1: Replace LR tuning cell (structure identical to existing, grid same)**

```python
lr_tuning_records = []
for c_val in LR_C_GRID:
    lr_cand = LogisticRegression(
        C=c_val, class_weight="balanced",
        max_iter=1000, random_state=RANDOM_STATE,
    )
    lr_cand.fit(X_train_tfidf, y_train)
    y_pred  = lr_cand.predict(X_val_tfidf)
    y_score = lr_cand.predict_proba(X_val_tfidf)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_score)
    lr_tuning_records.append({"model": "TF-IDF+LR", "C": c_val, **metrics})
    print(f"LR C={c_val}: F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")

lr_tuning_df = pd.DataFrame(lr_tuning_records)
best_lr_row  = lr_tuning_df.loc[lr_tuning_df["f1"].idxmax()]
best_lr_c    = best_lr_row["C"]
print(f"\nBest LR C={best_lr_c} | Val F1={best_lr_row['f1']:.4f}")
```

- [ ] **Step 2: Verify**

Three rows printed; best C selected. Expected best F1 ≈ 0.77 (consistent with v1).

---

## Task 6: TF-IDF + LinearSVC Tuning

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` — replace XGBoost cells (10-11) with LinearSVC

- [ ] **Step 1: Replace XGBoost markdown with LinearSVC markdown**

```markdown
## LinearSVC Tuning
Linear SVM on the same TF-IDF FeatureUnion. Wrapped in CalibratedClassifierCV to produce probabilities for ROC-AUC.
```

- [ ] **Step 2: Replace XGBoost tuning cell with LinearSVC tuning**

```python
svc_tuning_records = []
for c_val in SVC_C_GRID:
    base_svc  = LinearSVC(C=c_val, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    svc_cand  = CalibratedClassifierCV(base_svc, cv=3)
    svc_cand.fit(X_train_tfidf, y_train)
    y_pred    = svc_cand.predict(X_val_tfidf)
    y_score   = svc_cand.predict_proba(X_val_tfidf)[:, 1]
    metrics   = compute_metrics(y_val, y_pred, y_score)
    svc_tuning_records.append({"model": "TF-IDF+LinearSVC", "C": c_val, **metrics})
    print(f"LinearSVC C={c_val}: F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")

svc_tuning_df = pd.DataFrame(svc_tuning_records)
best_svc_row  = svc_tuning_df.loc[svc_tuning_df["f1"].idxmax()]
best_svc_c    = best_svc_row["C"]
print(f"\nBest LinearSVC C={best_svc_c} | Val F1={best_svc_row['f1']:.4f}")
```

- [ ] **Step 3: Verify**

Three rows printed. No errors. Best C selected.

---

## Task 7: ToxiGen-RoBERTa Embedding Generation

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` — replace MiniLM embedding generation cell (12-13)

- [ ] **Step 1: Add ToxiGen embedding markdown**

```markdown
## ToxiGen-RoBERTa Embedding Generation (Frozen)
Extract mean-pooled last hidden state from `tomh/toxigen_roberta`. Embeddings cached as numpy arrays to avoid re-encoding during LR tuning.
```

- [ ] **Step 2: Add ToxiGen embedding cell**

```python
def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (B, T, H)
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)

def encode_texts(texts, tokenizer, model, batch_size=64, max_length=128):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = tokenizer(batch, truncation=True, padding=True,
                              max_length=max_length, return_tensors="pt").to(DEVICE)
            out   = model(**enc)
            emb   = mean_pool(out, enc["attention_mask"]).cpu().numpy()
            all_embeddings.append(emb)
    return np.vstack(all_embeddings)

toxigen_tokenizer = AutoTokenizer.from_pretrained(TOXIGEN_MODEL_NAME)
toxigen_model     = AutoModel.from_pretrained(TOXIGEN_MODEL_NAME).to(DEVICE)

print("Encoding train set with ToxiGen-RoBERTa...")
toxigen_train_emb = encode_texts(train_texts, toxigen_tokenizer, toxigen_model)
print("Encoding val set...")
toxigen_val_emb   = encode_texts(val_texts,   toxigen_tokenizer, toxigen_model)
print("Encoding test set...")
toxigen_test_emb  = encode_texts(test_texts,  toxigen_tokenizer, toxigen_model)

print(f"ToxiGen embedding shapes: train={toxigen_train_emb.shape}, val={toxigen_val_emb.shape}")
# Expected: train=(127656, 768), val=(31915, 768), test=(63978, 768)

# Cache to disk
np.save(MODELS_DIR / "toxigen_train_emb.npy", toxigen_train_emb)
np.save(MODELS_DIR / "toxigen_val_emb.npy",   toxigen_val_emb)
np.save(MODELS_DIR / "toxigen_test_emb.npy",  toxigen_test_emb)
print("Embeddings cached.")
```

- [ ] **Step 3: Verify**

Shape `(127656, 768)` for train. Files written to `models/`.

---

## Task 8: ToxiGen + LR Tuning

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` — replace MiniLM+LR tuning cell (14-15)

- [ ] **Step 1: Add ToxiGen LR markdown**

```markdown
## ToxiGen-RoBERTa + LR Tuning
```

- [ ] **Step 2: Add ToxiGen LR tuning cell**

```python
toxigen_lr_tuning_records = []
for c_val in LR_C_GRID:
    tox_lr_cand = LogisticRegression(
        C=c_val, class_weight="balanced",
        max_iter=1000, random_state=RANDOM_STATE,
    )
    tox_lr_cand.fit(toxigen_train_emb, y_train)
    y_pred  = tox_lr_cand.predict(toxigen_val_emb)
    y_score = tox_lr_cand.predict_proba(toxigen_val_emb)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_score)
    toxigen_lr_tuning_records.append({"model": "ToxiGen-RoBERTa+LR", "C": c_val, **metrics})
    print(f"ToxiGen+LR C={c_val}: F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")

toxigen_lr_df       = pd.DataFrame(toxigen_lr_tuning_records)
best_toxigen_lr_row = toxigen_lr_df.loc[toxigen_lr_df["f1"].idxmax()]
best_toxigen_lr_c   = best_toxigen_lr_row["C"]
print(f"\nBest ToxiGen+LR C={best_toxigen_lr_c} | Val F1={best_toxigen_lr_row['f1']:.4f}")
```

---

## Task 9: Fine-tune MiniLM

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` — new section replacing old MiniLM frozen embedding section

- [ ] **Step 1: Add fine-tune MiniLM markdown**

```markdown
## Fine-tune MiniLM (end-to-end classification)
Uses `AutoModelForSequenceClassification` with a 2-class head. Trained with HuggingFace Trainer for 3 epochs. Best checkpoint selected by validation F1.
```

- [ ] **Step 2: Add tokenize + dataset cell**

```python
minilm_tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)

def tokenize_fn(examples):
    return minilm_tokenizer(
        examples["text"],
        truncation=True,
        padding=False,          # DataCollatorWithPadding handles this dynamically
        max_length=MINILM_MAX_LEN,
    )

def make_hf_dataset(texts, labels):
    ds = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])

hf_train = make_hf_dataset(train_texts, y_train)
hf_val   = make_hf_dataset(val_texts,   y_val)
print(f"HF train: {len(hf_train)} | HF val: {len(hf_val)}")
```

- [ ] **Step 3: Add compute_metrics_hf helper cell**

```python
f1_metric  = hf_evaluate.load("f1")
acc_metric = hf_evaluate.load("accuracy")

def compute_metrics_hf(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1  = f1_metric.compute(predictions=preds,  references=labels, average="binary")["f1"]
    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    return {"f1": f1, "accuracy": acc}
```

- [ ] **Step 4: Add Trainer setup + train cell**

```python
minilm_model = AutoModelForSequenceClassification.from_pretrained(
    MINILM_MODEL_NAME, num_labels=2
).to(DEVICE)

training_args = TrainingArguments(
    output_dir=str(MODELS_DIR / "minilm_finetuned"),
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=(DEVICE == "cuda"),
    logging_steps=200,
    seed=RANDOM_STATE,
    report_to="none",
)

data_collator = DataCollatorWithPadding(tokenizer=minilm_tokenizer)

trainer = Trainer(
    model=minilm_model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    tokenizer=minilm_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_hf,
)

trainer.train()
print("Fine-tuning complete.")
```

- [ ] **Step 5: Add MiniLM val evaluation cell**

```python
# Evaluate fine-tuned MiniLM on val set to record best val metrics
hf_val_pred  = trainer.predict(hf_val)
minilm_val_logits = hf_val_pred.predictions
minilm_val_preds  = np.argmax(minilm_val_logits, axis=-1)
minilm_val_scores = torch.softmax(torch.tensor(minilm_val_logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
minilm_val_metrics = compute_metrics(y_val, minilm_val_preds, minilm_val_scores)
print(f"Fine-tuned MiniLM Val: F1={minilm_val_metrics['f1']:.4f} | AUC={minilm_val_metrics['roc_auc']:.4f}")
```

---

## Task 10: Validation Comparison (all 4 models)

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cell 16-17 (validation comparison)

- [ ] **Step 1: Replace validation comparison cell**

```python
best_validation_records = [
    {"model_id": "lr",           "model": "TF-IDF+LR",             **{k: best_lr_row[k]         for k in METRIC_COLUMNS}, "C": best_lr_c},
    {"model_id": "linearsvc",    "model": "TF-IDF+LinearSVC",      **{k: best_svc_row[k]        for k in METRIC_COLUMNS}, "C": best_svc_c},
    {"model_id": "toxigen_lr",   "model": "ToxiGen-RoBERTa+LR",    **{k: best_toxigen_lr_row[k] for k in METRIC_COLUMNS}, "C": best_toxigen_lr_c},
    {"model_id": "minilm_ft",    "model": "MiniLM (fine-tuned)",   **minilm_val_metrics},
]

val_comparison_df = pd.DataFrame(best_validation_records)
print("\nValidation comparison:")
print(val_comparison_df[["model"] + METRIC_COLUMNS].to_string(index=False))
```

---

## Task 11: Refit Sklearn Models on train+val

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cells 18-19 (refit)

- [ ] **Step 1: Replace refit cell**

```python
train_val_set    = pd.concat([train_set, val_set], ignore_index=True)
train_val_texts  = train_val_set["comment_text_clean"].tolist()
y_train_val      = train_val_set["toxic_label"].values

# Refit TF-IDF union on full train+val
tfidf_union_final = FeatureUnion([
    ("word", TfidfVectorizer(**TFIDF_WORD_PARAMS)),
    ("char", TfidfVectorizer(**TFIDF_CHAR_PARAMS)),
])
X_train_val_tfidf = tfidf_union_final.fit_transform(train_val_texts)
X_test_final_tfidf = tfidf_union_final.transform(test_texts)

# Refit LR
final_lr = LogisticRegression(C=best_lr_c, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
final_lr.fit(X_train_val_tfidf, y_train_val)

# Refit LinearSVC
final_svc = CalibratedClassifierCV(
    LinearSVC(C=best_svc_c, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE), cv=3
)
final_svc.fit(X_train_val_tfidf, y_train_val)

# Refit ToxiGen+LR on cached train+val embeddings
toxigen_train_val_emb = np.vstack([toxigen_train_emb, toxigen_val_emb])
final_toxigen_lr = LogisticRegression(C=best_toxigen_lr_c, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
final_toxigen_lr.fit(toxigen_train_val_emb, y_train_val)

# Fine-tuned MiniLM is already at its best checkpoint — no sklearn refit needed
print("Refit complete.")
```

---

## Task 12: Held-Out Test Evaluation (all 4 models)

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cells 20-21 (test evaluation)

- [ ] **Step 1: Replace test evaluation cell**

```python
final_test_results = {}

# LR
lr_test_scores = final_lr.predict_proba(X_test_final_tfidf)[:, 1]
lr_test_preds  = (lr_test_scores >= THRESHOLD).astype(int)
final_test_results["lr"] = {
    "model": "TF-IDF+LR",
    **compute_metrics(y_test, lr_test_preds, lr_test_scores),
    "y_score": lr_test_scores, "y_pred": lr_test_preds,
}

# LinearSVC
svc_test_scores = final_svc.predict_proba(X_test_final_tfidf)[:, 1]
svc_test_preds  = (svc_test_scores >= THRESHOLD).astype(int)
final_test_results["linearsvc"] = {
    "model": "TF-IDF+LinearSVC",
    **compute_metrics(y_test, svc_test_preds, svc_test_scores),
    "y_score": svc_test_scores, "y_pred": svc_test_preds,
}

# ToxiGen+LR
tox_test_scores = final_toxigen_lr.predict_proba(toxigen_test_emb)[:, 1]
tox_test_preds  = (tox_test_scores >= THRESHOLD).astype(int)
final_test_results["toxigen_lr"] = {
    "model": "ToxiGen-RoBERTa+LR",
    **compute_metrics(y_test, tox_test_preds, tox_test_scores),
    "y_score": tox_test_scores, "y_pred": tox_test_preds,
}

# Fine-tuned MiniLM
hf_test     = make_hf_dataset(test_texts, y_test)
test_pred   = trainer.predict(hf_test)
minilm_test_logits = test_pred.predictions
minilm_test_preds  = np.argmax(minilm_test_logits, axis=-1)
minilm_test_scores = torch.softmax(torch.tensor(minilm_test_logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
final_test_results["minilm_ft"] = {
    "model": "MiniLM (fine-tuned)",
    **compute_metrics(y_test, minilm_test_preds, minilm_test_scores),
    "y_score": minilm_test_scores, "y_pred": minilm_test_preds,
}

# Print summary
test_summary = pd.DataFrame([
    {"model_id": k, **{m: v[m] for m in ["model"] + METRIC_COLUMNS}}
    for k, v in final_test_results.items()
])
print("\nHeld-out test results:")
print(test_summary[["model"] + METRIC_COLUMNS].to_string(index=False))
```

---

## Task 13: Figure Export (4 models)

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cells 22-23 (figure export)

- [ ] **Step 1: Replace figure export cell**

```python
# Confusion matrices
for model_id, result in final_test_results.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, result["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Clean", "Toxic"]).plot(ax=ax, colorbar=False)
    ax.set_title(result["model"])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_id}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved {model_id}_confusion_matrix.png")

# Combined ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
for model_id, result in final_test_results.items():
    fpr, tpr, _ = roc_curve(y_test, result["y_score"])
    auc = result["roc_auc"]
    ax.plot(fpr, tpr, label=f"{result['model']} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "combined_roc_curve.png", dpi=150)
plt.close(fig)
print("Saved combined_roc_curve.png")

# Summary metric comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
plot_df = test_summary.set_index("model")[METRIC_COLUMNS]
plot_df.T.plot(kind="bar", ax=ax)
ax.set_ylim(0, 1)
ax.set_title("Test Metrics — All Models")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "summary_metric_comparison.png", dpi=150)
plt.close(fig)
print("Saved summary_metric_comparison.png")
```

---

## Task 14: Artifact Save + Metadata

**Files:**
- Rewrite: `experiments/02_model_training.ipynb` Cells 24-25 (artifact save)

- [ ] **Step 1: Save sklearn artifacts**

```python
with open(MODELS_DIR / "tfidf_vectorizer.pkl",   "wb") as f: pickle.dump(tfidf_union_final, f)
with open(MODELS_DIR / "model_lr.pkl",            "wb") as f: pickle.dump(final_lr, f)
with open(MODELS_DIR / "model_linearsvc.pkl",     "wb") as f: pickle.dump(final_svc, f)
with open(MODELS_DIR / "model_toxigen_lr.pkl",    "wb") as f: pickle.dump(final_toxigen_lr, f)
print("Sklearn artifacts saved.")
```

- [ ] **Step 2: Save fine-tuned MiniLM**

```python
# trainer already saved checkpoints; explicitly save final best model
trainer.save_model(str(MODELS_DIR / "minilm_finetuned"))
minilm_tokenizer.save_pretrained(str(MODELS_DIR / "minilm_finetuned"))
print("Fine-tuned MiniLM saved to models/minilm_finetuned/")
```

- [ ] **Step 3: Select best model and write metadata**

```python
best_model_id  = test_summary.loc[test_summary["f1"].idxmax(), "model_id"]   # by test F1
best_model_row = test_summary[test_summary["model_id"] == best_model_id].iloc[0]

metadata = {
    "selected_model":    best_model_row["model"],
    "selected_model_id": best_model_id,
    "selection_reason":  (
        f"{best_model_row['model']} achieved the highest test F1={best_model_row['f1']:.4f}. "
        "Selected for content moderation: highest F1 balances precision and recall on the held-out set."
    ),
    "validation_metrics": {
        row["model_id"]: {m: row[m] for m in METRIC_COLUMNS}
        for _, row in val_comparison_df.iterrows()
        if "model_id" in row
    },
    "test_metrics": {
        model_id: {m: result[m] for m in METRIC_COLUMNS}
        for model_id, result in final_test_results.items()
    },
    "best_hyperparameters": {
        "lr":         {"C": best_lr_c},
        "linearsvc":  {"C": best_svc_c},
        "toxigen_lr": {"C": best_toxigen_lr_c},
        "minilm_ft":  {"epochs": 3, "batch_size": 32, "warmup_ratio": 0.1, "weight_decay": 0.01},
    },
    "artifact_paths": {
        "tfidf_vectorizer":  "models/tfidf_vectorizer.pkl",
        "model_lr":          "models/model_lr.pkl",
        "model_linearsvc":   "models/model_linearsvc.pkl",
        "model_toxigen_lr":  "models/model_toxigen_lr.pkl",
        "minilm_finetuned":  "models/minilm_finetuned/",
        "selected_model_metadata": "models/selected_model_metadata.json",
    },
    "tfidf_parameters": {
        "word": TFIDF_WORD_PARAMS,
        "char": TFIDF_CHAR_PARAMS,
    },
    "toxigen_model_name": TOXIGEN_MODEL_NAME,
    "minilm_model_name":  MINILM_MODEL_NAME,
    "random_state": RANDOM_STATE,
    "threshold":    THRESHOLD,
    "dataset_sizes": {
        "train": int(len(train_set)),
        "val":   int(len(val_set)),
        "train_plus_val": int(len(train_val_set)),
        "test":  int(len(test_set)),
    },
}

with open(MODELS_DIR / "selected_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSelected model: {metadata['selected_model']}")
print(f"Test F1: {best_model_row['f1']:.4f} | AUC: {best_model_row['roc_auc']:.4f}")
print("Metadata saved to models/selected_model_metadata.json")
```

- [ ] **Step 4: Verify all artifacts exist**

```python
expected_artifacts = [
    MODELS_DIR / "tfidf_vectorizer.pkl",
    MODELS_DIR / "model_lr.pkl",
    MODELS_DIR / "model_linearsvc.pkl",
    MODELS_DIR / "model_toxigen_lr.pkl",
    MODELS_DIR / "minilm_finetuned",
    MODELS_DIR / "selected_model_metadata.json",
]
for path in expected_artifacts:
    assert Path(path).exists(), f"Missing: {path}"
    print(f"OK: {path}")
```

---

## Self-Review Checklist

- [x] Colab setup cell (drive mount + os.chdir) — Task 1
- [x] requirements.txt updated (xgboost removed, torch/transformers/datasets/evaluate/accelerate added) — Task 0
- [x] FeatureUnion (word + char) on both LR and LinearSVC — Tasks 4, 5, 6, 11
- [x] LinearSVC wrapped in CalibratedClassifierCV for predict_proba — Tasks 6, 11
- [x] ToxiGen-RoBERTa frozen embeddings generated + cached — Task 7
- [x] ToxiGen+LR tuned separately from LR — Task 8
- [x] Fine-tuned MiniLM replaces (not adds alongside) frozen MiniLM+LR — Task 9
- [x] All 4 models in validation comparison — Task 10
- [x] All sklearn models refit on train+val; fine-tuned MiniLM already at best checkpoint — Task 11
- [x] All 4 models evaluated once on held-out test set — Task 12
- [x] Confusion matrix per model (4 total) + combined ROC + summary bar chart — Task 13
- [x] New artifact names (model_linearsvc, model_toxigen_lr, minilm_finetuned) — Task 14
- [x] metadata.json updated with 4-model structure — Task 14
- [x] `tfidf_vectorizer.pkl` now saves FeatureUnion (not just word vectorizer) — Task 11/14
- [x] `encode_texts` helper reused for both train/val/test embedding generation — Task 7
- [x] `make_hf_dataset` reused for train, val, and test — Tasks 9, 12
- [x] No XGBoost references remain — confirmed by replacing Cells 10-11
