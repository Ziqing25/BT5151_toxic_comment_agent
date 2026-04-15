# Model Training V2 — Design Spec

**Date:** 2026-04-08  
**Project:** BT5151 Toxic Comment Detection  
**Scope:** Replace `experiments/02_model_training.ipynb` entirely with a new version that updates the model lineup, adds char n-gram features, and introduces fine-tuned MiniLM.

---

## Context

The original notebook trained three models: TF-IDF+LR, TF-IDF+XGBoost, and frozen MiniLM+LR. XGBoost performed poorly (val F1=0.27, test ROC-AUC=0.56) and is being replaced. The notebook will now run entirely on Colab GPU.

---

## Architecture

Single notebook (`02_model_training.ipynb`) replacing the existing one. Reads from `processed_data/`, writes artifacts to `models/` and figures to `figures/`. Runs end-to-end on Colab GPU with Google Drive mounted.

### Model Lineup

| # | Model | Features |
|---|-------|----------|
| 1 | TF-IDF + Logistic Regression | FeatureUnion: word (1,2) + char_wb (3,5) |
| 2 | TF-IDF + LinearSVC | Same FeatureUnion |
| 3 | ToxiGen-RoBERTa (frozen) + LR | `tomh/toxigen_roberta` mean-pooled embeddings |
| 4 | Fine-tuned MiniLM | `all-MiniLM-L6-v2` + classification head, end-to-end |

---

## Section 1: Colab Setup

First cell mounts Google Drive and sets working directory to the project root on Drive before any imports or data loading.

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive/BT5151_toxic_comment_agent/experiments')
```

---

## Section 2: Feature Engineering

TF-IDF models use a `FeatureUnion` combining word and character n-grams:

```python
FeatureUnion([
    ('word', TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000)),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=30000))
])
```

- Fitted once on `train_set`, applied to val and test
- Captures deliberate misspellings (`a$$`, `f*ck`, `st0p`) via char n-grams
- Combined feature matrix: ~80K features
- Saved as `models/tfidf_vectorizer.pkl` (the full FeatureUnion object)

---

## Section 3: Models

### 3.1 TF-IDF + Logistic Regression

- `LogisticRegression(class_weight='balanced', max_iter=1000)`
- Tune `C` over `[0.1, 1.0, 5.0]` on validation F1
- Saved as `models/model_lr.pkl`

### 3.2 TF-IDF + LinearSVC

- `CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000))` for probability output
- Tune `C` over `[0.01, 0.1, 1.0]` on validation F1
- Saved as `models/model_linearsvc.pkl`

### 3.3 ToxiGen-RoBERTa (frozen) + LR

- Load `tomh/toxigen_roberta` from HuggingFace, run in eval mode
- Extract mean-pooled last hidden state as fixed embeddings
- Cache embeddings to numpy arrays on disk to avoid re-encoding during tuning
- Train `LogisticRegression(class_weight='balanced')`, tune `C` over `[0.1, 1.0, 5.0]`
- Saved as `models/model_toxigen_lr.pkl`

### 3.4 Fine-tuned MiniLM

- `AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', num_labels=2)`
- Tokenize with `AutoTokenizer`, `max_length=128`, truncation + padding
- `TrainingArguments`:
  - `num_train_epochs=3`
  - `per_device_train_batch_size=32`
  - `per_device_eval_batch_size=64`
  - `warmup_ratio=0.1`
  - `weight_decay=0.01`
  - `eval_strategy='epoch'`
  - `save_strategy='epoch'`
  - `load_best_model_at_end=True`
  - `metric_for_best_model='f1'`
- Saved as `models/minilm_finetuned/` (full HuggingFace model directory)

---

## Section 4: Training Flow

1. Colab setup + Drive mount
2. Install dependencies, imports, constants (`RANDOM_STATE=42`)
3. Load `train_set.csv`, `val_set.csv`, `test_set.csv`
4. Fit FeatureUnion on `train_set` → tune LR and LinearSVC on val F1
5. Generate ToxiGen-RoBERTa embeddings for train/val/test → tune LR
6. Fine-tune MiniLM on `train_set`, evaluate each epoch on `val_set`
7. Select best hyperparams for each sklearn model; refit all sklearn models on `train_set + val_set`
8. One-time evaluation on held-out `test_set` for all 4 models
9. Generate figures, save artifacts, write `selected_model_metadata.json`

---

## Section 5: Evaluation & Figures

**Metrics** (all models, test set): Accuracy, Precision, Recall, F1, ROC-AUC

**Figures** saved to `experiments/figures/`:
- `lr_confusion_matrix.png`
- `linearsvc_confusion_matrix.png`
- `toxigen_lr_confusion_matrix.png`
- `minilm_finetuned_confusion_matrix.png`
- `combined_roc_curve.png` — all 4 models on one plot
- `summary_metric_comparison.png` — comparison table/bar chart

---

## Section 6: Artifacts

```
models/
├── tfidf_vectorizer.pkl              # FeatureUnion (word + char)
├── model_lr.pkl
├── model_linearsvc.pkl
├── model_toxigen_lr.pkl
├── minilm_finetuned/                 # HuggingFace model directory
└── selected_model_metadata.json      # Updated with 4-model results
```

`selected_model_metadata.json` includes: selected model name, selection reason, validation metrics (all models), test metrics (all models), artifact paths, hyperparameters, `random_state=42`.

---

## Dependencies

Add to `experiments/requirements-experiments.txt` (replacing `xgboost`):

```
torch
transformers
datasets
evaluate
accelerate
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Replace XGBoost | LinearSVC | XGBoost failed (F1=0.27); LinearSVC is a strong linear baseline with different regularization from LR |
| Char n-grams | FeatureUnion, not separate model | Directly enriches all TF-IDF models; avoids a near-duplicate LR variant |
| ToxiGen-RoBERTa | Frozen extractor + LR | Already pre-trained on toxic content; frozen reps should be strong; avoids fine-tuning two transformers |
| Fine-tuned MiniLM | Replaces frozen MiniLM+LR | Fine-tuned > frozen for same model; frozen metrics already logged for comparison |
| Single notebook | Option C (full rewrite) | All models run on Colab GPU; cleaner than patching the old notebook |
| Colab setup | Drive mount + os.chdir | Matches user's existing workflow |
