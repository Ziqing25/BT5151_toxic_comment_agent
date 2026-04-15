# Toxic Comment Detection — Experiment Design

**Use case**: Content policy violation detection — binary classification of toxic vs clean comments.

**Dataset**: Jigsaw Toxic Comment Classification Challenge (Kaggle)

## Data Summary

| Split | Source | Rows | Notes |
|-------|--------|------|-------|
| Train | `raw_data/train.csv` | 159,571 | 6 binary label columns |
| Test | `raw_data/test.csv` + `raw_data/test_labels.csv` | 63,978 (labeled) | 89,186 rows with label=-1 are discarded |

**Label distribution** (train): ~90% clean, ~10% toxic (any label > 0). Severe imbalance on sub-labels (threat: 0.3%, severe_toxic: 1%).

**Binary label**: `toxic_label = 1` if any of `[toxic, severe_toxic, obscene, threat, insult, identity_hate] > 0`, else `0`.

---

## Phase 1: Data Preprocessing (`01_data_preprocessing.ipynb`)

**Input**: `raw_data/train.csv`, `raw_data/test.csv`, `raw_data/test_labels.csv`

### Steps

1. **Load & inspect** — shape, dtypes, nulls, duplicates.
2. **Create binary label** — collapse 6 labels into single `toxic_label`.
3. **Text cleaning**:
   - Lowercase
   - Remove URLs, HTML tags, special characters / non-ASCII
   - Remove extra whitespace
   - Preserve contractions and punctuation patterns meaningful to toxicity
   - No stemming / lemmatization (both TF-IDF and MiniLM work well without it; aggressive normalization can lose signal)
4. **EDA & visualizations**:
   - Class distribution bar chart (toxic vs clean)
   - Comment length distribution by class
   - Word cloud or top-N frequent words per class
   - Original 6-label co-occurrence heatmap
5. **Prepare test set** — filter `test_labels.csv` to rows with labels != -1, merge with `test.csv`, create binary label.
6. **Stratified train/validation split** from `train.csv` — 80/20, stratified on `toxic_label`.
7. **Save outputs** to `processed_data/`:
   - `train_set.csv`, `val_set.csv`, `test_set.csv` — columns: `id`, `comment_text_clean`, `toxic_label`

**Note**: Feature extraction (TF-IDF, embeddings) happens in the training notebook since it is model-specific.

---

## Phase 2: Model Training & Evaluation (`02_model_training.ipynb`)

**Input**: `processed_data/train_set.csv`, `val_set.csv`, `test_set.csv`

### Three Candidate Models

| # | Model | Feature Method | Details |
|---|-------|---------------|---------|
| 1 | TF-IDF + Logistic Regression | TF-IDF (unigrams + bigrams, max 50K features) | Linear baseline, interpretable |
| 2 | TF-IDF + XGBoost | Same TF-IDF matrix | Non-linear, captures feature interactions |
| 3 | MiniLM + Logistic Regression | `all-MiniLM-L6-v2` (384-dim) | Semantic embeddings, different feature space |

### Training Details

- **Logistic Regression**: `sklearn.linear_model.LogisticRegression`, tune regularization `C` on validation set.
- **XGBoost**: `xgboost.XGBClassifier`, tune `max_depth`, `n_estimators`, `learning_rate` on validation set.
- **MiniLM + LR**: Same LR setup on 384-dim embedding features.
- Handle class imbalance via `scale_pos_weight` (XGBoost) / `class_weight='balanced'` (LR).
- Hyperparameter tuning: grid search over 3-5 values per parameter using validation set performance.

### Evaluation (on held-out test set)

**Required metrics** (binary classification):
- Accuracy, Precision, Recall, F1-score, AUC-ROC

**Visualizations**:
- Confusion matrix for each model
- ROC curves — all 3 models on one plot
- Summary comparison table of all metrics

### Model Selection

- Compare all metrics side by side.
- Discuss precision/recall trade-off in the business context of content moderation (cost of missing toxic content vs cost of wrongly censoring clean content).
- Justify final model selection with reasoning, not just best metric.
- Save selected model + feature extractor to `models/`.

### Saved Artifacts

```
models/
├── tfidf_vectorizer.pkl
├── model_lr.pkl
├── model_xgboost.pkl
├── model_minilm_lr.pkl
└── selected_model_metadata.json   # which model, why, key metrics
```

---

## Project Structure

```
experiments/
├── experiment-design.md           # This file
├── 01_data_preprocessing.ipynb    # Data cleaning, EDA, feature engineering
├── 02_model_training.ipynb        # Train, evaluate, compare, select
├── processed_data/                # Cleaned CSVs
├── models/                        # Saved model artifacts
└── figures/                       # Visualizations for the report
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Task framing | Binary (toxic vs clean) | Maps directly to business question "is this a policy violation?" |
| Train/test split | Train on train.csv (80/20 stratified train/val), evaluate on labeled test rows | Provides genuinely held-out evaluation |
| No stemming/lemmatization | Preserve raw cleaned text | Both TF-IDF and MiniLM handle inflections well; aggressive normalization loses toxicity signal |
| Swap SVM for XGBoost | TF-IDF + XGBoost instead of TF-IDF + SVM | LR and SVM are both linear on same features — XGBoost adds non-linear comparison for richer analysis |
| Feature extraction in training notebook | Not in preprocessing | TF-IDF and embeddings are model-specific; preprocessing stays model-agnostic |
| Reproducibility | `random_state=42` everywhere | All splits, models, and shuffles use the same seed for reproducibility |
