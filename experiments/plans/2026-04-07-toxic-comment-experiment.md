# Toxic Comment Detection — Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible manual experiment pipeline for binary toxic comment detection, from preprocessing through model comparison and final model selection.

**Architecture:** Use one notebook for dataset preparation and EDA, then a second notebook for model-specific feature extraction, tuning, evaluation, and artifact export. Keep preprocessing model-agnostic, use validation F1 for model selection, then refit chosen configurations before one-time held-out test evaluation.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, xgboost, sentence-transformers, matplotlib, seaborn, jupyterlab

---

## Summary

- Create `experiments/requirements-experiments.txt`, `experiments/01_data_preprocessing.ipynb`, and `experiments/02_model_training.ipynb`.
- Standardize dataset interfaces to `id`, `comment_text_clean`, `toxic_label` in `experiments/processed_data/train_set.csv`, `val_set.csv`, and `test_set.csv`.
- Train three candidates: TF-IDF + Logistic Regression, TF-IDF + XGBoost, and MiniLM embeddings + Logistic Regression.
- Save report-ready figures under `experiments/figures/` and model artifacts plus selection metadata under `experiments/models/`.

## Implementation Changes

### Task 0: Environment setup

- Use local `python3.12` rather than `python3` 3.14.3 to avoid package compatibility issues with `xgboost` and `sentence-transformers`.
- Create repo-local virtual environment `.venv` and install from `experiments/requirements-experiments.txt`.
- Put these exact packages in the requirements file: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `sentence-transformers`, `matplotlib`, `seaborn`, `jupyterlab`.
- Use these commands:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experiments/requirements-experiments.txt
python -m ipykernel install --user --name toxic-comment-exp --display-name "Python (toxic-comment-exp)"
jupyter lab
```

### Task 1: `experiments/01_data_preprocessing.ipynb`

- Build the notebook as ordered sections/cells: imports/config, raw data load, label creation, text cleaning helpers, train EDA, labeled test preparation, stratified train/validation split, export checks, CSV/figure saves.
- Implement binary target as `1` when any of the six toxicity labels is positive; keep source label columns only long enough to compute the binary label and co-occurrence heatmap.
- Implement text cleaning to lowercase, remove URLs and HTML, strip non-ASCII/special characters that are not useful for modeling, normalize whitespace, and preserve meaningful token boundaries; do not stem or lemmatize.
- Generate and save these figures to `experiments/figures/`: class balance bar chart, comment-length-by-class distribution, top-token bar charts by class, and six-label co-occurrence heatmap.
- Prepare the held-out test set by filtering out `-1` rows from `raw_data/test_labels.csv`, merging onto `raw_data/test.csv`, creating `toxic_label`, and applying the same cleaning function.
- Split only `train.csv` into 80/20 stratified `train_set` and `val_set` with `random_state=42`; do not mix held-out test rows into training or validation.
- Save only `id`, `comment_text_clean`, `toxic_label` into the three processed CSVs and print shape/class-rate checks after each save.

### Task 2: `experiments/02_model_training.ipynb`

- Build the notebook as ordered sections/cells: imports/config, processed-data load, shared metric helpers, TF-IDF feature build, LR tuning, XGBoost tuning, MiniLM embedding generation, MiniLM+LR tuning, validation comparison, refit, held-out test evaluation, figure export, artifact save, final recommendation.
- Use shared constants: `RANDOM_STATE = 42`, TF-IDF `ngram_range=(1, 2)`, `max_features=50000`, and a fixed classification threshold of `0.5` for all three models.
- Tune on validation F1, reporting accuracy, precision, recall, F1, and ROC-AUC for every candidate configuration.
- Use compact grids so the notebook stays practical:
  - Logistic Regression `C`: `[0.1, 1.0, 5.0]`, `class_weight='balanced'`, `max_iter=1000`
  - XGBoost `n_estimators`: `[100, 200]`, `max_depth`: `[4, 6]`, `learning_rate`: `[0.05, 0.1]`, `scale_pos_weight` computed from the training split
  - MiniLM model: `all-MiniLM-L6-v2`, then LR with the same `C` grid as above
- After selecting the best hyperparameters per model on validation, refit each model on `train_set + val_set` before scoring once on `test_set`.
- Save these artifacts in `experiments/models/`: `tfidf_vectorizer.pkl`, `model_lr.pkl`, `model_xgboost.pkl`, `model_minilm_lr.pkl`, and `selected_model_metadata.json`.
- Make `selected_model_metadata.json` include at least: selected model name, validation F1, held-out test metrics, business justification, artifact paths, and `random_state`.
- Save these evaluation visuals to `experiments/figures/`: one confusion matrix per model, one combined ROC plot, and one summary metric comparison figure/table.

## Public Interfaces / Outputs

- Processed dataset contract: `id`, `comment_text_clean`, `toxic_label`
- Model artifact contract: saved vectorizer only for TF-IDF models; MiniLM model name stored in metadata rather than exporting transformer weights manually
- Selection metadata contract: JSON with `selected_model`, `selection_reason`, `validation_metrics`, `test_metrics`, `artifact_paths`, `random_state`

## Test Plan

- Verify preprocessing outputs have no null `comment_text_clean`, only binary `toxic_label`, and expected row counts: train + val = original train rows, test = labeled test rows only.
- Verify train/validation stratification by checking class-rate drift stays small between full train, `train_set`, and `val_set`.
- Verify model notebook runs top-to-bottom in a fresh kernel using the experiment environment without path edits.
- Verify all three models produce comparable metric rows, confusion matrices, ROC inputs, and saved artifacts.
- Verify `selected_model_metadata.json` matches the notebook conclusion and points to files that exist.

## Assumptions

- Use `python3.12` as the default experiment runtime because it is available locally and is safer than 3.14 for the required ML stack.
- Use top-token bar charts instead of a word cloud for clearer, more defensible EDA and one fewer optional dependency.
- Keep the project notebook-first; do not introduce standalone training scripts in this phase.
- Use validation F1 as the model-selection metric, then discuss precision/recall trade-offs qualitatively in the final comparison.
