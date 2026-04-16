# pipeline/build.py
from __future__ import annotations

import html
import json
import pickle
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .state import BuildState

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = "__end__"
    START = "__start__"
    StateGraph = None

# ── Constants ──────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
PROCESSED_COLUMNS = ["id", "comment_text_clean", "toxic_label"]

EVAL_TO_TRAIN_KEY = {
    "logistic_regression": "lr",
    "linear_svc":          "linearsvc",
    "toxigen_bert_lr":     "toxigen_lr",
    "minilm_ft":           "minilm_ft",
}

MODEL_LABELS = {
    "logistic_regression": "TF-IDF + LR",
    "linear_svc":          "TF-IDF + LinearSVC",
    "toxigen_bert_lr":     "ToxiGen-RoBERTa + LR",
    "minilm_ft":           "Fine-tuned MiniLM",
}

WEIGHTS = {"AUC-ROC": 0.35, "Recall": 0.30, "F1": 0.20, "Precision": 0.15}


# ── Path helpers ───────────────────────────────────────────────────────────────

def _models_dir(project_root: Path) -> Path:
    d = project_root / "models"
    d.mkdir(exist_ok=True)
    return d


def _processed_dir(project_root: Path) -> Path:
    d = project_root / "experiments" / "processed_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Text cleaning ──────────────────────────────────────────────────────────────

def _clean_text(text: Any) -> str:
    if pd.isna(text):
        return " "
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else " "


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["comment_text_clean"] = out["comment_text"].map(_clean_text)
    out["toxic_label"] = out[LABEL_COLS].gt(0).any(axis=1).astype(int)
    return out


# ── Plain function 1: Preprocessing ───────────────────────────────────────────

def load_and_preprocess_data(state: BuildState) -> BuildState:
    """Load raw CSVs, clean text, build binary label, stratified split, save CSVs."""
    root = Path(state["project_root"])
    train_raw = pd.read_csv(state["raw_train_path"])
    test_raw  = pd.read_csv(state["raw_test_path"])
    test_labels_raw = pd.read_csv(root / "raw_data" / "test_labels.csv")

    # Build train working set
    train_working = _add_features(train_raw)
    train_base = train_working[PROCESSED_COLUMNS].copy()

    # Build labeled test set (filter out rows where any label == -1)
    test_labeled = test_labels_raw.loc[
        test_labels_raw[LABEL_COLS].ne(-1).all(axis=1)
    ].copy()
    test_prepared = test_raw.merge(
        test_labeled[["id", *LABEL_COLS]], on="id", how="inner", validate="one_to_one"
    )
    test_prepared = _add_features(test_prepared)
    test_set = test_prepared[PROCESSED_COLUMNS].copy().reset_index(drop=True)

    # Stratified train / val split (80 / 20)
    train_set, val_set = train_test_split(
        train_base,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_base["toxic_label"],
    )
    train_set = train_set.reset_index(drop=True)
    val_set   = val_set.reset_index(drop=True)

    # Save
    proc = _processed_dir(root)
    train_path = proc / "train_set.csv"
    val_path   = proc / "val_set.csv"
    test_path  = proc / "test_set.csv"

    train_set.to_csv(train_path, index=False)
    val_set.to_csv(val_path,   index=False)
    test_set.to_csv(test_path, index=False)

    summary: dict[str, Any] = {
        "n_raw_train":   len(train_raw),
        "n_train":       len(train_set),
        "n_val":         len(val_set),
        "n_test":        len(test_set),
        "toxic_rate_train": float(train_set["toxic_label"].mean()),
        "toxic_rate_val":   float(val_set["toxic_label"].mean()),
        "toxic_rate_test":  float(test_set["toxic_label"].mean()),
    }

    return {
        "train_processed_path": str(train_path),
        "val_processed_path":   str(val_path),
        "test_processed_path":  str(test_path),
        "preprocessing_summary": summary,
    }


# ── Metric helper ──────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    y_true  = np.asarray(y_true).astype(int)
    y_pred  = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   auc,
    }


# ── Plain function 2: Model training ──────────────────────────────────────────

def train_candidate_models(state: BuildState) -> BuildState:
    """Train 4 candidates, refit on train+val, save artifacts to models/."""
    import torch
    from datasets import Dataset
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC
    from transformers import (
        AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, Trainer, TrainingArguments,
    )
    import evaluate as hf_evaluate

    root   = Path(state["project_root"])
    models = _models_dir(root)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processed splits
    train_set = pd.read_csv(state["train_processed_path"])
    val_set   = pd.read_csv(state["val_processed_path"])
    test_set  = pd.read_csv(state["test_processed_path"])

    train_texts = train_set["comment_text_clean"].tolist()
    val_texts   = val_set["comment_text_clean"].tolist()
    test_texts  = test_set["comment_text_clean"].tolist()
    y_train = train_set["toxic_label"].values
    y_val   = val_set["toxic_label"].values
    y_test  = test_set["toxic_label"].values

    TFIDF_WORD_PARAMS = dict(analyzer="word", ngram_range=(1, 2), max_features=50_000)
    TFIDF_CHAR_PARAMS = dict(analyzer="char_wb", ngram_range=(3, 5), max_features=30_000)
    LR_C_GRID  = [0.1, 1.0, 5.0]
    SVC_C_GRID = [0.01, 0.1, 1.0]
    THRESHOLD  = 0.5
    TOXIGEN_MODEL_NAME = "tomh/toxigen_roberta"
    MINILM_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
    MINILM_MAX_LEN     = 128

    # ── TF-IDF FeatureUnion ────────────────────────────────────────────────────
    tfidf_union = FeatureUnion([
        ("word", TfidfVectorizer(**TFIDF_WORD_PARAMS)),
        ("char", TfidfVectorizer(**TFIDF_CHAR_PARAMS)),
    ])
    X_train_tfidf = tfidf_union.fit_transform(train_texts)
    X_val_tfidf   = tfidf_union.transform(val_texts)

    # ── LR hyperparameter search ───────────────────────────────────────────────
    best_lr_f1, best_lr_c = -1.0, 1.0
    for c in LR_C_GRID:
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        m.fit(X_train_tfidf, y_train)
        f1 = _compute_metrics(y_val, m.predict(X_val_tfidf), m.predict_proba(X_val_tfidf)[:, 1])["f1"]
        if f1 > best_lr_f1:
            best_lr_f1, best_lr_c = f1, c

    # ── LinearSVC hyperparameter search ───────────────────────────────────────
    best_svc_f1, best_svc_c = -1.0, 0.1
    for c in SVC_C_GRID:
        base = LinearSVC(C=c, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
        m = CalibratedClassifierCV(base, cv=3)
        m.fit(X_train_tfidf, y_train)
        f1 = _compute_metrics(y_val, m.predict(X_val_tfidf), m.predict_proba(X_val_tfidf)[:, 1])["f1"]
        if f1 > best_svc_f1:
            best_svc_f1, best_svc_c = f1, c

    # ── ToxiGen-RoBERTa embeddings ─────────────────────────────────────────────
    def _mean_pool(model_output: Any, attention_mask: Any) -> Any:
        token_emb = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return (token_emb * mask).sum(1) / mask.sum(1)

    def _encode_texts(texts: list[str], tokenizer: Any, model: Any, batch: int = 64) -> np.ndarray:
        all_emb = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tokenizer(
                    texts[i:i+batch], truncation=True, padding=True,
                    max_length=128, return_tensors="pt",
                ).to(DEVICE)
                out = model(**enc)
                emb = _mean_pool(out, enc["attention_mask"]).cpu().numpy()
                all_emb.append(emb)
        return np.vstack(all_emb)

    tox_tok   = AutoTokenizer.from_pretrained(TOXIGEN_MODEL_NAME)
    tox_model = AutoModel.from_pretrained(TOXIGEN_MODEL_NAME).to(DEVICE)
    tox_train_emb = _encode_texts(train_texts, tox_tok, tox_model)
    tox_val_emb   = _encode_texts(val_texts,   tox_tok, tox_model)
    tox_test_emb  = _encode_texts(test_texts,  tox_tok, tox_model)
    np.save(models / "toxigen_train_emb.npy", tox_train_emb)
    np.save(models / "toxigen_val_emb.npy",   tox_val_emb)
    np.save(models / "toxigen_test_emb.npy",  tox_test_emb)

    best_tox_f1, best_tox_c = -1.0, 0.1
    for c in LR_C_GRID:
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        m.fit(tox_train_emb, y_train)
        f1 = _compute_metrics(y_val, m.predict(tox_val_emb), m.predict_proba(tox_val_emb)[:, 1])["f1"]
        if f1 > best_tox_f1:
            best_tox_f1, best_tox_c = f1, c

    # ── Fine-tune MiniLM ───────────────────────────────────────────────────────
    minilm_tok = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)

    def _tokenize(examples: dict) -> dict:
        return minilm_tok(examples["text"], truncation=True, padding=False, max_length=MINILM_MAX_LEN)

    def _make_ds(texts: list[str], labels: np.ndarray, desc: str) -> Any:
        ds = Dataset.from_dict({"text": texts, "label": labels.tolist()})
        return ds.map(_tokenize, batched=True, remove_columns=["text"], desc=desc)

    hf_train = _make_ds(train_texts, y_train, "Tokenizing train")
    hf_val   = _make_ds(val_texts,   y_val,   "Tokenizing val")

    f1_metric  = hf_evaluate.load("f1")
    acc_metric = hf_evaluate.load("accuracy")

    def _compute_hf(eval_pred: Any) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1": f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"],
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        }

    minilm_model = AutoModelForSequenceClassification.from_pretrained(
        MINILM_MODEL_NAME, num_labels=2
    ).to(DEVICE)

    train_args = TrainingArguments(
        output_dir=str(models / "minilm_finetuned"),
        num_train_epochs=3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
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

    trainer = Trainer(
        model=minilm_model,
        args=train_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        processing_class=minilm_tok,
        data_collator=DataCollatorWithPadding(tokenizer=minilm_tok),
        compute_metrics=_compute_hf,
    )
    trainer.train()

    # MiniLM val metrics
    hf_val_pred = trainer.predict(hf_val)
    minilm_val_logits = hf_val_pred.predictions
    minilm_val_preds  = np.argmax(minilm_val_logits, axis=-1)
    minilm_val_scores = torch.softmax(
        torch.tensor(minilm_val_logits, dtype=torch.float32), dim=-1
    )[:, 1].numpy()
    minilm_val_m = _compute_metrics(y_val, minilm_val_preds, minilm_val_scores)

    # ── Refit on train+val ─────────────────────────────────────────────────────
    train_val_set = pd.concat([train_set, val_set], ignore_index=True)
    tv_texts      = train_val_set["comment_text_clean"].tolist()
    y_tv          = train_val_set["toxic_label"].values

    tfidf_final = FeatureUnion([
        ("word", TfidfVectorizer(**TFIDF_WORD_PARAMS)),
        ("char", TfidfVectorizer(**TFIDF_CHAR_PARAMS)),
    ])
    X_tv_tfidf = tfidf_final.fit_transform(tv_texts)

    final_lr = LogisticRegression(C=best_lr_c, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    final_lr.fit(X_tv_tfidf, y_tv)

    final_svc = CalibratedClassifierCV(
        LinearSVC(C=best_svc_c, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE), cv=3
    )
    final_svc.fit(X_tv_tfidf, y_tv)

    tox_tv_emb = np.vstack([tox_train_emb, tox_val_emb])
    final_tox_lr = LogisticRegression(C=best_tox_c, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    final_tox_lr.fit(tox_tv_emb, y_tv)

    # Fine-tuned MiniLM: already trained - save best checkpoint
    trainer.save_model(str(models / "minilm_finetuned"))
    minilm_tok.save_pretrained(str(models / "minilm_finetuned"))

    # ── Save sklearn artifacts ─────────────────────────────────────────────────
    with open(models / "tfidf_vectorizer.pkl",  "wb") as f: pickle.dump(tfidf_final, f)
    with open(models / "model_lr.pkl",          "wb") as f: pickle.dump(final_lr, f)
    with open(models / "model_linearsvc.pkl",   "wb") as f: pickle.dump(final_svc, f)
    with open(models / "model_toxigen_lr.pkl",  "wb") as f: pickle.dump(final_tox_lr, f)

    # ── Test set metrics (for metadata) ───────────────────────────────────────
    X_test_tfidf = tfidf_final.transform(test_texts)
    lr_test_s  = final_lr.predict_proba(X_test_tfidf)[:, 1]
    svc_test_s = final_svc.predict_proba(X_test_tfidf)[:, 1]
    tox_test_s = final_tox_lr.predict_proba(tox_test_emb)[:, 1]

    hf_test = _make_ds(test_texts, y_test, "Tokenizing test")
    test_pred = trainer.predict(hf_test)
    minilm_test_logits = test_pred.predictions
    minilm_test_preds  = np.argmax(minilm_test_logits, axis=-1)
    minilm_test_scores = torch.softmax(
        torch.tensor(minilm_test_logits, dtype=torch.float32), dim=-1
    )[:, 1].numpy()

    # Re-evaluate each final (refitted) model on val set for the metadata JSON
    X_val_final = tfidf_final.transform(val_texts)
    val_metrics = {
        "lr":         _compute_metrics(y_val, final_lr.predict(X_val_final),       final_lr.predict_proba(X_val_final)[:, 1]),
        "linearsvc":  _compute_metrics(y_val, final_svc.predict(X_val_final),      final_svc.predict_proba(X_val_final)[:, 1]),
        "toxigen_lr": _compute_metrics(y_val, final_tox_lr.predict(tox_val_emb),   final_tox_lr.predict_proba(tox_val_emb)[:, 1]),
        "minilm_ft":  minilm_val_m,
    }
    test_metrics = {
        "lr":         _compute_metrics(y_test, (lr_test_s  >= THRESHOLD).astype(int), lr_test_s),
        "linearsvc":  _compute_metrics(y_test, (svc_test_s >= THRESHOLD).astype(int), svc_test_s),
        "toxigen_lr": _compute_metrics(y_test, (tox_test_s >= THRESHOLD).astype(int), tox_test_s),
        "minilm_ft":  _compute_metrics(y_test, minilm_test_preds, minilm_test_scores),
    }

    metadata: dict[str, Any] = {
        "validation_metrics": val_metrics,
        "test_metrics":       test_metrics,
        "best_hyperparameters": {
            "lr":         {"C": best_lr_c},
            "linearsvc":  {"C": best_svc_c},
            "toxigen_lr": {"C": best_tox_c},
            "minilm_ft":  {"epochs": 3, "batch_size": 128, "weight_decay": 0.01},
        },
        "artifact_paths": {
            "tfidf_vectorizer":  "models/tfidf_vectorizer.pkl",
            "model_lr":          "models/model_lr.pkl",
            "model_linearsvc":   "models/model_linearsvc.pkl",
            "model_toxigen_lr":  "models/model_toxigen_lr.pkl",
            "minilm_finetuned":  "models/minilm_finetuned/",
            "toxigen_test_emb":  "models/toxigen_test_emb.npy",
            "selected_model_metadata": "models/selected_model_metadata.json",
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

    metadata_path = models / "selected_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "train_metadata_path": str(metadata_path),
        "candidate_model_ids": ["logistic_regression", "linear_svc", "toxigen_bert_lr", "minilm_ft"],
    }
