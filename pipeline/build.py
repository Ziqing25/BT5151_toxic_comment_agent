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
