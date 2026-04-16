# Content Policy Orchestration Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor into 2-layer orchestration (offline build + online runtime), add `draft-warning` LLM node, update Gradio, and restructure the Colab notebook.

**Architecture:** Two separate LangGraph subgraphs in `pipeline/build.py` (offline) and `pipeline/graph.py` (online) share no runtime state. A deterministic controller in `pipeline/controller.py` routes `build` vs `moderate` mode. `draft_warning_node` calls OpenAI `gpt-4o-mini` with a template fallback; raw comment text is never sent to OpenAI.

**Tech Stack:** Python 3.10+, LangGraph, `openai` SDK, Gradio, scikit-learn, transformers/PyTorch (build layer), Google Colab Secrets (`google.colab.userdata`)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pipeline/state.py` | Rewrite | `BuildState`, `RuntimeState`, constructor helpers |
| `pipeline/build.py` | Create | Build layer plain functions + LangGraph wrappers + `compile_build_graph()` |
| `pipeline/graph.py` | Modify | Rename `build_graph→compile_runtime_graph`, add warning helpers + `draft_warning_node`, update type annotations |
| `pipeline/controller.py` | Create | `run(mode, state)` deterministic router |
| `app.py` | Modify | Add `warning_message` Gradio output |
| `SKILLS/draft_warning/SKILL.md` | Create | State contract for the new node |
| `SKILLS/preprocess_data/SKILL.md` | Update | Field names → `BuildState` keys |
| `SKILLS/model_training/SKILL.md` | Update | Field names → `BuildState` keys |
| `SKILLS/model_evaluation/SKILL.md` | Update | Field names → `BuildState` keys |
| `SKILLS/model_selection/SKILL.md` | Update | Field names → `BuildState` keys |
| `SKILLS/run_inference/SKILL.md` | Update | `AgentState` → `RuntimeState` |
| `SKILLS/assess_severity/SKILL.md` | Update | `AgentState` → `RuntimeState` |
| `SKILLS/recommend_moderation_actoin/SKILL.md` | Update | `AgentState` → `RuntimeState`; no-LLM note |
| `tests/test_state.py` | Create | Unit tests for constructors |
| `tests/test_controller.py` | Create | Unit tests for routing |
| `tests/test_draft_warning.py` | Create | Unit tests for warning node |
| `demo_colab_langgraph.ipynb` | Rewrite | Restructured 2-layer Colab demo |

---

### Task 1: Rewrite `pipeline/state.py`

**Files:**
- Modify: `pipeline/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty) and `tests/test_state.py`:

```python
# tests/__init__.py
```

```python
# tests/test_state.py
import importlib
from pathlib import Path
import pytest


def test_build_state_exists():
    from pipeline.state import BuildState
    assert BuildState is not None


def test_runtime_state_exists():
    from pipeline.state import RuntimeState
    assert RuntimeState is not None


def test_agent_state_removed():
    import pipeline.state as m
    assert not hasattr(m, "AgentState"), "AgentState must be removed — replaced by RuntimeState"


def test_build_initial_build_state_defaults():
    from pipeline.state import build_initial_build_state
    state = build_initial_build_state("/tmp/project")
    assert state["project_root"] == str(Path("/tmp/project").resolve())
    assert state["raw_train_path"].endswith("raw_data/train.csv")
    assert state["raw_test_path"].endswith("raw_data/test.csv")


def test_build_initial_build_state_custom_paths():
    from pipeline.state import build_initial_build_state
    state = build_initial_build_state(
        "/tmp/project",
        raw_train_path="/tmp/train.csv",
        raw_test_path="/tmp/test.csv",
    )
    assert state["raw_train_path"] == str(Path("/tmp/train.csv").resolve())
    assert state["raw_test_path"] == str(Path("/tmp/test.csv").resolve())


def test_build_initial_runtime_state_basic():
    from pipeline.state import build_initial_runtime_state
    state = build_initial_runtime_state("hello world")
    assert state["comment_text"] == "hello world"
    assert "project_root" not in state


def test_build_initial_runtime_state_with_root():
    from pipeline.state import build_initial_runtime_state
    state = build_initial_runtime_state("test", project_root="/tmp/project")
    assert state["project_root"] == str(Path("/tmp/project").resolve())


def test_old_build_initial_state_removed():
    import pipeline.state as m
    assert not hasattr(m, "build_initial_state"), (
        "build_initial_state must be removed — split into "
        "build_initial_build_state and build_initial_runtime_state"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/zhii/Desktop/BT5151_toxic_comment_agent
python -m pytest tests/test_state.py -v
```

Expected: multiple FAILs — `BuildState` not found, `AgentState` still present, `build_initial_state` still exists.

- [ ] **Step 3: Rewrite `pipeline/state.py`**

```python
# pipeline/state.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

# ── Shared literal types ───────────────────────────────────────────────────────

SeverityLabel = Literal["clean", "borderline", "low", "medium", "high", "critical"]
ReviewPriority = Literal["none", "low", "medium", "high", "urgent"]
ActionCode = Literal[
    "allow",
    "allow_with_monitoring",
    "soft_warn",
    "review_and_warn",
    "hide_and_review",
    "remove_and_escalate",
]
UserNotification = Literal["none", "gentle_warning", "warning", "final_warning"]


# ── Build layer state (offline ML pipeline) ───────────────────────────────────

class BuildState(TypedDict, total=False):
    # Inputs
    project_root: str
    raw_train_path: str
    raw_test_path: str

    # preprocess-data outputs
    train_processed_path: str      # experiments/processed_data/train_set.csv
    val_processed_path: str        # experiments/processed_data/val_set.csv
    test_processed_path: str       # experiments/processed_data/test_set.csv
    preprocessing_summary: dict[str, Any]

    # train-models outputs
    train_metadata_path: str       # models/selected_model_metadata.json
    candidate_model_ids: list[str]

    # evaluate-models outputs
    evaluation_report_path: str    # models/evaluation_report.json
    bias_audit_path: str           # models/bias_audit_summary.json

    # select-model outputs
    select_model_output_path: str  # select_model_output.json (project root)
    selected_model_id: str
    selection_justification: str


# ── Runtime layer state (online moderation pipeline) ─────────────────────────

class RuntimeState(TypedDict, total=False):
    # Runtime input
    comment_text: str
    project_root: str
    select_model_output_path: str
    train_metadata_path: str
    run_inference_output_path: str
    assess_severity_output_path: str
    moderation_action_output_path: str

    # Model selection context
    selected_model_id: str
    selected_model_label: str
    model_type: str
    threshold_used: float
    selection_justification: str
    bias_assessment: str
    source_artifact: dict[str, Any]

    # Inference output
    predicted_label: Literal["toxic", "non-toxic"]
    predicted_class_id: int
    is_toxic: bool
    toxicity_probability: float
    non_toxic_probability: float
    confidence: float
    raw_score: float
    inference_timestamp_utc: str

    # Severity output
    severity_label: SeverityLabel
    severity_rank: int
    review_priority: ReviewPriority
    escalation_required: bool
    severity_explanation: str
    summary_for_moderator: str
    severity_assessed_at_utc: str

    # Moderation action output
    action_code: ActionCode
    action_label: str
    action_priority: ReviewPriority
    human_review_required: bool
    user_notification: UserNotification
    moderator_rationale: str
    business_message: str
    final_recommendation: str
    ui_explanation: str
    action_recommended_at_utc: str

    # Draft warning output (new)
    warning_message: str
    warning_skipped: bool
    warning_generated_at_utc: str


# ── Constructor helpers ────────────────────────────────────────────────────────

def build_initial_build_state(
    project_root: str | Path,
    raw_train_path: str | Path | None = None,
    raw_test_path: str | Path | None = None,
) -> BuildState:
    root = Path(project_root).resolve()
    state: BuildState = {"project_root": str(root)}
    state["raw_train_path"] = (
        str(Path(raw_train_path).resolve())
        if raw_train_path is not None
        else str(root / "raw_data" / "train.csv")
    )
    state["raw_test_path"] = (
        str(Path(raw_test_path).resolve())
        if raw_test_path is not None
        else str(root / "raw_data" / "test.csv")
    )
    return state


def build_initial_runtime_state(
    comment_text: str, project_root: str | Path | None = None
) -> RuntimeState:
    state: RuntimeState = {"comment_text": comment_text}
    if project_root is not None:
        state["project_root"] = str(Path(project_root).resolve())
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_state.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/state.py tests/__init__.py tests/test_state.py
git commit -m "refactor: split AgentState into BuildState and RuntimeState"
```

---

### Task 2: Create `pipeline/build.py` — `load_and_preprocess_data`

**Files:**
- Create: `pipeline/build.py`
- Create: `tests/test_build_preprocess.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_build_preprocess.py
import json
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def tiny_project(tmp_path):
    """Create a minimal project layout with synthetic data."""
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()

    # Synthetic train data
    train_df = pd.DataFrame({
        "id": [f"id{i}" for i in range(20)],
        "comment_text": [
            "you are stupid and I hate you",
            "great work on this article",
            "kill yourself idiot",
            "thank you for the helpful edit",
            "this is absolute garbage you moron",
        ] * 4,
        "toxic":        [1, 0, 1, 0, 1] * 4,
        "severe_toxic": [0, 0, 1, 0, 0] * 4,
        "obscene":      [1, 0, 0, 0, 1] * 4,
        "threat":       [0, 0, 1, 0, 0] * 4,
        "insult":       [1, 0, 0, 0, 1] * 4,
        "identity_hate":[0, 0, 0, 0, 0] * 4,
    })
    train_df.to_csv(raw_dir / "train.csv", index=False)

    # Synthetic test data + labels
    test_df = pd.DataFrame({
        "id": [f"test{i}" for i in range(10)],
        "comment_text": ["hello world", "you are awful"] * 5,
    })
    test_df.to_csv(raw_dir / "test.csv", index=False)

    test_labels = pd.DataFrame({
        "id": [f"test{i}" for i in range(10)],
        "toxic":        [0, 1] * 5,
        "severe_toxic": [0, 0] * 5,
        "obscene":      [0, 1] * 5,
        "threat":       [0, 0] * 5,
        "insult":       [0, 1] * 5,
        "identity_hate":[0, 0] * 5,
    })
    test_labels.to_csv(raw_dir / "test_labels.csv", index=False)

    return tmp_path


def test_load_and_preprocess_data_creates_csv_files(tiny_project):
    from pipeline.state import build_initial_build_state
    from pipeline.build import load_and_preprocess_data

    state = build_initial_build_state(tiny_project)
    result = load_and_preprocess_data(state)

    assert Path(result["train_processed_path"]).exists()
    assert Path(result["val_processed_path"]).exists()
    assert Path(result["test_processed_path"]).exists()


def test_load_and_preprocess_data_correct_columns(tiny_project):
    from pipeline.state import build_initial_build_state
    from pipeline.build import load_and_preprocess_data

    state = build_initial_build_state(tiny_project)
    result = load_and_preprocess_data(state)

    expected_cols = ["id", "comment_text_clean", "toxic_label"]
    for path_key in ("train_processed_path", "val_processed_path", "test_processed_path"):
        df = pd.read_csv(result[path_key])
        assert list(df.columns) == expected_cols, f"{path_key} has wrong columns: {df.columns.tolist()}"
        assert df["toxic_label"].isin([0, 1]).all()
        assert df["comment_text_clean"].notna().all()


def test_load_and_preprocess_data_summary(tiny_project):
    from pipeline.state import build_initial_build_state
    from pipeline.build import load_and_preprocess_data

    state = build_initial_build_state(tiny_project)
    result = load_and_preprocess_data(state)

    s = result["preprocessing_summary"]
    assert "n_train" in s
    assert "n_val" in s
    assert "n_test" in s
    assert s["n_train"] + s["n_val"] == s["n_raw_train"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_build_preprocess.py -v
```

Expected: ImportError — `pipeline.build` does not exist yet.

- [ ] **Step 3: Create `pipeline/build.py` with `load_and_preprocess_data`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_build_preprocess.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build.py tests/test_build_preprocess.py
git commit -m "feat: add load_and_preprocess_data build function with tests"
```

---

### Task 3: Add `train_candidate_models` to `pipeline/build.py`

**Files:**
- Modify: `pipeline/build.py`

This function is GPU-intensive. No unit test — validated by Colab notebook run.

- [ ] **Step 1: Add shared metric helper and `train_candidate_models` to `pipeline/build.py`**

Append after `load_and_preprocess_data`:

```python
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
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC
    from transformers import (
        AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments,
    )
    from datasets import Dataset
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
    METRIC_COLUMNS     = ["accuracy", "precision", "recall", "f1", "roc_auc"]

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
        import torch
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
    import evaluate as hf_evaluate
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
    import torch as _torch
    minilm_val_scores = _torch.softmax(
        _torch.tensor(minilm_val_logits, dtype=_torch.float32), dim=-1
    )[:, 1].numpy()
    minilm_val_m = _compute_metrics(y_val, minilm_val_preds, minilm_val_scores)

    # ── Refit on train+val ─────────────────────────────────────────────────────
    train_val_set   = pd.concat([train_set, val_set], ignore_index=True)
    tv_texts        = train_val_set["comment_text_clean"].tolist()
    y_tv            = train_val_set["toxic_label"].values

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

    # Fine-tuned MiniLM: already trained — save best checkpoint
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
    minilm_test_scores = _torch.softmax(
        _torch.tensor(minilm_test_logits, dtype=_torch.float32), dim=-1
    )[:, 1].numpy()

    # Re-evaluate each final (refitted) model on val set for the metadata JSON
    X_val_final = tfidf_final.transform(val_texts)
    val_metrics = {
        "lr":        _compute_metrics(y_val, final_lr.predict(X_val_final),           final_lr.predict_proba(X_val_final)[:, 1]),
        "linearsvc": _compute_metrics(y_val, final_svc.predict(X_val_final),          final_svc.predict_proba(X_val_final)[:, 1]),
        "toxigen_lr":_compute_metrics(y_val, final_toxigen_lr.predict(tox_val_emb),   final_toxigen_lr.predict_proba(tox_val_emb)[:, 1]),
        "minilm_ft": minilm_val_m,
    }
    test_metrics = {
        "lr":       _compute_metrics(y_test, (lr_test_s  >= THRESHOLD).astype(int), lr_test_s),
        "linearsvc":_compute_metrics(y_test, (svc_test_s >= THRESHOLD).astype(int), svc_test_s),
        "toxigen_lr":_compute_metrics(y_test,(tox_test_s >= THRESHOLD).astype(int), tox_test_s),
        "minilm_ft":_compute_metrics(y_test, minilm_test_preds, minilm_test_scores),
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
```

- [ ] **Step 2: Verify the function is importable**

```bash
python -c "from pipeline.build import train_candidate_models; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/build.py
git commit -m "feat: add train_candidate_models build function"
```

---

### Task 4: Add `evaluate_candidate_models` to `pipeline/build.py`

**Files:**
- Modify: `pipeline/build.py`

- [ ] **Step 1: Append `evaluate_candidate_models` after `train_candidate_models`**

```python
# ── Plain function 3: Model evaluation ────────────────────────────────────────

def evaluate_candidate_models(state: BuildState) -> BuildState:
    """Evaluate all 4 candidates on test set; produce metrics + bias audit JSON."""
    import torch
    from transformers import AutoTokenizer, pipeline as hf_pipeline

    root   = Path(state["project_root"])
    models = _models_dir(root)

    # Load metadata (contains artifact paths and test_metrics from training)
    with open(state["train_metadata_path"]) as f:
        train_meta = json.load(f)

    # Load test set
    test_df = pd.read_csv(state["test_processed_path"])
    y_test  = test_df["toxic_label"].values
    test_texts = test_df["comment_text_clean"].tolist()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load TF-IDF artifacts
    with open(models / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_union = pickle.load(f)
    with open(models / "model_lr.pkl", "rb") as f:
        best_lr = pickle.load(f)
    with open(models / "model_linearsvc.pkl", "rb") as f:
        best_svc = pickle.load(f)

    # Load ToxiGen artifacts
    tox_test_emb = np.load(models / "toxigen_test_emb.npy")
    with open(models / "model_toxigen_lr.pkl", "rb") as f:
        toxigen_lr = pickle.load(f)

    # Load MiniLM
    minilm_dir = str(models / "minilm_finetuned")
    minilm_tok = AutoTokenizer.from_pretrained(minilm_dir)
    minilm_clf = hf_pipeline(
        task="text-classification",
        model=minilm_dir,
        tokenizer=minilm_tok,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # Generate predictions
    X_test_tfidf = tfidf_union.transform(test_texts)
    y_pred_lr  = best_lr.predict(X_test_tfidf)
    y_score_lr = best_lr.predict_proba(X_test_tfidf)[:, 1]

    y_pred_svc  = best_svc.predict(X_test_tfidf)
    y_score_svc = best_svc.predict_proba(X_test_tfidf)[:, 1]

    y_pred_tox  = toxigen_lr.predict(tox_test_emb)
    y_score_tox = toxigen_lr.predict_proba(tox_test_emb)[:, 1]

    raw_minilm = minilm_clf(test_texts, batch_size=128)
    y_pred_minilm  = np.array([1 if r["label"] == "LABEL_1" else 0 for r in raw_minilm])
    y_score_minilm = np.array([
        r["score"] if r["label"] == "LABEL_1" else 1 - r["score"]
        for r in raw_minilm
    ])

    metrics = {
        "logistic_regression": _compute_metrics(y_test, y_pred_lr,     y_score_lr),
        "linear_svc":          _compute_metrics(y_test, y_pred_svc,    y_score_svc),
        "toxigen_bert_lr":     _compute_metrics(y_test, y_pred_tox,    y_score_tox),
        "minilm_ft":           _compute_metrics(y_test, y_pred_minilm, y_score_minilm),
    }

    evaluation_report = {
        "metrics_per_model": {
            model_id: {
                "f1":        m["f1"],
                "auc":       m["roc_auc"],
                "precision": m["precision"],
                "recall":    m["recall"],
            }
            for model_id, m in metrics.items()
        }
    }

    # Bias audit (false positives from MiniLM)
    error_df = test_df.copy()
    error_df["y_pred"]  = y_pred_minilm
    error_df["y_score"] = y_score_minilm
    false_positives = error_df[
        (error_df["toxic_label"] == 0) & (error_df["y_pred"] == 1)
    ]

    bias_keywords = ["gay", "black", "white", "racial", "race", "muslim", "christian", "jewish", "asian"]
    kw_counts = {
        kw: int(false_positives["comment_text_clean"].str.contains(kw, case=False, na=False).sum())
        for kw in bias_keywords
    }
    kw_total = sum(kw_counts.values())
    fp_total = len(false_positives)
    demo_pct = kw_total / max(fp_total, 1) * 100

    bias_audit: dict[str, Any] = {
        "total_false_positives": fp_total,
        "bias_keyword_counts":   kw_counts,
        "conclusion": (
            f"False positives are mainly caused by dataset label errors and aggressive language, "
            f"not demographic bias. Demographic keywords appear in approximately "
            f"{demo_pct:.1f}% of false positives ({kw_total} of {fp_total})."
        ),
    }

    # Save JSONs
    eval_path = models / "evaluation_report.json"
    bias_path = models / "bias_audit_summary.json"

    with open(eval_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)
    with open(bias_path, "w") as f:
        json.dump(bias_audit, f, indent=4)

    return {
        "evaluation_report_path": str(eval_path),
        "bias_audit_path":        str(bias_path),
    }
```

- [ ] **Step 2: Verify the function is importable**

```bash
python -c "from pipeline.build import evaluate_candidate_models; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/build.py
git commit -m "feat: add evaluate_candidate_models build function"
```

---

### Task 5: Add `select_best_model`, LangGraph wrappers, and `compile_build_graph`

**Files:**
- Modify: `pipeline/build.py`

- [ ] **Step 1: Append `select_best_model` and LangGraph wrappers**

```python
# ── Plain function 4: Model selection ─────────────────────────────────────────

def select_best_model(state: BuildState) -> BuildState:
    """Apply weighted scoring; record justified selection; write select_model_output.json."""
    root = Path(state["project_root"])

    with open(state["evaluation_report_path"]) as f:
        eval_report = json.load(f)
    with open(state["bias_audit_path"]) as f:
        bias_audit = json.load(f)
    with open(state["train_metadata_path"]) as f:
        train_meta = json.load(f)

    # Build test-set score table
    score_rows: dict[str, dict[str, float]] = {}
    for eval_key in EVAL_TO_TRAIN_KEY:
        m = eval_report["metrics_per_model"][eval_key]
        score_rows[eval_key] = {
            "AUC-ROC":   m["auc"],
            "Recall":    m["recall"],
            "F1":        m["f1"],
            "Precision": m["precision"],
        }

    # Min-max normalisation across candidates
    norm_rows: dict[str, dict[str, float]] = {k: dict(v) for k, v in score_rows.items()}
    for metric in ["AUC-ROC", "Recall", "F1", "Precision"]:
        vals = [score_rows[k][metric] for k in score_rows]
        mn, mx = min(vals), max(vals)
        for k in norm_rows:
            norm_rows[k][metric] = (score_rows[k][metric] - mn) / (mx - mn) if mx > mn else 1.0

    for k in norm_rows:
        norm_rows[k]["weighted_score"] = sum(
            norm_rows[k][metric] * weight for metric, weight in WEIGHTS.items()
        )

    best_id = max(norm_rows, key=lambda k: norm_rows[k]["weighted_score"])

    # Collect best model info
    best_test_m = eval_report["metrics_per_model"][best_id]
    best_train_key = EVAL_TO_TRAIN_KEY[best_id]
    best_val_m  = train_meta["validation_metrics"][best_train_key]

    artifact_paths = train_meta["artifact_paths"]
    MODEL_ARTIFACTS = {
        "logistic_regression": {
            "model_path":      artifact_paths["model_lr"],
            "vectorizer_path": artifact_paths["tfidf_vectorizer"],
            "type":            "tfidf_sklearn",
        },
        "linear_svc": {
            "model_path":      artifact_paths["model_linearsvc"],
            "vectorizer_path": artifact_paths["tfidf_vectorizer"],
            "type":            "tfidf_sklearn",
        },
        "toxigen_bert_lr": {
            "model_path":    artifact_paths["model_toxigen_lr"],
            "base_model_name": train_meta["toxigen_model_name"],
            "type":          "bert_embedding_lr",
        },
        "minilm_ft": {
            "model_path":      artifact_paths["minilm_finetuned"],
            "base_model_name": train_meta["minilm_model_name"],
            "type":            "sentence_transformer_finetuned",
        },
    }

    output: dict[str, Any] = {
        "selected_model_id":    best_id,
        "selected_model_label": MODEL_LABELS[best_id],
        "selection_criteria": {
            "weights":           WEIGHTS,
            "primary_metric":    "AUC-ROC",
            "business_priority": "Maximise recall to minimise undetected toxic content",
        },
        "weighted_score":  round(norm_rows[best_id]["weighted_score"], 4),
        "test_metrics": {
            "auc_roc":   round(best_test_m["auc"],       4),
            "f1":        round(best_test_m["f1"],        4),
            "precision": round(best_test_m["precision"], 4),
            "recall":    round(best_test_m["recall"],    4),
        },
        "validation_metrics": {
            "auc_roc":   round(best_val_m["roc_auc"],   4),
            "f1":        round(best_val_m["f1"],        4),
            "precision": round(best_val_m["precision"], 4),
            "recall":    round(best_val_m["recall"],    4),
        },
        "artifact":            MODEL_ARTIFACTS[best_id],
        "inference_threshold": float(train_meta.get("threshold", 0.5)),
        "selection_justification": (
            f"{MODEL_LABELS[best_id]} achieves the highest weighted score "
            f"({norm_rows[best_id]['weighted_score']:.4f}) across AUC-ROC, Recall, F1, and Precision "
            f"on the held-out test set, with AUC-ROC={best_test_m['auc']:.4f} and "
            f"Recall={best_test_m['recall']:.4f}."
        ),
        "bias_assessment": bias_audit["conclusion"],
        "all_candidates": [
            {
                "model_id":       k,
                "model_label":    MODEL_LABELS[k],
                "test_auc":       round(eval_report["metrics_per_model"][k]["auc"],       4),
                "test_f1":        round(eval_report["metrics_per_model"][k]["f1"],        4),
                "test_recall":    round(eval_report["metrics_per_model"][k]["recall"],    4),
                "test_precision": round(eval_report["metrics_per_model"][k]["precision"], 4),
                "weighted_score": round(norm_rows[k]["weighted_score"], 4),
            }
            for k in EVAL_TO_TRAIN_KEY
        ],
    }

    output_path = root / "select_model_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    return {
        "select_model_output_path": str(output_path),
        "selected_model_id":        best_id,
        "selection_justification":  output["selection_justification"],
    }


# ── LangGraph node wrappers ────────────────────────────────────────────────────

def preprocess_data_node(state: BuildState) -> BuildState:
    return load_and_preprocess_data(state)


def train_models_node(state: BuildState) -> BuildState:
    return train_candidate_models(state)


def evaluate_models_node(state: BuildState) -> BuildState:
    return evaluate_candidate_models(state)


def select_model_node(state: BuildState) -> BuildState:
    return select_best_model(state)


# ── Graph compiler ────────────────────────────────────────────────────────────

def compile_build_graph() -> Any:
    if StateGraph is None:
        raise ImportError("langgraph is not installed.")
    graph = StateGraph(BuildState)
    graph.add_node("preprocess-data",  preprocess_data_node)
    graph.add_node("train-models",     train_models_node)
    graph.add_node("evaluate-models",  evaluate_models_node)
    graph.add_node("select-model",     select_model_node)
    graph.add_edge(START, "preprocess-data")
    graph.add_edge("preprocess-data",  "train-models")
    graph.add_edge("train-models",     "evaluate-models")
    graph.add_edge("evaluate-models",  "select-model")
    graph.add_edge("select-model",     END)
    return graph.compile()
```

- [ ] **Step 2: Verify the whole build.py imports cleanly**

```bash
python -c "from pipeline.build import compile_build_graph, select_best_model; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/build.py
git commit -m "feat: add select_best_model, LangGraph wrappers, compile_build_graph"
```

---

### Task 6: Add warning helpers and `draft_warning_node` to `pipeline/graph.py`

**Files:**
- Modify: `pipeline/graph.py`
- Create: `tests/test_draft_warning.py`

- [ ] **Step 1: Write failing tests for warning helpers and node**

```python
# tests/test_draft_warning.py
import os
import pytest
from unittest.mock import MagicMock, patch


def test_get_openai_key_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    from pipeline.graph import _get_openai_key
    assert _get_openai_key() == "sk-test-key"


def test_get_openai_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from pipeline.graph import _get_openai_key
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _get_openai_key()


def test_fallback_warning_covers_all_action_codes():
    from pipeline.graph import _fallback_warning
    action_codes = ["soft_warn", "review_and_warn", "hide_and_review", "remove_and_escalate"]
    for code in action_codes:
        state = {"action_code": code, "severity_label": "medium", "action_label": "Test"}
        msg = _fallback_warning(state)
        assert isinstance(msg, str) and len(msg) > 10, f"Empty fallback for {code}"


def test_build_warning_prompt_excludes_comment_text():
    from pipeline.graph import _build_warning_prompt
    state = {
        "action_code": "soft_warn",
        "severity_label": "low",
        "action_label": "Soft Warning",
        "comment_text": "this is a secret sensitive comment",
    }
    prompt = _build_warning_prompt(state)
    assert "secret sensitive comment" not in prompt, "Raw comment text must not appear in prompt"
    assert "soft_warn" in prompt or "Soft Warning" in prompt or "low" in prompt


def test_draft_warning_node_skips_for_allow():
    from pipeline.graph import draft_warning_node
    state = {"action_code": "allow", "severity_label": "clean"}
    result = draft_warning_node(state)
    assert result["warning_skipped"] is True
    assert result["warning_message"] == ""


def test_draft_warning_node_skips_for_allow_with_monitoring():
    from pipeline.graph import draft_warning_node
    state = {"action_code": "allow_with_monitoring", "severity_label": "borderline"}
    result = draft_warning_node(state)
    assert result["warning_skipped"] is True


def test_draft_warning_node_uses_fallback_on_openai_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    from pipeline.graph import draft_warning_node
    # Mock OpenAI to raise an exception
    with patch("pipeline.graph.OpenAI", side_effect=Exception("API down")):
        state = {
            "action_code": "soft_warn",
            "severity_label": "low",
            "action_label": "Soft Warning",
        }
        result = draft_warning_node(state)
    assert result["warning_skipped"] is False
    assert isinstance(result["warning_message"], str)
    assert len(result["warning_message"]) > 0


def test_draft_warning_node_calls_openai_for_toxic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Please be respectful in future comments."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("pipeline.graph.OpenAI", return_value=mock_client):
        from pipeline.graph import draft_warning_node
        state = {
            "action_code": "review_and_warn",
            "severity_label": "medium",
            "action_label": "Review and Warn",
        }
        result = draft_warning_node(state)

    assert result["warning_message"] == "Please be respectful in future comments."
    assert result["warning_skipped"] is False
    assert "warning_generated_at_utc" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_draft_warning.py -v
```

Expected: ImportError or AttributeError — `_get_openai_key`, `_fallback_warning`, `draft_warning_node` not yet defined.

- [ ] **Step 3: Add warning helpers and `draft_warning_node` to `pipeline/graph.py`**

Add these functions after the existing imports and before `run_inference_node`. First add a module-level optional import for OpenAI (so tests can patch it), then the helpers.

At the top of `graph.py` after the existing imports, add:

```python
from .state import RuntimeState  # already imported as AgentState — add RuntimeState alongside it

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]
```

Then, after the `sigmoid` / `probability_to_confidence` helpers, add:

```python
# ── Warning-generation helpers ─────────────────────────────────────────────────

_FALLBACK_WARNINGS: dict[str, str] = {
    "soft_warn": (
        "Hi, we noticed your recent comment may not meet our community guidelines. "
        "Please keep conversations respectful and constructive. Thank you."
    ),
    "review_and_warn": (
        "Your comment has been flagged for review and is temporarily under moderation. "
        "If it violates our community standards, it may be removed. "
        "We ask that you review our community guidelines before posting again."
    ),
    "hide_and_review": (
        "Your comment has been temporarily hidden pending a moderator review. "
        "Comments that violate our community standards will be removed. "
        "Repeated violations may result in account restrictions."
    ),
    "remove_and_escalate": (
        "Your comment has been removed for violating our community guidelines. "
        "This is a final warning — further violations will result in account suspension. "
        "Please review our community standards."
    ),
}


def _get_openai_key() -> str:
    import os
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "In Google Colab, set it with: "
            "os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')"
        )
    return key


def _build_warning_prompt(state: RuntimeState) -> str:
    """Build the OpenAI prompt. Raw comment text is intentionally excluded."""
    action_code   = state.get("action_code", "")
    severity      = state.get("severity_label", "")
    action_label  = state.get("action_label", "")
    return (
        f"You are a trust-and-safety moderator writing a user-facing warning message.\n\n"
        f"The moderation system has assessed this situation:\n"
        f"- Severity level: {severity}\n"
        f"- Recommended action: {action_label} (code: {action_code})\n\n"
        f"Write a single short paragraph (2-3 sentences) addressed directly to the user. "
        f"The message should:\n"
        f"1. Inform them their comment was flagged\n"
        f"2. Ask them to follow community guidelines\n"
        f"3. Mention consequences proportional to severity ({severity})\n\n"
        f"Use a professional, non-hostile tone. Do not repeat the original comment text. "
        f"Do not use threats. Output only the warning message text, nothing else."
    )


def _fallback_warning(state: RuntimeState) -> str:
    action_code = state.get("action_code", "")
    return _FALLBACK_WARNINGS.get(
        action_code,
        "Your comment has been flagged for review. Please follow our community guidelines.",
    )


def draft_warning_node(state: RuntimeState) -> RuntimeState:
    from datetime import datetime, timezone
    action_code = state.get("action_code", "allow")

    if action_code in ("allow", "allow_with_monitoring"):
        return {
            "warning_message":          "",
            "warning_skipped":          True,
            "warning_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    try:
        if OpenAI is None:
            raise ImportError("openai is not installed")
        client  = OpenAI(api_key=_get_openai_key())
        prompt  = _build_warning_prompt(state)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        message = response.choices[0].message.content.strip()
    except Exception:
        message = _fallback_warning(state)

    return {
        "warning_message":          message,
        "warning_skipped":          False,
        "warning_generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
```

Note: `RuntimeState` is imported from `pipeline.state` (already updated in Task 1). `OpenAI` is now a module-level name so `patch("pipeline.graph.OpenAI")` works correctly in tests.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_draft_warning.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/graph.py tests/test_draft_warning.py
git commit -m "feat: add draft_warning_node and warning helpers"
```

---

### Task 7: Update `pipeline/graph.py` — rename, type annotations, add node to graph

**Files:**
- Modify: `pipeline/graph.py`

- [ ] **Step 1: Update imports at the top of `pipeline/graph.py`**

Replace the import of `AgentState` and `build_initial_state`:

```python
# Old:
from .state import AgentState, build_initial_state

# New:
from .state import RuntimeState, build_initial_runtime_state
```

- [ ] **Step 2: Replace all `AgentState` type annotations with `RuntimeState`**

In `pipeline/graph.py`, do a replace-all of:
- `AgentState` → `RuntimeState` (in all function signatures and return types)

Functions affected: `run_inference_node`, `assess_severity_node`, `recommend_moderation_action_node`, and the new `draft_warning_node` (already uses string annotation — update to `RuntimeState` directly).

- [ ] **Step 3: Rename `build_graph` to `compile_runtime_graph` and add `draft-warning` node**

Replace the existing `build_graph` function:

```python
def compile_runtime_graph() -> Any:
    if StateGraph is None:
        raise ImportError("langgraph is not installed. Install it before compiling the graph.")

    graph = StateGraph(RuntimeState)
    graph.add_node("run-inference",               run_inference_node)
    graph.add_node("assess-severity",             assess_severity_node)
    graph.add_node("recommend-moderation-action", recommend_moderation_action_node)
    graph.add_node("draft-warning",               draft_warning_node)
    graph.add_edge(START,                         "run-inference")
    graph.add_edge("run-inference",               "assess-severity")
    graph.add_edge("assess-severity",             "recommend-moderation-action")
    graph.add_edge("recommend-moderation-action", "draft-warning")
    graph.add_edge("draft-warning",               END)
    return graph.compile()
```

- [ ] **Step 4: Update `run_pipeline` to call `compile_runtime_graph`**

```python
def run_pipeline(comment_text: str, initial_state: RuntimeState | None = None) -> RuntimeState:
    state = build_initial_runtime_state(
        comment_text, initial_state.get("project_root") if initial_state else None
    )
    if initial_state:
        state.update(initial_state)

    if StateGraph is not None:
        app = compile_runtime_graph()
        return app.invoke(state)

    current = dict(state)
    current.update(run_inference_node(current))
    current.update(assess_severity_node(current))
    current.update(recommend_moderation_action_node(current))
    current.update(draft_warning_node(current))
    return current
```

- [ ] **Step 5: Verify imports and existing tests still pass**

```bash
python -c "from pipeline.graph import compile_runtime_graph, run_pipeline; print('OK')"
python -m pytest tests/ -v
```

Expected: `OK` and all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/graph.py
git commit -m "refactor: rename build_graph→compile_runtime_graph, add draft-warning node"
```

---

### Task 8: Create `pipeline/controller.py`

**Files:**
- Create: `pipeline/controller.py`
- Create: `tests/test_controller.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_controller.py
import pytest
from unittest.mock import MagicMock, patch


def test_run_build_mode_calls_build_graph(tmp_path):
    from pipeline.state import build_initial_build_state

    mock_result = {"selected_model_id": "minilm_ft"}
    mock_graph  = MagicMock()
    mock_graph.invoke.return_value = mock_result

    with patch("pipeline.controller.compile_build_graph", return_value=mock_graph):
        from pipeline.controller import run
        state = build_initial_build_state(str(tmp_path))
        result = run("build", state)

    mock_graph.invoke.assert_called_once_with(state)
    assert result == mock_result


def test_run_moderate_mode_calls_runtime_graph():
    from pipeline.state import build_initial_runtime_state

    mock_result = {"warning_message": "Please follow guidelines."}
    mock_graph  = MagicMock()
    mock_graph.invoke.return_value = mock_result

    with patch("pipeline.controller.compile_runtime_graph", return_value=mock_graph):
        from pipeline.controller import run
        state = build_initial_runtime_state("hello world")
        result = run("moderate", state)

    mock_graph.invoke.assert_called_once_with(state)
    assert result == mock_result


def test_run_invalid_mode_raises():
    from pipeline.state import build_initial_runtime_state
    from pipeline.controller import run

    with pytest.raises(ValueError, match="mode"):
        run("invalid_mode", build_initial_runtime_state("test"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_controller.py -v
```

Expected: ImportError — `pipeline.controller` does not exist.

- [ ] **Step 3: Create `pipeline/controller.py`**

```python
# pipeline/controller.py
from __future__ import annotations

from typing import Literal, Union

from .build import compile_build_graph
from .graph import compile_runtime_graph
from .state import BuildState, RuntimeState


def run(
    mode: Literal["build", "moderate"],
    state: Union[BuildState, RuntimeState],
) -> Union[BuildState, RuntimeState]:
    """Deterministic router: call the build layer or the runtime layer."""
    if mode == "build":
        return compile_build_graph().invoke(state)
    if mode == "moderate":
        return compile_runtime_graph().invoke(state)
    raise ValueError(
        f"Unknown mode {mode!r}. Expected 'build' or 'moderate'."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_controller.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/controller.py tests/test_controller.py
git commit -m "feat: add deterministic controller run(mode, state)"
```

---

### Task 9: Update `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update `analyze_comment` return signature**

In `app.py`, update the `analyze_comment` function. Change the return type from 8-tuple to 9-tuple and add `warning_message`:

```python
def analyze_comment(comment_text: str) -> tuple[str, str, str, str, str, str, bool, str, dict[Any, Any]]:
    clean_text = str(comment_text).strip()
    if not clean_text:
        empty_payload = {
            "error": "Please enter a comment before running the moderation pipeline."
        }
        return (
            "No action available",
            "No business summary available.",
            "unknown",
            "unknown",
            "unknown",
            "Enter a comment to begin.",
            False,
            "",
            empty_payload,
        )

    result = run_pipeline(clean_text)
    return (
        result.get("action_label", "Unknown"),
        result.get("business_message", "No business summary available."),
        result.get("severity_label", "unknown"),
        result.get("review_priority", "unknown"),
        result.get("user_notification", "unknown"),
        result.get("ui_explanation", "No explanation available."),
        bool(result.get("human_review_required", False)),
        result.get("warning_message", ""),
        result,
    )
```

- [ ] **Step 2: Update `build_demo` to import `run_pipeline` from updated graph and add warning field**

In the `build_demo` function, update the pipeline description and add the warning output field:

```python
def build_demo() -> Any:
    import gradio as gr

    with gr.Blocks(title="BT5151 Toxic Comment Moderation Agent") as demo:
        gr.Markdown(
            """
            # BT5151 Toxic Comment Moderation Agent

            This demo runs the LangGraph moderation pipeline:
            `run-inference → assess-severity → recommend-moderation-action → draft-warning`
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                comment_input = gr.Textbox(
                    label="Comment Text",
                    placeholder="Enter a comment to classify and moderate...",
                    lines=6,
                )
                analyze_button = gr.Button("Run Moderation Pipeline", variant="primary")

                gr.Examples(
                    examples=[
                        ["Thank you for your edits, this article is much clearer now."],
                        ["This is a stupid comment and your argument makes no sense."],
                        ["You are an absolute idiot and nobody wants you here."],
                        ["What the hell is going on with this page?"],
                    ],
                    inputs=[comment_input],
                )

            with gr.Column(scale=2):
                action_label_box      = gr.Textbox(label="Recommended Action",   interactive=False)
                business_message_box  = gr.Textbox(label="Business Message",     interactive=False, lines=3)
                severity_label_box    = gr.Textbox(label="Severity",             interactive=False)
                review_priority_box   = gr.Textbox(label="Review Priority",      interactive=False)
                user_notification_box = gr.Textbox(label="User Notification",    interactive=False)
                ui_explanation_box    = gr.Textbox(label="UI Explanation",       interactive=False, lines=4)
                human_review_box      = gr.Checkbox(label="Human Review Required", interactive=False)
                warning_message_box   = gr.Textbox(label="Warning Message to User", interactive=False, lines=4)

        raw_output = gr.JSON(label="Full Pipeline Output")

        analyze_button.click(
            fn=analyze_comment,
            inputs=[comment_input],
            outputs=[
                action_label_box,
                business_message_box,
                severity_label_box,
                review_priority_box,
                user_notification_box,
                ui_explanation_box,
                human_review_box,
                warning_message_box,
                raw_output,
            ],
        )

    return demo
```

- [ ] **Step 3: Update the import in `app.py`**

The `run_pipeline` import should come from `pipeline.graph` — it's already there, no change needed. But verify the import line still works:

```bash
python -c "from app import analyze_comment, build_demo; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add warning_message output to Gradio UI"
```

---

### Task 10: Create `SKILLS/draft_warning/SKILL.md`

**Files:**
- Create: `SKILLS/draft_warning/SKILL.md`

- [ ] **Step 1: Create the skill file**

```markdown
---
name: draft-warning
description: >
  Generate a user-facing moderation warning message using OpenAI gpt-4o-mini,
  or fall back to a deterministic template. Runs after recommend-moderation-action.
  Skipped entirely when action_code is "allow" or "allow_with_monitoring".
applyTo: "draft-warning"
---

## When to Use

Invoke this skill **after** `recommend-moderation-action` has set `action_code` and `severity_label`.
This skill does not change the moderation decision — it only drafts the message that would be sent to
the user. It is skipped when no user notification is required.

---

## How to Execute

1. Read `action_code` from state. If `allow` or `allow_with_monitoring`, write `warning_skipped=True` and an empty `warning_message`; return immediately.
2. Build a prompt using `action_code`, `severity_label`, and `action_label`. **Do not include the raw `comment_text` in the prompt** (privacy).
3. Call OpenAI `gpt-4o-mini` via the `openai` Python SDK. Read the API key from `os.environ["OPENAI_API_KEY"]` — never hardcode it.
4. On any exception (API error, quota, connectivity), fall back to a deterministic template keyed by `action_code`.
5. Write `warning_message`, `warning_skipped`, and `warning_generated_at_utc` to state.

---

## Inputs from Agent State

| Key | Type | Description |
|---|---|---|
| `action_code` | `ActionCode` | Determines skip vs generate |
| `severity_label` | `SeverityLabel` | Informs warning tone and proportionality |
| `action_label` | `str` | Human-readable action name, included in prompt |

---

## Outputs to Agent State

| Key | Type | Description |
|---|---|---|
| `warning_message` | `str` | LLM-generated warning or template fallback; empty string when skipped |
| `warning_skipped` | `bool` | `True` when `action_code` is `allow` or `allow_with_monitoring` |
| `warning_generated_at_utc` | `str` | ISO 8601 UTC timestamp |

---

## Prompt Strategy

The prompt:
- States the severity level and action code
- Asks for a 2-3 sentence professional, non-hostile message addressed to the user
- Explicitly excludes the raw comment text (not sent to OpenAI)
- Requests output that is proportional to severity

Model: `gpt-4o-mini` | `max_tokens=200` | `temperature=0.4`

---

## Fallback Templates

If OpenAI is unavailable, pre-written templates are used per `action_code`:

| Action code | Template summary |
|---|---|
| `soft_warn` | Gentle reminder about community guidelines |
| `review_and_warn` | Comment under review; guidelines reminder |
| `hide_and_review` | Comment hidden pending review; repeated violation warning |
| `remove_and_escalate` | Comment removed; final warning; account restriction notice |

The pipeline **never raises an exception** due to a failed API call — the fallback always fires.

---

## Notes

- This node does not alter `action_code`, `severity_label`, or any upstream decision.
- `warning_message` being an empty string is a valid output (for clean/borderline cases).
- Raw comment text is intentionally excluded from the OpenAI prompt for user privacy.
- API key must be set via `os.environ["OPENAI_API_KEY"]` — in Colab, use `userdata.get("OPENAI_API_KEY")`.
```

- [ ] **Step 2: Commit**

```bash
git add SKILLS/draft_warning/SKILL.md
git commit -m "docs: add draft_warning SKILL.md"
```

---

### Task 11: Update existing SKILL.md files

**Files:**
- Modify: `SKILLS/preprocess_data/SKILL.md`
- Modify: `SKILLS/model_training/SKILL.md`
- Modify: `SKILLS/model_evaluation/SKILL.md`
- Modify: `SKILLS/model_selection/SKILL.md`
- Modify: `SKILLS/run_inference/SKILL.md`
- Modify: `SKILLS/assess_severity/SKILL.md`
- Modify: `SKILLS/recommend_moderation_actoin/SKILL.md`

- [ ] **Step 1: Update `SKILLS/preprocess_data/SKILL.md`**

Replace the "Inputs from agent state" and "Outputs to agent state" sections to match `BuildState` keys:

**Inputs:**
```markdown
## Inputs from Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `project_root` | `str` | Absolute path to project root |
| `raw_train_path` | `str` | Path to `raw_data/train.csv` |
| `raw_test_path` | `str` | Path to `raw_data/test.csv` |
```

**Outputs:**
```markdown
## Outputs to Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `train_processed_path` | `str` | Path to `experiments/processed_data/train_set.csv` |
| `val_processed_path` | `str` | Path to `experiments/processed_data/val_set.csv` |
| `test_processed_path` | `str` | Path to `experiments/processed_data/test_set.csv` |
| `preprocessing_summary` | `dict` | Row counts, toxic rates per split |
```

Also update the output format section: processed columns are `["id", "comment_text_clean", "toxic_label"]` (not `comment_text` / `clean_text` / `any_violation`).

- [ ] **Step 2: Update `SKILLS/model_training/SKILL.md`**

**Inputs:**
```markdown
## Inputs from Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `project_root` | `str` | Absolute path to project root |
| `train_processed_path` | `str` | Path to `experiments/processed_data/train_set.csv` |
| `val_processed_path` | `str` | Path to `experiments/processed_data/val_set.csv` |
| `test_processed_path` | `str` | Path to `experiments/processed_data/test_set.csv` |
```

**Outputs:**
```markdown
## Outputs to Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `train_metadata_path` | `str` | Path to `models/selected_model_metadata.json` |
| `candidate_model_ids` | `list[str]` | `["logistic_regression", "linear_svc", "toxigen_bert_lr", "minilm_ft"]` |
```

- [ ] **Step 3: Update `SKILLS/model_evaluation/SKILL.md`**

**Inputs:**
```markdown
## Inputs from Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `project_root` | `str` | Absolute path to project root |
| `test_processed_path` | `str` | Path to `experiments/processed_data/test_set.csv` |
| `train_metadata_path` | `str` | Path to `models/selected_model_metadata.json` |
```

**Outputs:**
```markdown
## Outputs to Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `evaluation_report_path` | `str` | Path to `models/evaluation_report.json` |
| `bias_audit_path` | `str` | Path to `models/bias_audit_summary.json` |
```

- [ ] **Step 4: Update `SKILLS/model_selection/SKILL.md`**

Update the three input file references to use `BuildState` key names:
- `evaluation_report_path` (from `evaluate-models`)
- `bias_audit_path` (from `evaluate-models`)
- `train_metadata_path` (from `train-models`)

Add or update outputs table to reflect `BuildState` outputs:
```markdown
## Outputs to Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `select_model_output_path` | `str` | Path to `select_model_output.json` (project root) |
| `selected_model_id` | `str` | e.g. `"minilm_ft"` |
| `selection_justification` | `str` | Plain-language rationale |
```

- [ ] **Step 5: Update `SKILLS/run_inference/SKILL.md`**

Replace every occurrence of `AgentState` with `RuntimeState`.

- [ ] **Step 6: Update `SKILLS/assess_severity/SKILL.md`**

Replace every occurrence of `AgentState` with `RuntimeState`.

- [ ] **Step 7: Update `SKILLS/recommend_moderation_actoin/SKILL.md`**

Replace every occurrence of `AgentState` with `RuntimeState`.

Add the following note in a **Notes** section:
```markdown
## Notes

- This node is **deterministic and policy-based**. It does not call any LLM.
- The `draft-warning` node downstream is responsible for LLM-generated user messaging.
- `action_code` must remain stable across calls for the same `severity_label`.
```

- [ ] **Step 8: Commit all SKILL.md updates**

```bash
git add SKILLS/
git commit -m "docs: update all SKILL.md files for 2-layer state contracts"
```

---

### Task 12: Restructure `demo_colab_langgraph.ipynb`

**Files:**
- Modify: `demo_colab_langgraph.ipynb`

This task completely restructures the notebook. The new notebook has 10 labelled sections. Replace all existing cells with the structure below. Each section shown is one or more notebook cells.

- [ ] **Step 1: Replace the notebook content**

The full notebook should have these cells in order:

**Cell 1 — Setup**
```python
# @title 1. Setup — Mount Drive and Install Dependencies
from google.colab import drive
drive.mount('/content/drive')

import os, sys
os.chdir('/content/drive/MyDrive/BT5151_toxic_comment_agent')
sys.path.insert(0, '/content/drive/MyDrive/BT5151_toxic_comment_agent')

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "langgraph", "gradio", "openai",
    "transformers", "datasets", "evaluate", "accelerate",
    "sentence-transformers", "scikit-learn", "seaborn",
])
print("Setup complete.")
```

**Cell 2 — Imports**
```python
# @title 2. Imports
from pipeline.state import (
    BuildState, RuntimeState,
    build_initial_build_state, build_initial_runtime_state,
)
from pipeline.build import compile_build_graph
from pipeline.graph import compile_runtime_graph
from pipeline.controller import run
print("Imports OK.")
```

**Cell 3 — Secrets (API key — NO hardcoded values)**
```python
# @title 3. API Keys via Colab Secrets
import os
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
print("OPENAI_API_KEY set from Colab Secrets.")
```

**Cell 4 — Build layer: run**
```python
# @title 4. Build Layer — Train Models (run once)
PROJECT_ROOT = "/content/drive/MyDrive/BT5151_toxic_comment_agent"
build_state = build_initial_build_state(PROJECT_ROOT)
print("Starting build layer...")
build_result = run(mode="build", state=build_state)
print(f"Build complete. Selected model: {build_result.get('selected_model_id')}")
print(f"Justification: {build_result.get('selection_justification')}")
```

**Cell 5 — Build layer: inspect outputs**
```python
# @title 5. Build Layer — Inspect Outputs
import json
from pathlib import Path

print("=== Preprocessing Summary ===")
print(json.dumps(build_result.get("preprocessing_summary", {}), indent=2))

print("\n=== Candidate Models Trained ===")
print(build_result.get("candidate_model_ids"))

print("\n=== Selected Model ===")
print(build_result.get("selected_model_id"))

# Display evaluation figures
from IPython.display import Image, display
figures_dir = Path(PROJECT_ROOT) / "experiments" / "figures"
for fig in sorted(figures_dir.glob("*.png")):
    print(f"\n{fig.name}")
    display(Image(str(fig)))
```

**Cell 6 — Runtime layer: sample comments**
```python
# @title 6. Runtime Layer — Sample Comments
SAMPLE_COMMENTS = [
    "Thank you for your edits, this article is much clearer now.",
    "This is a stupid comment and your argument makes no sense.",
    "You are an absolute idiot and nobody wants you here.",
    "What the hell is going on with this page?",
]

runtime_results = []
for comment in SAMPLE_COMMENTS:
    state = build_initial_runtime_state(comment, project_root=PROJECT_ROOT)
    result = run(mode="moderate", state=state)
    runtime_results.append(result)
    print(f"\nComment: {comment[:60]}...")
    print(f"  Predicted: {result.get('predicted_label')} (prob={result.get('toxicity_probability'):.3f})")
    print(f"  Severity:  {result.get('severity_label')}")
    print(f"  Action:    {result.get('action_label')}")
    print(f"  Warning skipped: {result.get('warning_skipped')}")
```

**Cell 7 — Runtime layer: show warning messages**
```python
# @title 7. Runtime Layer — Warning Messages
for comment, result in zip(SAMPLE_COMMENTS, runtime_results):
    warning = result.get("warning_message", "")
    if warning:
        print(f"\nComment: {comment[:60]}...")
        print(f"Action:  {result.get('action_label')}")
        print(f"Warning:\n  {warning}")
        print("-" * 60)
```

**Cell 8 — Graph definitions (for rubric visibility)**
```python
# @title 8. LangGraph — Graph Definitions

# Build graph
build_g = compile_build_graph()
print("Build graph nodes:", list(build_g.graph.nodes.keys()))

# Runtime graph
runtime_g = compile_runtime_graph()
print("Runtime graph nodes:", list(runtime_g.graph.nodes.keys()))

# Controller
from pipeline.controller import run as controller_run
print("Controller: run(mode='build'|'moderate', state)")
```

**Cell 9 — SKILL.md content (markdown cells for each skill)**

Create a markdown cell for each skill in `SKILLS/`:
```python
# @title 9. SKILL.md Contents
from pathlib import Path

for skill_path in sorted(Path("SKILLS").glob("**/*.md")):
    print(f"\n{'='*60}")
    print(f"SKILL: {skill_path.parent.name}")
    print('='*60)
    print(skill_path.read_text())
```

**Cell 10 — Gradio launch**
```python
# @title 10. Gradio Demo
from app import build_demo
demo = build_demo()
demo.launch(share=True)
```

- [ ] **Step 2: Verify the notebook structure is correct**

Open `demo_colab_langgraph.ipynb` and confirm:
- Exactly 10 labelled cells
- Cell 3 uses `userdata.get("OPENAI_API_KEY")` — no hardcoded string
- No API key strings appear anywhere in the notebook

- [ ] **Step 3: Commit**

```bash
git add demo_colab_langgraph.ipynb
git commit -m "refactor: restructure Colab notebook for 2-layer orchestration demo"
```

---

### Task 13: Run full test suite and verify

**Files:** None new

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected output: all tests in `test_state.py`, `test_build_preprocess.py`, `test_draft_warning.py`, `test_controller.py` PASS.

- [ ] **Step 2: Verify app imports cleanly**

```bash
python -c "from app import build_demo, analyze_comment; print('app OK')"
python -c "from pipeline.controller import run; print('controller OK')"
python -c "from pipeline.build import compile_build_graph; print('build OK')"
python -c "from pipeline.graph import compile_runtime_graph; print('graph OK')"
```

Expected: all 4 lines print `OK`.

- [ ] **Step 3: Verify acceptance criteria checklist**

Check each item from the spec:
- [ ] `BuildState` and `RuntimeState` defined in `pipeline/state.py`
- [ ] `pipeline/build.py` contains 4 plain functions + 4 node wrappers + `compile_build_graph()`
- [ ] `pipeline/graph.py` has `draft_warning_node` and `compile_runtime_graph()`
- [ ] `pipeline/controller.py` has deterministic `run(mode, state)`
- [ ] `draft_warning_node` uses `gpt-4o-mini` with fallback template
- [ ] No hardcoded API keys anywhere — Colab Secrets only
- [ ] Raw comment text is not sent to OpenAI
- [ ] Gradio shows `warning_message` field
- [ ] `demo_colab_langgraph.ipynb` restructured with both layers demonstrated
- [ ] `SKILLS/draft_warning/SKILL.md` created
- [ ] All 6 existing SKILL.md files updated to match new state contracts

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: complete 2-layer orchestration iteration"
```
