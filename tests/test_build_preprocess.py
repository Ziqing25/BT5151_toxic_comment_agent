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


def test_transformer_batch_sizes_are_128():
    from pipeline import build

    assert build.TOXIGEN_BATCH_SIZE == 128
    assert build.MINILM_BATCH_SIZE == 128
