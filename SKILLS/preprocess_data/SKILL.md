---

name: preprocess-data
description: Clean raw text data, construct binary labels for content policy violation, and prepare train-validation splits for downstream modeling.
----------------------------------------------------------------------------------------------------------------------------------------------------

## When to use

This skill is used at the beginning of the pipeline to prepare raw text data for machine learning. It ensures that the dataset is cleaned, structured, and ready for model training and evaluation.

---

## How to execute

1. Load the raw dataset (`train.csv`) containing comment text and toxicity labels.
2. Perform initial exploratory data analysis (EDA), including:

   * Inspecting dataset shape and columns
   * Checking for missing values
   * Examining label distribution to identify class imbalance
3. Clean the text data by:

   * Converting all text to lowercase
   * Removing newline (`\n`) and carriage return (`\r`) characters
   * Normalizing whitespace
4. Construct a binary target variable `toxic_label`:

   * Label = 1 if any of the six toxicity categories is present
   * Label = 0 otherwise
5. Split the dataset into training and validation sets using stratified sampling:

   * Maintain consistent class distribution between splits
6. Output the processed datasets for downstream model training and evaluation.

---

## Inputs from Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `project_root` | `str` | Absolute path to project root |
| `raw_train_path` | `str` | Path to `raw_data/train.csv` |
| `raw_test_path` | `str` | Path to `raw_data/test.csv` |

---

## Outputs to Agent State (BuildState)

| Key | Type | Description |
|---|---|---|
| `train_processed_path` | `str` | Path to `experiments/processed_data/train_set.csv` |
| `val_processed_path` | `str` | Path to `experiments/processed_data/val_set.csv` |
| `test_processed_path` | `str` | Path to `experiments/processed_data/test_set.csv` |
| `preprocessing_summary` | `dict` | Row counts, toxic rates per split |

---

## Output format

The output datasets contain the following columns:

* `id`
* `comment_text_clean`
* `toxic_label`

These datasets are saved as CSV files:

* `experiments/processed_data/train_set.csv`
* `experiments/processed_data/val_set.csv`
* `experiments/processed_data/test_set.csv`

---

## Notes

* The preprocessing step uses **light text cleaning** to preserve important linguistic signals such as punctuation and offensive language, which are critical for toxicity detection.
* The dataset exhibits **class imbalance**, with non-toxic comments significantly outnumbering toxic ones. Stratified splitting is applied to ensure fair model evaluation.
* A binary target (`toxic_label`) is constructed to simplify the classification task while maintaining alignment with the business objective of detecting policy violations.
* The processed outputs are designed to support both modeling pipelines:

  * TF-IDF + Gradient Boosting
  * Word Embeddings + Logistic Regression
