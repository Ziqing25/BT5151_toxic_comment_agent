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
4. Construct a binary target variable `any_violation`:

   * Label = 1 if any of the six toxicity categories is present
   * Label = 0 otherwise
5. Split the dataset into training and validation sets using stratified sampling:

   * Maintain consistent class distribution between splits
6. Output the processed datasets for downstream model training and evaluation.

---

## Inputs from agent state

* `train_df`: Raw training dataset
* `comment_text`: Text column containing user comments
* `label_cols`: List of toxicity labels

  * toxic
  * severe_toxic
  * obscene
  * threat
  * insult
  * identity_hate

---

## Outputs to agent state

* `train_split`: Processed training dataset
* `val_split`: Processed validation dataset
* `clean_text`: Cleaned version of input text
* `any_violation`: Binary classification target

---

## Output format

The output datasets contain the following columns:

* `id`
* `comment_text`
* `clean_text`
* original label columns (six toxicity categories)
* `any_violation`

These datasets are saved as CSV files:

* `train_processed.csv`
* `val_processed.csv`

---

## Notes

* The preprocessing step uses **light text cleaning** to preserve important linguistic signals such as punctuation and offensive language, which are critical for toxicity detection.
* The dataset exhibits **class imbalance**, with non-toxic comments significantly outnumbering toxic ones. Stratified splitting is applied to ensure fair model evaluation.
* A binary target (`any_violation`) is constructed to simplify the classification task while maintaining alignment with the business objective of detecting policy violations.
* The processed outputs are designed to support both modeling pipelines:

  * TF-IDF + Gradient Boosting
  * Word Embeddings + Logistic Regression
