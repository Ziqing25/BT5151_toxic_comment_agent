# BT5151 Toxic Comment Agent

This repository contains our BT5151 group project on toxic comment moderation. The project is organised as a multi-stage machine learning and agent workflow:

`preprocess-data -> train-models -> evaluate-models -> select-model -> run-inference -> assess-severity -> recommend-moderation-action`

The upstream machine learning stages are documented and demonstrated in notebooks and `SKILL.md` files. The deployed moderation flow is implemented as a LangGraph-style online pipeline for inference and business-facing decision support.

## What This Project Does

- Trains and compares multiple toxic comment classification models
- Selects one final model using a justified decision process
- Runs inference on new comments
- Translates model output into severity and moderation actions
- Exposes the final business-facing result through Gradio

## Main Files and Folders

### Data

- `raw_data/`
  Contains the original toxic comment dataset files used for preprocessing and model development.

- `processed_data/`
  Contains processed train / validation / test splits used by the ML pipeline.

### Models and Saved Artefacts

- `models/`
  Stores trained model artefacts used by the online moderation pipeline.

- `models/minilm_finetuned/`
  The final fine-tuned MiniLM classifier directory.

- `models/tfidf_vectorizer.pkl`
  Saved TF-IDF vectorizer for baseline models.

- `models/model_lr.pkl`
  Saved TF-IDF + Logistic Regression model.

- `models/model_linearsvc.pkl`
  Saved TF-IDF + LinearSVC model.

- `models/model_toxigen_lr.pkl`
  Saved ToxiGen embedding + Logistic Regression model.

- `models/selected_model_metadata.json`
  Training-stage metadata describing validation/test metrics, hyperparameters, and artefact paths.

- `select_model_output.json`
  Output of the `select-model` stage. This is the bridge between the offline ML pipeline and the online inference pipeline.

### Notebooks

- `experiments/`
  Working experiment notebooks, plots, and supporting development artefacts.

- `pipeline/nodes/01_data_preprocessing.ipynb`
  Demonstrates the `preprocess-data` stage.

- `pipeline/nodes/02_model_training.ipynb`
  Demonstrates the `train-models` stage.

- `pipeline/nodes/03_model_evaluation.ipynb`
  Demonstrates the `evaluate-models` stage.

- `pipeline/nodes/04_select_model.ipynb`
  Demonstrates the `select-model` stage.

- `pipeline/nodes/05_run_inference.ipynb`
  Demonstrates the `run-inference` stage.

- `pipeline/nodes/06_assess_severity.ipynb`
  Demonstrates the `assess-severity` downstream skill.

- `pipeline/nodes/07_recommend_moderation_action.ipynb`
  Demonstrates the `recommend-moderation-action` downstream skill.

- `demo_colab_langgraph.ipynb`
  Colab-oriented demo notebook for the online moderation pipeline.

### Agent Skills

- `SKILLS/`
  Contains the `SKILL.md` files that describe each stage as an inspectable agent skill.

- `SKILLS/preprocess_data.md`
  Skill description for data preprocessing.

- `SKILLS/model_evaluation.md`
  Skill description for model evaluation.

- `SKILLS/model_selection.md`
  Skill description for model selection.

- `SKILLS/run_inference.md`
  Skill description for inference on new comments.

- `SKILLS/assess_severity.md`
  Skill description for turning raw model output into a severity level.

- `SKILLS/recommend_moderation_action.md`
  Skill description for producing a business-facing moderation action.

### Online Pipeline Code

- `pipeline/state.py`
  Defines the shared online agent state used by the deployed moderation pipeline.

- `pipeline/graph.py`
  Implements the online moderation graph:
  `run-inference -> assess-severity -> recommend-moderation-action`

- `app.py`
  Minimal Gradio application that calls the pipeline and displays the final moderation result.

## Why There Are JSON Files

The JSON files act as handoff artefacts between stages.

Examples:

- `select_model_output.json`
  Passes the final selected model and threshold from `select-model` to `run-inference`.

- `run_inference_output.json`
  Passes raw prediction output from `run-inference` to `assess-severity`.

- `assess_severity_output.json`
  Passes severity results from `assess-severity` to `recommend-moderation-action`.

- `moderation_action_output.json`
  Stores the final business-facing moderation recommendation for UI display or debugging.

These files help us:

- keep stages inspectable
- debug each stage independently
- connect offline and online parts of the pipeline
- demonstrate intermediate outputs in notebooks and presentations

## Recommended Reading Order

If you are new to the repository, the easiest order is:

1. Read this `README.md`
2. Open `pipeline/nodes/04_select_model.ipynb`
3. Open `pipeline/nodes/05_run_inference.ipynb`
4. Open `pipeline/nodes/06_assess_severity.ipynb`
5. Open `pipeline/nodes/07_recommend_moderation_action.ipynb`
6. Review `pipeline/state.py` and `pipeline/graph.py`
7. Run `app.py` or `demo_colab_langgraph.ipynb`

## How To Run the Demo Locally

Start the Gradio app:

```bash
python3 app.py --share
```

If the local environment has dependency conflicts, use the Colab notebook instead:

```text
demo_colab_langgraph.ipynb
```

## How To Run the Demo in Colab

Use:

```text
demo_colab_langgraph.ipynb
```

This notebook is intended to be the easiest demonstration entry point for the online moderation pipeline in Google Colab.

## Current Architecture

There are two conceptual layers in this repository:

### 1. Offline ML pipeline

This includes:

- `preprocess-data`
- `train-models`
- `evaluate-models`
- `select-model`

These stages are mainly demonstrated through notebooks and saved artefacts.

### 2. Online moderation pipeline

This includes:

- `run-inference`
- `assess-severity`
- `recommend-moderation-action`

These stages are used at demo time to process a new user comment and produce a business-facing moderation result.

## Notes on API Keys

The current core moderation workflow does not require an external API key.

That is intentional:

- the classification model is local
- severity is rule-based
- moderation action is rule-based

This keeps the core system reproducible and aligned with the project requirement that the main ML logic should not be a black-box API call.

If needed in the future, an API-based explanation stage could be added as an optional enhancement for generating natural-language explanations, but it is not required for the current pipeline.

## Common Outputs You Will See

For a new comment, the online pipeline produces:

- predicted toxicity label
- toxic probability
- confidence
- severity level
- review priority
- moderation action
- business-facing explanation

## Important Submission Reminder

For the coursework submission, make sure the final package includes:

- runnable notebooks
- one `SKILL.md` per required stage
- trained model artefacts or reproducible access to them
- a working demo notebook for Colab
- the technical report and presentation materials

## Short Summary

If you only remember three things about this repository:

1. The notebooks explain each stage
2. The JSON files connect the stages
3. `app.py` and `demo_colab_langgraph.ipynb` are the main demo entry points
