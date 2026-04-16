# Content Policy Orchestration Iteration — Design Spec

**Date:** 2026-04-16
**Project:** BT5151 Toxic Comment Detection Agent
**Scope:** Upgrade from single-layer runtime-only pipeline to 2-layer orchestration (offline build + online runtime), add `draft-warning` LLM node, update Gradio, restructure Colab notebook.

---

## 1. Context

The current system has:
- A working runtime graph (`run-inference → assess-severity → recommend-moderation-action`) in `pipeline/graph.py`
- A single `AgentState` TypedDict covering only runtime fields in `pipeline/state.py`
- Build layer logic (preprocessing, training, evaluation, model selection) living only as Jupyter notebook cells in `pipeline/nodes/`
- No `draft-warning` node
- Gradio UI that shows moderation result but no user-facing warning message

This iteration formalises the 2-layer architecture, adds the `draft-warning` node, and restructures the Colab notebook for end-to-end submission.

---

## 2. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM for `draft-warning` | OpenAI `gpt-4o-mini` | As specified in PRD; lower cost, stable for demo |
| Build layer implementation | Full extraction — plain Python functions | LangGraph nodes are thin wrappers; Colab cells call same functions |
| State design | Two separate TypedDicts (`BuildState`, `RuntimeState`) | Clean layer separation; controller never merges them |
| Module layout | Approach B — `build.py` + refactored `graph.py` + `controller.py` | Each file has one clear job; minimal blast radius |
| Controller type | Deterministic `run(mode, state)` | No LLM routing needed; keeps the system stable and explainable |
| API key handling | Google Colab Secrets (`userdata.get`) | No hardcoded keys anywhere in submission |

---

## 3. File Changes

| File | Change |
|---|---|
| `pipeline/state.py` | Replace `AgentState` with `BuildState` + `RuntimeState` + shared literals |
| `pipeline/build.py` | **New** — plain build functions + `compile_build_graph()` |
| `pipeline/graph.py` | **Refactor** — rename `build_graph()` → `compile_runtime_graph()`, add `draft_warning_node`, update type annotations |
| `pipeline/controller.py` | **New** — `run(mode, state)` deterministic router |
| `app.py` | Add `warning_message` output field to Gradio UI |
| `SKILLS/draft_warning/SKILL.md` | **New** — state contract for the new node |
| `SKILLS/preprocess_data/SKILL.md` | Update state field names to `BuildState` keys |
| `SKILLS/model_training/SKILL.md` | Update state field names to `BuildState` keys |
| `SKILLS/model_evaluation/SKILL.md` | Update state field names to `BuildState` keys |
| `SKILLS/model_selection/SKILL.md` | Update state field names to `BuildState` keys |
| `SKILLS/run_inference/SKILL.md` | Update `AgentState` → `RuntimeState` |
| `SKILLS/assess_severity/SKILL.md` | Update `AgentState` → `RuntimeState` |
| `SKILLS/recommend_moderation_actoin/SKILL.md` | Update `AgentState` → `RuntimeState`; clarify no LLM |
| `demo_colab_langgraph.ipynb` | Restructure cells for end-to-end 2-layer demo |

---

## 4. Execution Flow

### Build layer (offline — run once to produce model artifacts)

```
preprocess-data
  └─▶ train-models
        └─▶ evaluate-models
              └─▶ select-model
                    └─▶ [artifacts persisted to models/]
```

### Runtime layer (online — per-comment moderation)

```
run-inference
  └─▶ assess-severity
        └─▶ recommend-moderation-action
              └─▶ draft-warning  (skipped when action_code in {"allow", "allow_with_monitoring"})
```

### Controller

```python
# pipeline/controller.py
from pipeline.build import compile_build_graph
from pipeline.graph import compile_runtime_graph
from pipeline.state import BuildState, RuntimeState

def run(mode: Literal["build", "moderate"], state: BuildState | RuntimeState):
    if mode == "build":
        return compile_build_graph().invoke(state)
    return compile_runtime_graph().invoke(state)
```

---

## 5. State Design

### `BuildState` (`pipeline/state.py`)

```python
class BuildState(TypedDict, total=False):
    # Inputs
    project_root: str
    raw_train_path: str
    raw_test_path: str

    # preprocess-data outputs
    train_processed_path: str
    val_processed_path: str
    test_processed_path: str
    preprocessing_summary: dict

    # train-models outputs
    train_metadata_path: str
    candidate_model_ids: list[str]

    # evaluate-models outputs
    evaluation_report_path: str
    bias_audit_path: str

    # select-model outputs
    select_model_output_path: str
    selected_model_id: str
    selection_justification: str
```

### `RuntimeState` (`pipeline/state.py`)

All existing `AgentState` fields are preserved (renamed). Three new fields added:

```python
# NEW fields only — all existing inference/severity/action fields unchanged
warning_message: str             # LLM output or template fallback
warning_skipped: bool            # True when action is allow/allow_with_monitoring
warning_generated_at_utc: str
```

Path fields (`project_root`, `select_model_output_path`, `train_metadata_path`) move from the old `AgentState` into `RuntimeState` unchanged.

### Helper constructors (`pipeline/state.py`)

`build_initial_state()` splits into two functions:

```python
def build_initial_build_state(project_root: str | Path, ...) -> BuildState: ...
def build_initial_runtime_state(comment_text: str, project_root: str | Path | None = None) -> RuntimeState: ...
```

The old `build_initial_state()` is removed.

---

## 6. Build Layer (`pipeline/build.py`)

### Plain functions

| Function | Reads from state | Writes to state |
|---|---|---|
| `load_and_preprocess_data(state)` | `raw_train_path`, `raw_test_path`, `project_root` | `train_processed_path`, `val_processed_path`, `test_processed_path`, `preprocessing_summary` |
| `train_candidate_models(state)` | `train_processed_path`, `val_processed_path` | `train_metadata_path`, `candidate_model_ids` |
| `evaluate_candidate_models(state)` | `test_processed_path`, `train_metadata_path` | `evaluation_report_path`, `bias_audit_path` |
| `select_best_model(state)` | `evaluation_report_path`, `bias_audit_path`, `train_metadata_path` | `select_model_output_path`, `selected_model_id`, `selection_justification` |

Logic is extracted from `pipeline/nodes/01–04` notebooks, not rewritten from scratch.

### LangGraph node wrappers (thin)

```python
def preprocess_data_node(state: BuildState) -> BuildState:
    return load_and_preprocess_data(state)

def train_models_node(state: BuildState) -> BuildState:
    return train_candidate_models(state)

def evaluate_models_node(state: BuildState) -> BuildState:
    return evaluate_candidate_models(state)

def select_model_node(state: BuildState) -> BuildState:
    return select_best_model(state)
```

### `compile_build_graph()`

```python
def compile_build_graph():
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

---

## 7. Runtime Layer (`pipeline/graph.py`)

### Existing nodes

`run_inference_node`, `assess_severity_node`, `recommend_moderation_action_node` — logic unchanged; type annotations updated to `RuntimeState`.

`run_pipeline()` updated to call `compile_runtime_graph()` instead of the old `build_graph()`.

### New: `draft_warning_node`

```python
def draft_warning_node(state: RuntimeState) -> RuntimeState:
    action_code = state["action_code"]

    # Skip for non-actionable decisions
    if action_code in ("allow", "allow_with_monitoring"):
        return {
            "warning_message": "",
            "warning_skipped": True,
            "warning_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # Try OpenAI; fall back to template on any failure
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_get_openai_key())
        prompt = _build_warning_prompt(state)
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
        "warning_message": message,
        "warning_skipped": False,
        "warning_generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
```

**`_get_openai_key()`** — reads `os.environ["OPENAI_API_KEY"]`; raises `RuntimeError` with a clear message if missing.

**`_build_warning_prompt(state)`** — uses `action_code`, `severity_label`, `action_label` only. Raw comment text is **not** sent to OpenAI (privacy).

**`_fallback_warning(state)`** — dict keyed by `action_code` returning pre-written template strings. Pipeline never breaks if OpenAI is unavailable.

**Skip logic** — implemented inside the node (not as a conditional edge) for simplicity.

### `compile_runtime_graph()`

```python
def compile_runtime_graph():
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

---

## 8. Gradio (`app.py`)

`analyze_comment()` return tuple gains `warning_message` as the last field before the raw JSON output.

New UI field added after `human_review_required`:

```python
warning_message_box = gr.Textbox(
    label="Warning Message to User",
    interactive=False,
    lines=4,
)
```

Visible for all actions; empty string displayed when warning is skipped (clean, non-toxic comment).

The pipeline description in the Gradio header updates to reflect the 4-node runtime chain.

---

## 9. Colab Notebook Structure (`demo_colab_langgraph.ipynb`)

| Cell # | Label | Content |
|---|---|---|
| 1 | **Setup** | Mount Drive, `os.chdir`, `!pip install` |
| 2 | **Imports** | `sys.path` insert, import from `pipeline.*` |
| 3 | **Secrets** | `os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")` |
| 4 | **Build layer — run** | `build_initial_build_state(...)` → `run(mode="build", state=...)` |
| 5 | **Build layer — inspect** | Print `selected_model_id`, display evaluation figures |
| 6 | **Runtime layer — sample comments** | Loop over 4 examples, `run(mode="moderate", state=...)` |
| 7 | **Runtime layer — warning output** | Print `warning_message` for toxic examples |
| 8 | **Graph definitions** | Show both subgraph structures (rubric visibility) |
| 9 | **SKILL.md content** | Markdown cells for each skill |
| 10 | **Gradio launch** | `build_demo().launch(share=True)` |

**API key rule:** `OPENAI_API_KEY` appears only in cell 3 via `userdata.get()`. No key string appears anywhere else in the notebook.

---

## 10. SKILL.md Updates

### New: `SKILLS/draft_warning/SKILL.md`

State contract:

| Field | Direction | Description |
|---|---|---|
| `action_code` | reads | Determines skip or generate |
| `severity_label` | reads | Informs warning tone |
| `action_label` | reads | Included in prompt |
| `warning_message` | writes | LLM output or template fallback |
| `warning_skipped` | writes | `True` when skipped |
| `warning_generated_at_utc` | writes | ISO timestamp |

Includes: when to use, how to execute, prompt strategy, fallback behaviour, OpenAI model spec.

### Existing SKILL.md updates

- `preprocess_data`, `model_training`, `model_evaluation`, `model_selection`: state field names → `BuildState` keys
- `run_inference`, `assess_severity`, `recommend_moderation_actoin`: `AgentState` → `RuntimeState`
- `recommend_moderation_actoin`: add explicit note that this node does not call any LLM

---

## 11. Out of Scope (this iteration)

- `update-dashboard-log` node (PRD marks as "if time permits" — excluded)
- Multi-label classification (remains binary)
- Production backend or persistent database
- LLM-based supervisor or router

---

## 12. Acceptance Criteria

- [ ] `BuildState` and `RuntimeState` defined in `pipeline/state.py`
- [ ] `pipeline/build.py` contains 4 plain functions + 4 node wrappers + `compile_build_graph()`
- [ ] `pipeline/graph.py` has `draft_warning_node` and `compile_runtime_graph()`
- [ ] `pipeline/controller.py` has deterministic `run(mode, state)`
- [ ] `draft_warning_node` uses `gpt-4o-mini` with fallback template
- [ ] No hardcoded API keys anywhere — Colab Secrets only
- [ ] Raw comment text is not sent to OpenAI
- [ ] Gradio shows `warning_message` field
- [ ] `demo_colab_langgraph.ipynb` runs end-to-end in Colab with both layers demonstrated
- [ ] `SKILLS/draft_warning/SKILL.md` created
- [ ] All 6 existing SKILL.md files updated to match new state contracts
