from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import RuntimeState, build_initial_runtime_state

AgentState = RuntimeState
build_initial_state = build_initial_runtime_state

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency during local setup
    END = "__end__"
    START = "__start__"
    StateGraph = None


def detect_project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).resolve()).resolve()
    candidates = [start, *start.parents]
    markers = [
        "select_model_output.json",
        "models/selected_model_metadata.json",
        "raw_data",
    ]
    for candidate in candidates:
        if sum((candidate / marker).exists() for marker in markers) >= 2:
            return candidate
    raise FileNotFoundError("Could not automatically detect project root.")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def resolve_project_path(state: AgentState, key: str, default_relative: str) -> Path:
    project_root = Path(state.get("project_root") or detect_project_root())
    raw = state.get(key)
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (project_root / path).resolve()
    return (project_root / default_relative).resolve()


def sanitize_text(text: str) -> str:
    clean = str(text).strip()
    if not clean:
        raise ValueError("comment_text cannot be empty.")
    return clean


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def probability_to_confidence(prob: float) -> float:
    return round(abs(prob - 0.5) * 2.0, 4)


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
        "This is a final warning - further violations will result in account suspension. "
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
    action_code  = state.get("action_code", "")
    severity     = state.get("severity_label", "")
    action_label = state.get("action_label", "")
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
        "warning_message":          message,
        "warning_skipped":          False,
        "warning_generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def mean_pool(model_output: Any, attention_mask: Any) -> Any:
    import torch

    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def get_artifact_config(selection_state: dict[str, Any], train_meta: dict[str, Any], state: AgentState) -> dict[str, Any]:
    project_root = Path(state.get("project_root") or detect_project_root())
    model_id = selection_state["selected_model_id"]
    artifact = dict(selection_state.get("artifact", {}))
    metadata_paths = train_meta.get("artifact_paths", {})

    default_configs = {
        "logistic_regression": {
            "type": "tfidf_sklearn",
            "vectorizer_path": metadata_paths.get("tfidf_vectorizer"),
            "model_path": metadata_paths.get("model_lr"),
        },
        "lr": {
            "type": "tfidf_sklearn",
            "vectorizer_path": metadata_paths.get("tfidf_vectorizer"),
            "model_path": metadata_paths.get("model_lr"),
        },
        "linear_svc": {
            "type": "tfidf_sklearn",
            "vectorizer_path": metadata_paths.get("tfidf_vectorizer"),
            "model_path": metadata_paths.get("model_linearsvc"),
        },
        "linearsvc": {
            "type": "tfidf_sklearn",
            "vectorizer_path": metadata_paths.get("tfidf_vectorizer"),
            "model_path": metadata_paths.get("model_linearsvc"),
        },
        "toxigen_bert_lr": {
            "type": "bert_embedding_lr",
            "base_model_name": train_meta.get("toxigen_model_name", "tomh/toxigen_roberta"),
            "model_path": metadata_paths.get("model_toxigen_lr"),
        },
        "toxigen_lr": {
            "type": "bert_embedding_lr",
            "base_model_name": train_meta.get("toxigen_model_name", "tomh/toxigen_roberta"),
            "model_path": metadata_paths.get("model_toxigen_lr"),
        },
        "minilm_ft": {
            "type": "sentence_transformer_finetuned",
            "model_path": metadata_paths.get("minilm_finetuned"),
            "base_model_name": train_meta.get("minilm_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        },
    }

    merged = default_configs.get(model_id, {}).copy()
    merged.update({k: v for k, v in artifact.items() if v is not None})

    for path_key in ("model_path", "vectorizer_path"):
        if merged.get(path_key):
            path = Path(merged[path_key])
            merged[path_key] = str(path if path.is_absolute() else (project_root / path).resolve())

    return merged


def predict_with_tfidf_sklearn(text: str, artifact_cfg: dict[str, Any]) -> tuple[float, float]:
    vectorizer_path = Path(artifact_cfg["vectorizer_path"])
    model_path = Path(artifact_cfg["model_path"])

    with vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)
    with model_path.open("rb") as f:
        model = pickle.load(f)

    features = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        toxic_prob = float(model.predict_proba(features)[0, 1])
        raw_score = toxic_prob
    elif hasattr(model, "decision_function"):
        raw_score = float(model.decision_function(features)[0])
        toxic_prob = float(sigmoid(raw_score))
    else:
        toxic_prob = float(model.predict(features)[0])
        raw_score = toxic_prob
    return toxic_prob, raw_score


def predict_with_embedding_lr(text: str, artifact_cfg: dict[str, Any]) -> tuple[float, float]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(artifact_cfg["base_model_name"])
    encoder = AutoModel.from_pretrained(artifact_cfg["base_model_name"]).to(device)
    encoder.eval()

    with Path(artifact_cfg["model_path"]).open("rb") as f:
        classifier = pickle.load(f)

    enc = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = encoder(**enc)
        embeddings = mean_pool(outputs, enc["attention_mask"]).cpu().numpy()

    toxic_prob = float(classifier.predict_proba(embeddings)[0, 1])
    return toxic_prob, toxic_prob


def predict_with_finetuned_transformer(text: str, artifact_cfg: dict[str, Any]) -> tuple[float, float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = artifact_cfg["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    enc = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(0).detach().cpu()

    if getattr(logits, "numel", lambda: 1)() == 1:
        raw_score = float(logits.item())
        toxic_prob = float(sigmoid(raw_score))
    else:
        probs = logits.softmax(dim=-1).numpy()
        toxic_prob = float(probs[-1])
        raw_score = float(logits[-1].item())
    return toxic_prob, raw_score


def run_inference_node(state: AgentState) -> AgentState:
    comment_text = sanitize_text(state["comment_text"])
    selection_path = resolve_project_path(state, "select_model_output_path", "select_model_output.json")
    train_metadata_path = resolve_project_path(state, "train_metadata_path", "models/selected_model_metadata.json")
    output_path = resolve_project_path(state, "run_inference_output_path", "run_inference_output.json")

    selection_state = load_json(selection_path)
    train_meta = load_json(train_metadata_path)
    artifact_cfg = get_artifact_config(selection_state, train_meta, state)
    model_type = artifact_cfg["type"]
    threshold = float(selection_state.get("inference_threshold", train_meta.get("threshold", 0.5)))

    if model_type == "tfidf_sklearn":
        toxic_prob, raw_score = predict_with_tfidf_sklearn(comment_text, artifact_cfg)
    elif model_type == "bert_embedding_lr":
        toxic_prob, raw_score = predict_with_embedding_lr(comment_text, artifact_cfg)
    elif model_type == "sentence_transformer_finetuned":
        toxic_prob, raw_score = predict_with_finetuned_transformer(comment_text, artifact_cfg)
    else:
        raise ValueError(f"Unsupported artifact type: {model_type}")

    is_toxic = bool(toxic_prob >= threshold)
    payload: AgentState = {
        "comment_text": comment_text,
        "selected_model_id": selection_state["selected_model_id"],
        "selected_model_label": selection_state.get("selected_model_label", ""),
        "model_type": model_type,
        "threshold_used": threshold,
        "predicted_label": "toxic" if is_toxic else "non-toxic",
        "predicted_class_id": int(is_toxic),
        "is_toxic": is_toxic,
        "toxicity_probability": round(float(toxic_prob), 4),
        "non_toxic_probability": round(float(1.0 - toxic_prob), 4),
        "confidence": probability_to_confidence(toxic_prob),
        "raw_score": round(float(raw_score), 4),
        "selection_justification": selection_state.get("selection_justification", ""),
        "bias_assessment": selection_state.get("bias_assessment", ""),
        "source_artifact": artifact_cfg,
        "inference_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(output_path, payload)
    return payload


def assess_severity_node(state: AgentState) -> AgentState:
    toxicity_probability = float(state["toxicity_probability"])
    confidence = float(state["confidence"])
    is_toxic = bool(state["is_toxic"])

    if not is_toxic:
        if toxicity_probability < 0.2:
            severity_label = "clean"
            severity_rank = 0
            escalation_required = False
            review_priority = "none"
            explanation = "The comment is well below the toxic threshold and does not show a meaningful moderation risk."
        else:
            severity_label = "borderline"
            severity_rank = 1
            escalation_required = False
            review_priority = "low"
            explanation = "The comment is currently classified as non-toxic, but it is close enough to the decision boundary that downstream review logic should treat it as borderline."
    else:
        if toxicity_probability >= 0.9 and confidence >= 0.8:
            severity_label = "critical"
            severity_rank = 4
            escalation_required = True
            review_priority = "urgent"
            explanation = "The comment is classified as toxic with both very high probability and high confidence, suggesting a severe moderation risk that should be escalated immediately."
        elif toxicity_probability >= 0.75:
            severity_label = "high"
            severity_rank = 3
            escalation_required = True
            review_priority = "high"
            explanation = "The comment is classified as toxic with strong probability, indicating a material risk of abuse or harassment and a likely need for moderator intervention."
        elif toxicity_probability >= 0.6:
            severity_label = "medium"
            severity_rank = 2
            escalation_required = False
            review_priority = "medium"
            explanation = "The comment is classified as toxic, but the model signal is moderate rather than overwhelming. It likely warrants review or a softer intervention."
        else:
            severity_label = "low"
            severity_rank = 1
            escalation_required = False
            review_priority = "low"
            explanation = "The comment crossed the toxicity threshold, but only weakly. It should be monitored or reviewed before a strong moderation action is taken."

    summary_for_moderator = (
        f"Predicted {state['predicted_label']} with toxicity probability {toxicity_probability:.4f} "
        f"and confidence {confidence:.4f}. Assigned severity: {severity_label}."
    )

    payload: AgentState = {
        "severity_label": severity_label,
        "severity_rank": severity_rank,
        "review_priority": review_priority,
        "escalation_required": escalation_required,
        "severity_explanation": explanation,
        "summary_for_moderator": summary_for_moderator,
        "severity_assessed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = resolve_project_path(state, "assess_severity_output_path", "assess_severity_output.json")
    dump_json(output_path, {**state, **payload})
    return payload


def recommend_moderation_action_node(state: AgentState) -> AgentState:
    severity_label = state["severity_label"]
    toxicity_probability = float(state["toxicity_probability"])
    confidence = float(state["confidence"])

    if severity_label == "clean":
        action_code = "allow"
        action_label = "Allow Comment"
        action_priority = "none"
        human_review_required = False
        user_notification = "none"
        moderator_rationale = "The comment is comfortably below the toxic threshold and does not need intervention."
        business_message = "No action is recommended. The comment can remain visible."
    elif severity_label == "borderline":
        action_code = "allow_with_monitoring"
        action_label = "Allow but Monitor"
        action_priority = "low"
        human_review_required = False
        user_notification = "none"
        moderator_rationale = "The comment is currently non-toxic but sits close to the decision boundary, so lightweight monitoring is appropriate."
        business_message = "Allow the comment to remain visible, but keep it in a low-priority monitoring queue."
    elif severity_label == "low":
        action_code = "soft_warn"
        action_label = "Soft Warning"
        action_priority = "low"
        human_review_required = True
        user_notification = "gentle_warning"
        moderator_rationale = "The content crossed the toxicity threshold weakly, so a softer response is safer than immediate removal."
        business_message = "Queue the comment for review and consider sending the user a soft civility warning."
    elif severity_label == "medium":
        action_code = "review_and_warn"
        action_label = "Review and Warn"
        action_priority = "medium"
        human_review_required = True
        user_notification = "warning"
        moderator_rationale = "The model indicates meaningful toxicity, but the evidence is not strong enough to justify automatic removal without review."
        business_message = "Send the case to a moderator review queue and issue a warning if the comment violates policy."
    elif severity_label == "high":
        action_code = "hide_and_review"
        action_label = "Hide and Review"
        action_priority = "high"
        human_review_required = True
        user_notification = "warning"
        moderator_rationale = "The comment has a strong toxic signal and should be hidden quickly while waiting for moderator confirmation."
        business_message = "Temporarily hide the comment and send it for high-priority moderator review."
    else:
        action_code = "remove_and_escalate"
        action_label = "Remove and Escalate"
        action_priority = "urgent"
        human_review_required = True
        user_notification = "final_warning"
        moderator_rationale = "The comment shows a very strong toxic signal and should be removed immediately with escalation to urgent moderator attention."
        business_message = "Remove the comment immediately and escalate the case to the urgent moderation queue."

    payload: AgentState = {
        "action_code": action_code,
        "action_label": action_label,
        "action_priority": action_priority,
        "human_review_required": human_review_required,
        "user_notification": user_notification,
        "moderator_rationale": moderator_rationale,
        "business_message": business_message,
        "final_recommendation": (
            f"Recommended action: {action_label}. Severity is {severity_label} "
            f"with toxicity probability {toxicity_probability:.4f} and confidence {confidence:.4f}."
        ),
        "ui_explanation": f"This comment was assessed as {severity_label}. {business_message}",
        "action_recommended_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = resolve_project_path(state, "moderation_action_output_path", "moderation_action_output.json")
    dump_json(output_path, {**state, **payload})
    return payload


def build_graph() -> Any:
    if StateGraph is None:
        raise ImportError("langgraph is not installed. Install it before compiling the graph.")

    graph = StateGraph(AgentState)
    graph.add_node("run-inference", run_inference_node)
    graph.add_node("assess-severity", assess_severity_node)
    graph.add_node("recommend-moderation-action", recommend_moderation_action_node)
    graph.add_edge(START, "run-inference")
    graph.add_edge("run-inference", "assess-severity")
    graph.add_edge("assess-severity", "recommend-moderation-action")
    graph.add_edge("recommend-moderation-action", END)
    return graph.compile()


def run_pipeline(comment_text: str, initial_state: AgentState | None = None) -> AgentState:
    state = build_initial_state(comment_text, initial_state.get("project_root") if initial_state else None)
    if initial_state:
        state.update(initial_state)

    if StateGraph is not None:
        app = build_graph()
        return app.invoke(state)

    current = dict(state)
    current.update(run_inference_node(current))
    current.update(assess_severity_node(current))
    current.update(recommend_moderation_action_node(current))
    return current
