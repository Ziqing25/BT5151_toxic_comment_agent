from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline.graph import detect_project_root, run_pipeline


def analyze_comment(comment_text: str) -> tuple[str, str, str, str, str, str, bool, dict[str, Any]]:
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
        result,
    )


def build_demo() -> Any:
    import gradio as gr

    with gr.Blocks(title="BT5151 Toxic Comment Moderation Agent") as demo:
        gr.Markdown(
            """
            # BT5151 Toxic Comment Moderation Agent

            This demo runs the LangGraph moderation pipeline:
            `run-inference -> assess-severity -> recommend-moderation-action`
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
                action_label = gr.Textbox(label="Recommended Action", interactive=False)
                business_message = gr.Textbox(label="Business Message", interactive=False, lines=3)
                severity_label = gr.Textbox(label="Severity", interactive=False)
                review_priority = gr.Textbox(label="Review Priority", interactive=False)
                user_notification = gr.Textbox(label="User Notification", interactive=False)
                ui_explanation = gr.Textbox(label="UI Explanation", interactive=False, lines=4)
                human_review_required = gr.Checkbox(label="Human Review Required", interactive=False)

        raw_output = gr.JSON(label="Full Pipeline Output")

        analyze_button.click(
            fn=analyze_comment,
            inputs=[comment_input],
            outputs=[
                action_label,
                business_message,
                severity_label,
                review_priority,
                user_notification,
                ui_explanation,
                human_review_required,
                raw_output,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BT5151 toxic comment moderation Gradio app.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--server-port", type=int, default=7860, help="Port to bind the Gradio app to.")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server host for the Gradio app.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = detect_project_root(Path(__file__).resolve())
    print(f"Detected project root: {project_root}")
    print("Launching Gradio app...")

    try:
        demo = build_demo()
    except Exception as exc:
        raise RuntimeError(
            "Failed to build the Gradio app. "
            "Please check that gradio and the ML dependencies are installed correctly."
        ) from exc

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
