# Components
Gradio supplies Image, Button, Textbox, Markdown, HTML, Row and Blocks. The local risk badge and metrics formatter are in app.py.
```python
"""Minimal Gradio interface for the hallucination-aware inspection pipeline."""

from __future__ import annotations

import argparse
import html
from collections.abc import Sequence
from typing import Any

import gradio as gr
from PIL import Image

import pipeline


def risk_badge(risk: str, detected: bool) -> str:
    """Return an accessible, self-contained risk indicator."""
    styles = {
        "Low": ("#14532d", "#dcfce7"),
        "Medium": ("#854d0e", "#fef9c3"),
        "High": ("#991b1b", "#fee2e2"),
        "N/A": ("#334155", "#e2e8f0"),
    }
    foreground, background = styles.get(risk, styles["N/A"])
    symbols = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "N/A": "⚪"}
    label = risk if detected else "N/A — no detection"
    symbol = symbols.get(risk, symbols["N/A"])
    return (
        '<div role="status" aria-label="Hallucination risk" '
        f'style="padding:12px 16px;border-radius:10px;font-weight:700;'
        f'color:{foreground};background:{background};">'
        f"{symbol} Hallucination risk: {html.escape(label)}</div>"
    )


def metrics_markdown(result: dict[str, Any]) -> str:
    """Format auditable detector and gate metrics for the interface."""
    thresholds = result["thresholds"]
    confidences = ", ".join(f"{value:.4f}" for value in result["tta_confidences"])
    summary = f"""| Metric | Value |
|---|---:|
| Detection | `{"Yes" if result["detected"] else "No"}` |
| Mean TTA confidence | `{result["confidence"]:.2%}` |
| Raw confidence | `{result["raw_confidence"]:.2%}` |
| TTA variance (population stddev) | `{result["variance"]:.4f}` |
| Hallucination risk | `{result["hallucination_risk"]}` |
| Confidence threshold | `{thresholds["confidence"]:.2f}` |
| Standard-deviation threshold | `{thresholds["variance"]:.2f}` |
| Processing time | `{result["processing_time"]:.3f} s` |
| VLM called | `{result["vlm_called"]}` |
| VLM succeeded | `{result["vlm_succeeded"]}` |

**Five deterministic confidence passes:** `{confidences}`

**Gate evidence:** `{result["gate_log"]}` — {result["gate_reason"]}
"""
    if result.get("detections"):
        summary += "\n**Per-region results** (image risk is the highest region risk):\n\n"
        summary += "| Region | Mean TTA confidence | TTA stddev | Risk | Explanation surfaced |\n|---|---:|---:|---|---|\n"
        for i, region in enumerate(result["detections"], 1):
            summary += f"| {i} | {region['confidence']:.2%} | {region['variance']:.4f} | {region['hallucination_risk']} | {region['vlm_succeeded']} |\n"
    return summary


def inspect_for_ui(image: Image.Image | None) -> tuple[Image.Image, str, str, str]:
    """Run the unchanged public pipeline entrypoint and adapt its result for Gradio."""
    if image is None:
        raise gr.Error("Upload an image before running inspection.")
    try:
        result = pipeline.inspect(image)
    except Exception as exc:
        raise gr.Error(f"Inspection failed: {type(exc).__name__}: {exc}") from exc
    return (
        result["annotated_image"],
        result["explanation"],
        risk_badge(result["hallucination_risk"], result["detected"]),
        metrics_markdown(result),
    )


def build_demo() -> gr.Blocks:
    """Construct the interface without launching a server."""
    with gr.Blocks(title="Hallucination-Aware Crack Inspection") as demo:
        gr.Markdown(
            """# Hallucination-Aware Crack Inspection

Detector → TTA uncertainty gate → grounded VLM explanation when allowed. Upload a surface image; uncertain cases are withheld for human review. This demonstration is not a safety certification or substitute for professional inspection.
"""
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input image")
            output_image = gr.Image(
                type="pil", label="Annotated detection", interactive=False
            )
        inspect_button = gr.Button("Run inspection", variant="primary")
        risk = gr.HTML(label="Risk decision")
        explanation = gr.Textbox(label="Grounded result", lines=3, interactive=False)
        metrics = gr.Markdown(label="Metrics and gate evidence")
        inspect_button.click(
            fn=inspect_for_ui,
            inputs=input_image,
            outputs=[output_image, explanation, risk, metrics],
            api_name="inspect",
        )
        gr.Markdown(
            "Numbered red boxes show all original-pass detections. Each region is gated independently. `variance` in machine-readable results contains population standard deviation, not mathematical variance."
        )
    return demo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```
