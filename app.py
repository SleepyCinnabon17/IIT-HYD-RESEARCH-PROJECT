"""Minimal Gradio interface for the hallucination-aware inspection pipeline."""

from __future__ import annotations

import argparse
import html
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

import pipeline

REPOSITORY = "https://github.com/SleepyCinnabon17/IIT-HYD-RESEARCH-PROJECT"
WAITING_STATUS = '<div class="risk-state risk-ready" role="status"><span class="eyebrow">Inspection status</span><strong>Awaiting image</strong><span>Upload a surface photograph to begin.</span></div>'


def risk_badge(risk: str, detected: bool) -> str:
    """Return an accessible, self-contained risk indicator."""
    label = risk if detected else "N/A — no detection"
    state = "review" if risk in {"High", "Medium"} and detected else "ready"
    detail = "At least one region needs human review." if state == "review" else "Read each region's observation below. This is not a safety assessment."
    return (
        f'<div class="risk-state risk-{state}" role="status" aria-label="Hallucination risk">'
        '<span class="eyebrow">Inspection status</span>'
        f'<strong>Hallucination risk: {html.escape(label)}</strong><span>{detail}</span></div>'
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
    theme = gr.themes.Base(
        primary_hue="red", secondary_hue="neutral", neutral_hue="neutral",
        font=["Arial", "Helvetica", "sans-serif"],
        font_mono=["Consolas", "Courier New", "monospace"],
        radius_size=gr.themes.sizes.radius_none,
    )
    with gr.Blocks(
        title="Crack Inspection | Evidence Before Explanation", theme=theme,
        css=Path(__file__).with_name("ui.css").read_text(encoding="utf-8"),
        analytics_enabled=False, delete_cache=(3600, 3600),
    ) as demo:
        gr.HTML(f'''<nav class="masthead" aria-label="Project navigation">
<div><span class="eyebrow">Visual inspection / Research tool</span><span class="project-name">Hallucination-aware crack inspection</span></div>
<div class="nav-links"><a href="{REPOSITORY}" target="_blank" rel="noopener noreferrer">Source</a><a href="{REPOSITORY}/blob/main/REPORT.md" target="_blank" rel="noopener noreferrer">Evaluation report</a></div></nav>
<header class="intro"><p class="eyebrow red-ink">Inspect the surface. Question the result.</p>
<h1>Evidence before<br><span>explanation.</span></h1>
<p class="intro-copy">Locate visible cracks, check how consistently they are detected, and release an observation only when the evidence passes the gate.</p></header>
<div class="process-strip" aria-label="Inspection process"><div><b>01</b> Detect regions</div><div><b>02</b> Check stability</div><div><b>03</b> Verify claims</div></div>''')
        with gr.Row(equal_height=True, elem_id="workspace"):
            with gr.Column(min_width=280):
                gr.HTML('<div class="section-label"><span>01 / Input</span><span>Surface photograph</span></div>')
                input_image = gr.Image(
                    type="pil", label="Upload a surface image", sources=["upload"],
                    height=340, elem_id="input-image", show_share_button=False,
                )
            with gr.Column(min_width=280):
                gr.HTML('<div class="section-label"><span>02 / Result</span><span>Numbered detection regions</span></div>')
                output_image = gr.Image(
                    type="pil", label="Inspection result", interactive=False,
                    height=340, elem_id="output-image", show_share_button=False,
                )
        with gr.Row(elem_id="actions"):
            inspect_button = gr.Button("Run inspection", variant="primary", elem_id="run-inspection", scale=0, min_width=240)
            gr.HTML('<p class="run-note">One image at a time. The first explanation may take several minutes while the model loads.</p>')
        with gr.Row(elem_id="results-row"):
            with gr.Column(scale=2, min_width=280):
                gr.HTML('<div class="section-label"><span>03 / Observation</span></div>')
                explanation = gr.Textbox(
                    label="Observation", show_label=False, lines=5, interactive=False,
                    placeholder="Run an inspection to see the observation or reason for human review.",
                    elem_id="observation",
                )
            with gr.Column(scale=1, min_width=260):
                risk = gr.HTML(WAITING_STATUS, label="Risk decision", elem_id="risk")
                gr.HTML('<p class="review-note">Each region is assessed separately. The overall risk reflects the region needing the most review.</p>')
        with gr.Accordion("Detailed evidence / confidence and gate decisions", open=False, elem_id="evidence-details"):
            metrics = gr.Markdown("Evidence will appear after inspection.", label="Metrics and gate evidence")
        inspect_button.click(
            fn=inspect_for_ui,
            inputs=input_image,
            outputs=[output_image, explanation, risk, metrics],
            api_name="inspect",
        )
        gr.Markdown(
            "**Research demonstration.** A detection or Low-risk result is not a structural safety assessment. Uncertain results need human review.",
            elem_id="research-note",
        )
        gr.HTML('<div class="site-footer"><span>Visual evidence. Explicit uncertainty.</span><span>Detection / Stability / Grounding</span></div>')
    return demo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    build_demo().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_file_size="20mb",
        show_api=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
