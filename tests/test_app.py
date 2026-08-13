from unittest.mock import patch

import gradio as gr
from PIL import Image

import app


def fake_result(*, risk: str = "High", detected: bool = True) -> dict:
    return {
        "annotated_image": Image.new("RGB", (8, 8), "white"),
        "explanation": "Explanation withheld — flagged for human review",
        "hallucination_risk": risk,
        "detected": detected,
        "raw_confidence": 0.42,
        "confidence": 0.40,
        "variance": 0.05,
        "tta_confidences": [0.42, 0.41, 0.39, 0.38, 0.40],
        "thresholds": {"confidence": 0.35, "variance": 0.03},
        "processing_time": 1.25,
        "vlm_called": False,
        "vlm_succeeded": False,
        "gate_log": "VLM SKIPPED: HIGH RISK",
        "gate_reason": "Explanation withheld because the gate was not satisfied.",
    }


def test_ui_adapter_uses_public_inspect_and_formats_evidence() -> None:
    image = Image.new("RGB", (8, 8), "gray")
    with patch.object(
        app.pipeline, "inspect", return_value=fake_result()
    ) as inspect_spy:
        annotated, explanation, badge, metrics = app.inspect_for_ui(image)

    inspect_spy.assert_called_once_with(image)
    assert annotated.size == (8, 8)
    assert explanation.startswith("Explanation withheld")
    assert "Hallucination risk: High" in badge
    assert "VLM SKIPPED: HIGH RISK" in metrics
    assert "`0.03`" in metrics


def test_ui_rejects_missing_input() -> None:
    try:
        app.inspect_for_ui(None)
    except gr.Error as exc:
        assert "Upload an image" in str(exc)
    else:
        raise AssertionError("Expected missing UI input to fail.")


def test_demo_builds_without_loading_models() -> None:
    demo = app.build_demo()
    assert isinstance(demo, gr.Blocks)
