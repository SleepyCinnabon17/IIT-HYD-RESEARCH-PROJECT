from __future__ import annotations

import statistics

import numpy as np
import pytest
from PIL import Image

import pipeline
from pipeline import (
    NO_DEFECT_MESSAGE,
    VLM_ERROR_MESSAGE,
    WITHHELD_MESSAGE,
    DetectionPass,
    PipelineConfig,
    classify_risk,
    deterministic_tta,
    expand_box,
    inspect,
    load_image,
    run_detection_with_uncertainty,
)


def solid_image(size=(100, 80), value=180):
    return Image.new("RGB", size, (value, value, value))


def sequence_predictor(confidences, boxes=None):
    values = iter(confidences)
    box_values = iter(boxes or [[10.0, 10.0, 40.0, 40.0]] * len(confidences))

    def predict(_detector, _image, _config):
        return DetectionPass(float(next(values)), next(box_values))

    return predict


def test_real_predictor_preserves_pil_rgb_input_contract():
    captured = {}

    class FakeDetector:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return [type("Result", (), {"boxes": None})()]

    result = pipeline._predict_top(
        FakeDetector(), solid_image(), PipelineConfig(device="cpu")
    )
    assert isinstance(captured["source"], Image.Image)
    assert captured["source"].mode == "RGB"
    assert result == DetectionPass(confidence=0.0, box=None)


class FakeVlm:
    def __init__(
        self,
        answer="A narrow dark crack extends diagonally across the visible surface. Extra sentence.",
    ):
        self.answer = answer
        self.crop_size = None
        self.prompt = None

    def encode_image(self, crop):
        self.crop_size = crop.size
        return "encoded"

    def answer_question(self, encoded, prompt, tokenizer):
        assert encoded == "encoded"
        assert tokenizer == "tokenizer"
        self.prompt = prompt
        return self.answer


def test_tta_is_exactly_five_ordered_and_deterministic():
    image = Image.fromarray(np.arange(12 * 10 * 3, dtype=np.uint8).reshape(10, 12, 3))
    first = deterministic_tta(image)
    second = deterministic_tta(image)
    assert [name for name, _ in first] == list(pipeline.TTA_NAMES)
    assert len(first) == 5
    for (_, left), (_, right) in zip(first, second):
        assert np.array_equal(np.asarray(left), np.asarray(right))


def test_missing_detection_becomes_zero_and_statistics_are_population_stddev():
    confidences = [0.8, 0.0, 0.6, 0.4, 0.2]
    boxes = [[1, 1, 20, 20], None, [1, 1, 20, 20], [1, 1, 20, 20], [1, 1, 20, 20]]
    result = run_detection_with_uncertainty(
        solid_image(),
        detector=object(),
        predictor=sequence_predictor(confidences, boxes),
    )
    assert result["tta_confidences"] == confidences
    assert result["confidence"] == pytest.approx(statistics.fmean(confidences))
    assert result["variance"] == pytest.approx(statistics.pstdev(confidences))
    assert result["raw_confidence"] == 0.8
    assert (
        result["uncertainty_metric"]
        == "population_standard_deviation_of_tta_confidence"
    )
    assert result["schema_version"] == "1.0"
    assert result["decision_trace"]["gate_permitted"] is False


def test_no_base_detection_wins_even_if_tta_detects():
    result = run_detection_with_uncertainty(
        solid_image(),
        detector=object(),
        predictor=sequence_predictor(
            [0, 0.8, 0.8, 0.8, 0.8], [None] + [[1, 1, 20, 20]] * 4
        ),
    )
    assert result["detected"] is False
    assert result["box"] is None
    assert result["hallucination_risk"] == "N/A"


@pytest.mark.parametrize(
    ("detected", "confidence", "stddev", "expected"),
    [
        (False, 0.9, 0.0, "N/A"),
        (True, 0.35, 0.15, "Low"),
        (True, 0.3499, 0.15, "Medium"),
        (True, 0.9, 0.15001, "Medium"),
        (True, 0.9, 0.225, "Medium"),
        (True, 0.9, 0.22501, "High"),
    ],
)
def test_risk_boundaries_are_strict(detected, confidence, stddev, expected):
    assert classify_risk(detected, confidence, stddev, 0.35, 0.15) == expected


def test_low_risk_calls_vlm_once_on_expanded_crop_and_normalizes_one_sentence():
    fake = FakeVlm()
    calls = []

    def loader(config):
        calls.append(config)
        return fake, "tokenizer"

    result = inspect(
        solid_image(),
        detector=object(),
        predictor=sequence_predictor([0.8] * 5),
        vlm_loader=loader,
    )
    assert len(calls) == 1
    assert result["vlm_called"] is True
    assert result["vlm_succeeded"] is True
    assert result["hallucination_risk"] == "Low"
    assert (
        result["explanation"]
        == "A narrow dark crack extends diagonally across the visible surface."
    )
    assert result["crop_box"] == [5, 5, 45, 45]
    assert fake.crop_size == (40, 40)
    assert "not a diagnosis" in fake.prompt
    assert result["decision_trace"]["gate_permitted"] is True
    assert result["gate_reason"].startswith("A base detection exists")
    assert len(result["normalized_image_sha256"]) == 64
    assert result["models"]["vlm_revision"] == "2024-08-26"
    assert result["runtime"] == {
        "device": "cpu",
        "detector_dtype": "float32",
        "vlm_dtype": "float32",
    }


@pytest.mark.parametrize(
    "answer",
    (
        "The structure is unsafe and may fail.",
        "The crack is caused by water damage.",
        "The crack is 5 mm wide.",
        "The wall passes safety compliance.",
        "The crack indicates potential structural damage or instability.",
    ),
)
def test_unsupported_vlm_claims_fail_closed(answer):
    fake = FakeVlm(answer=answer)
    result = inspect(
        solid_image(),
        detector=object(),
        predictor=sequence_predictor([0.8] * 5),
        vlm_loader=lambda _config: (fake, "tokenizer"),
    )
    assert result["vlm_called"] is True
    assert result["vlm_succeeded"] is False
    assert result["explanation"] == VLM_ERROR_MESSAGE
    assert "unsupported diagnostic or measurement" in result["vlm_error"]


@pytest.mark.parametrize(
    ("confidences", "message"),
    [
        ([0.3] * 5, WITHHELD_MESSAGE),
        ([0.9, 0, 0.9, 0, 0.9], WITHHELD_MESSAGE),
    ],
)
def test_medium_or_high_risk_never_loads_vlm(confidences, message):
    loader_calls = 0

    def forbidden_loader(_config):
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("VLM loader must not execute")

    result = inspect(
        solid_image(),
        detector=object(),
        predictor=sequence_predictor(confidences),
        vlm_loader=forbidden_loader,
    )
    assert result["hallucination_risk"] in {"Medium", "High"}
    assert result["explanation"] == message
    assert result["vlm_called"] is False
    assert loader_calls == 0


def test_no_detection_never_loads_vlm():
    def forbidden_loader(_config):
        raise AssertionError("VLM loader must not execute")

    result = inspect(
        solid_image(),
        detector=object(),
        predictor=sequence_predictor([0] * 5, [None] * 5),
        vlm_loader=forbidden_loader,
    )
    assert result["explanation"] == NO_DEFECT_MESSAGE
    assert result["vlm_called"] is False
    assert result["gate_log"] == "VLM SKIPPED: NO DETECTION"
    assert result["crop_box"] is None


def test_vlm_failure_fails_closed_without_fabricated_description():
    def broken_loader(_config):
        raise MemoryError("test memory limit")

    result = inspect(
        solid_image(),
        detector=object(),
        predictor=sequence_predictor([0.8] * 5),
        vlm_loader=broken_loader,
    )
    assert result["vlm_called"] is True
    assert result["vlm_succeeded"] is False
    assert result["explanation"] == VLM_ERROR_MESSAGE
    assert result["vlm_error"] == "MemoryError: test memory limit"


def test_expand_box_clamps_and_rejects_invalid_area():
    assert expand_box([0, 0, 20, 20], (100, 80)) == (0, 0, 23, 23)
    assert expand_box([90, 70, 100, 80], (100, 80)) == (88, 68, 100, 80)
    with pytest.raises(ValueError, match="zero or negative"):
        expand_box([5, 5, 5, 10], (100, 80))


@pytest.mark.parametrize(
    "box",
    ([1, 2, 3], [1, 2, float("nan"), 4], [1, 2, 1, 4]),
)
def test_detector_rejects_malformed_boxes_before_serialization(box):
    predictor = sequence_predictor([0.8] * 5, [box] * 5)
    with pytest.raises(ValueError, match="Detection box"):
        run_detection_with_uncertainty(
            solid_image(), detector=object(), predictor=predictor
        )


def test_invalid_inputs_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="required"):
        load_image(None)
    with pytest.raises(ValueError, match="does not exist"):
        load_image(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="Unable to read"):
        load_image(bad)


def test_annotated_image_is_returned_without_mutating_input():
    source = solid_image()
    original = np.asarray(source).copy()
    result = inspect(source, detector=object(), predictor=sequence_predictor([0.3] * 5))
    assert isinstance(result["annotated_image"], Image.Image)
    assert np.array_equal(np.asarray(source), original)
    assert not np.array_equal(np.asarray(result["annotated_image"]), original)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_threshold": -0.1}, "confidence_threshold"),
        ({"confidence_threshold": float("nan")}, "confidence_threshold"),
        ({"variance_threshold": 0.6}, "variance_threshold"),
        ({"detector_confidence_floor": 1.1}, "detector_confidence_floor"),
        ({"image_size": 0}, "image_size"),
        ({"crop_expansion": -0.1}, "crop_expansion"),
        ({"device": "gpu"}, "device"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig(**kwargs)


def test_normalized_fingerprint_is_stable_and_pixel_sensitive():
    first = inspect(
        solid_image(value=100),
        detector=object(),
        predictor=sequence_predictor([0.3] * 5),
    )
    repeat = inspect(
        solid_image(value=100),
        detector=object(),
        predictor=sequence_predictor([0.3] * 5),
    )
    changed = inspect(
        solid_image(value=101),
        detector=object(),
        predictor=sequence_predictor([0.3] * 5),
    )
    assert first["normalized_image_sha256"] == repeat["normalized_image_sha256"]
    assert first["normalized_image_sha256"] != changed["normalized_image_sha256"]


def test_cli_invalid_config_returns_json_error_without_traceback(capsys):
    exit_code = pipeline.main(["unused.jpg", "--confidence-threshold", "-0.1"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert '"error"' in captured.out
    assert "confidence_threshold" in captured.out
    assert "Traceback" not in captured.out
