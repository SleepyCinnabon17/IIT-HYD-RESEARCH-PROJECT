import pytest
from PIL import Image

import inspection_view as view

POLICY = {"raw_confidence_threshold": 0.05, "localization_threshold": 0.3}


def region(score=0.8, risk="Low", **kwargs):
    return dict(
        raw_confidence=score,
        hallucination_risk=risk,
        box=[5, 5, 35, 35],
        vlm_succeeded=False,
        vlm_called=False,
        vlm_error=None,
        **kwargs,
    )


def test_unstable_real_candidate_is_not_labeled_hallucination():
    result = {"detections": [region(risk="High")]}
    got = view.summarize(Image.new("RGB", (80, 80)), result, POLICY)
    assert got["present"] and got["selected_ids"] == [1]
    assert "Crack detector: 1 region" == got["heading"]
    assert "withheld" in got["observation"]
    assert "hallucination" not in got["observation"].lower()
    assert got["blocked_claims"] == 0


def test_many_weak_candidates_do_not_obscure_two_strong_regions_or_change_ids():
    image = Image.new("RGB", (80, 80), "gray")
    original = image.tobytes()
    regions = [region(0.01) for _ in range(30)] + [
        region(0.8),
        region(0.7, risk="Medium"),
    ]
    got = view.summarize(image, {"detections": regions}, POLICY)
    assert got["selected_ids"] == [31, 32] and got["hidden_count"] == 30
    assert len(regions) == 32
    assert "Region 31:" in got["observation"] and "Region 32:" in got["observation"]
    assert image.tobytes() == original


def test_blocked_language_does_not_remove_crack_evidence():
    r = region()
    r.update(
        vlm_called=True, vlm_error="ValueError: Grounding verifier: wrong location"
    )
    got = view.summarize(Image.new("RGB", (80, 80)), {"detections": [r]}, POLICY)
    assert got["present"] and got["selected_ids"] == [1]
    assert got["blocked_claims"] == 1
    assert "Unsupported language claim blocked" in got["observation"]


def test_model_error_is_not_claimed_to_be_detected_hallucination():
    r = region()
    r.update(vlm_called=True, vlm_error="MemoryError: insufficient memory")
    assert "model error" in view.claim_status(r)
    assert "blocked" not in view.claim_status(r)


def test_possible_presence_without_reliable_box_is_explicit():
    got = view.summarize(
        Image.new("RGB", (80, 80)), {"detections": [region(0.15)]}, POLICY
    )
    assert got["present"] and not got["selected_ids"]
    assert "location uncertain" in got["heading"]


def test_no_candidates_does_not_claim_crack_free():
    got = view.summarize(Image.new("RGB", (80, 80)), {"detections": []}, POLICY)
    assert not got["present"]
    assert "does not prove" in got["observation"]


def test_boundary_is_inclusive_and_nonmutating():
    r = region(0.3)
    got = view.summarize(Image.new("RGB", (80, 80)), {"detections": [r]}, POLICY)
    assert got["selected_ids"] == [1]
    assert r["raw_confidence"] == 0.3


@pytest.mark.parametrize(
    "error",
    [
        "Grounding verifier: inconsistent location",
        "unsupported diagnostic or measurement",
    ],
)
def test_only_explicit_verifier_failures_are_blocked_claims(error):
    r = region()
    r.update(vlm_called=True, vlm_error=error)
    assert view.claim_status(r) == "Unsupported language claim blocked"


@pytest.mark.parametrize("risk", ["Medium", "High"])
def test_mixed_detector_scores_show_both_colors_independent_of_language_gate(risk):
    image = Image.new("RGB", (80, 80), "gray")
    weak = region(0.2, risk="Low")
    weak["box"] = [45, 45, 75, 75]
    got = view.summarize(image, {"detections": [region(risk=risk), weak]}, POLICY)
    assert got["image"].getpixel((8, 35)) == (228, 60, 60)
    assert got["image"].getpixel((48, 75)) == (238, 238, 238)
    assert got["image"].getpixel((54, 75)) == image.getpixel((54, 75))
    assert got["selected_ids"] == [1] and got["uncertain_ids"] == [2]
    assert "Dashed white" in got["detail"]


def test_white_preview_is_bounded_and_does_not_force_red_boxes():
    regions = [region(0.1 + i * 0.01) for i in range(10)]
    got = view.summarize(Image.new("RGB", (80, 80)), {"detections": regions}, POLICY)
    assert got["selected_ids"] == []
    assert got["uncertain_ids"] == [10, 9, 8, 7, 6]
    assert got["hidden_count"] == 5


def test_gate_reason_identifies_missing_matches_and_variation():
    r = region(risk="High", tta_confidences=[0.82, 0.8, 0, 0.78, 0.81],
               variance=0.32, confidence=0.642,
               thresholds={"variance": 0.016384, "confidence": 0.35})
    reason = view.language_gate_reason(r)
    assert "not matched in 1/5" in reason
    assert "32.0 percentage points" in reason
    assert "language limit of 1.6" in reason
    assert "mean score" not in reason


def test_low_mean_explained_without_claiming_variation_failure():
    r = region(confidence=0.2, variance=0.001,
               thresholds={"variance": 0.016384, "confidence": 0.35})
    assert "mean score 20%" in view.language_gate_reason(r)
    assert "variation" not in view.language_gate_reason(r)
