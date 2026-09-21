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
    assert "crack candidate" in got["heading"]
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
