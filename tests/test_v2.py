import numpy as np
import pytest
from PIL import Image

import evaluate_v2 as ev
import pipeline as p


def multi_predictor(missing_secondary=False):
    calls = 0

    def predict(_detector, _image, _config):
        nonlocal calls
        index = calls
        calls += 1
        boxes = [[5, 10, 25, 40], [70, 10, 90, 40]]
        if index == 1:
            boxes = [[100-b[2], b[1], 100-b[0], b[3]] for b in boxes]
        result = [p.DetectionPass(.8, boxes[0]), p.DetectionPass(.6, boxes[1])]
        if missing_secondary and index in (1, 3):
            result = result[:1]
        return result
    return predict


class Vlm:
    def encode_image(self, image):
        return image

    def answer_question(self, *_):
        return 'A dark crack is visible.'


def test_each_region_gets_gate_and_vlm_only_for_stable_region():
    calls = []
    result = p.inspect(Image.new('RGB', (100, 80)), detector=object(),
                       config=p.PipelineConfig(device='cpu', variance_threshold=.03),
                       all_predictor=multi_predictor(True),
                       vlm_loader=lambda cfg: (calls.append(cfg) or Vlm(), None))
    first, second = result['detections']
    assert first['tta_confidences'] == [.8]*5
    assert second['tta_confidences'] == [.6, 0, .6, 0, .6]
    assert first['vlm_succeeded'] and not second['vlm_called']
    assert len(calls) == 1
    assert result['hallucination_risk'] == 'High'
    assert 'Region 2 (High)' in result['explanation']
    assert first['box'] != second['box']
    assert result['pass_box_counts'] == [2, 1, 2, 1, 2]


def test_all_stable_regions_are_explained():
    calls = []
    result = p.inspect(Image.new('RGB', (100, 80)), detector=object(),
                       config=p.PipelineConfig(device='cpu', variance_threshold=.03),
                       all_predictor=multi_predictor(),
                       vlm_loader=lambda cfg: (calls.append(cfg) or Vlm(), None))
    assert len(calls) == 2
    assert all(r['vlm_succeeded'] for r in result['detections'])


def test_matching_is_one_to_one_and_does_not_swap_by_confidence_rank():
    base = [p.DetectionPass(.9, [0, 0, 20, 20]), p.DetectionPass(.8, [60, 0, 80, 20])]
    candidates = [p.DetectionPass(.95, [60, 0, 80, 20]), p.DetectionPass(.4, [0, 0, 20, 20])]
    assert p.match_detections(base, candidates) == [.4, .95]
    assert sum(c > 0 for c in p.match_detections(base*2, candidates)) == 2


@pytest.mark.parametrize('name,box,aug,expected', [
    ('horizontal_flip', [70, 10, 90, 30], (100, 80), [10, 10, 30, 30]),
    ('vertical_flip', [10, 50, 30, 70], (100, 80), [10, 10, 30, 30]),
    ('scale_0.85', [8.5, 8.5, 25.5, 25.5], (85, 68), [10, 10, 30, 30]),
])
def test_inverse_coordinates(name, box, aug, expected):
    assert p.inverse_box(box, name, (100, 80), aug) == pytest.approx(expected)


def test_rotation_inverse_uses_pil_counterclockwise_convention():
    # A point to the right of center moves upward for PIL +90 rotation.
    assert p.inverse_box([49, 19, 51, 21], 'rotation_+90', (100, 100), (100, 100)) == pytest.approx([79, 49, 81, 51])


@pytest.mark.parametrize('text,passed', [
    ('The region is at the left.', True),
    ('The region is at the right.', False),
    ('The bounding region is horizontal.', True),
    ('The bounding region is vertical.', False),
    ('The region has width 20%.', True),
    ('The region has width 90%.', False),
    ('The confidence is 80%.', True),
    ('The confidence is 99%.', False),
    ('There is no crack.', False),
    ('The structure is unsafe.', False),
    ('The region is not at the left.', False),
])
def test_geometry_verification(text, passed):
    assert p.verify_grounding(text, [5, 5, 25, 15], (100, 100), .8)['passed'] == passed


def test_unmeasured_appearance_is_explicitly_not_certified():
    result = p.verify_grounding('A branching dark line is visible.', [5, 5, 25, 15], (100, 100), .8)
    assert result['checked_claims'] == []
    assert 'not certified' in result['scope']


def test_grounding_failure_withholds_real_output():
    class Wrong(Vlm):
        def answer_question(self, *_):
            return 'The region has area 1%.'
    result = p.inspect(Image.new('RGB', (100, 80)), detector=object(),
                       config=p.PipelineConfig(device='cpu', variance_threshold=.03),
                       all_predictor=multi_predictor(), vlm_loader=lambda _: (Wrong(), None))
    assert all(not d['vlm_succeeded'] and d['grounding']['passed'] is False for d in result['detections'])


def test_tta_diversity_and_reproducibility():
    image = Image.fromarray(np.random.default_rng(3).integers(0, 255, (100, 100, 3), dtype=np.uint8))
    a, b = p.deterministic_tta(image, 15), p.deterministic_tta(image, 15)
    assert len({name for name, _ in a}) == 15
    assert all(x.tobytes() == y.tobytes() for (_, x), (_, y) in zip(a, b))
    assert len({im.tobytes() for _, im in a}) == 15


def test_trust_truth_rejects_duplicate_and_wrong_region():
    labels = ev.region_truth([{'box': [0, 0, 20, 20]}, {'box': [0, 0, 20, 20]}, {'box': [60, 60, 80, 80]}], [[0, 0, 20, 20]])
    assert sum(labels) == 1
    assert labels[-1] is False


def test_no_accepted_regions_has_no_false_explanation_claim():
    result = ev.operating([True, False], [.8, .8], [.1, .2], 0.)
    assert result['false_explanation_rate_proxy'] is None
    assert result['one_sided_95_upper'] == 1.


def test_confidence_intervals_do_not_claim_certainty_for_all_positive_predictor():
    result = ev.bootstrap_metrics([True]*199 + [False], [True]*200)
    assert result['ci95']['f1']['width'] == 0
    assert result['simultaneous_rate_ci95']['f1']['width'] > 0
    assert result['simultaneous_rate_ci95']['recall']['low'] < 1
    assert result['exact_binomial_ci95']['specificity'] == pytest.approx([0, .975])


def test_missing_ground_truth_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.label_boxes(tmp_path / 'missing.txt', (100, 100))


def test_inference_floor_is_applied_to_every_candidate():
    def predict(*_):
        return [p.DetectionPass(.009, [1, 1, 50, 50]), p.DetectionPass(.01, [0, 0, 100, 80])]
    result = p.inspect(Image.new('RGB', (100, 80)), detector=object(),
                       config=p.PipelineConfig(device='cpu'), all_predictor=predict,
                       vlm_loader=lambda _: pytest.fail('Low-confidence region must not call VLM'))
    assert len(result['detections']) == 1
    assert result['detections'][0]['raw_confidence'] == .01


def test_all_detection_path_handles_empty_results_without_vlm():
    calls = []
    def predict(*_):
        calls.append(1)
        return []
    result = p.inspect(Image.new('RGB', (100, 80)), detector=object(),
                       config=p.PipelineConfig(device='cpu'), all_predictor=predict,
                       vlm_loader=lambda _: pytest.fail('No detection must not call VLM'))
    assert len(calls) == 5
    assert result['detections'] == []
    assert result['explanation'] == p.NO_DEFECT_MESSAGE


def test_deployment_calibration_rejects_wrong_weights(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(p, '__file__', str(tmp_path / 'pipeline.py'))
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'models'
    folder.mkdir()
    (folder / 'crack_yolov8s_best.pt').write_bytes(b'test checkpoint')
    (folder / 'gate_calibration.json').write_text(json.dumps({
        'variance_threshold': .02, 'confidence_threshold': .35, 'tta_passes': 5,
        'detector_sha256': 'incorrect',
    }))
    with pytest.raises(ValueError, match='does not match'):
        p.deployed_config()
