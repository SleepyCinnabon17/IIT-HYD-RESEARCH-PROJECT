"""Hallucination-aware crack inspection pipeline.

The public entry point is :func:`inspect`. The detector always runs before the
language model, and only a low-risk detection may cross the explanation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

WITHHELD_MESSAGE = "Explanation withheld — flagged for human review"
NO_DEFECT_MESSAGE = "No defect detected."
VLM_ERROR_MESSAGE = "Explanation unavailable — VLM failed closed; human review required"
TTA_NAMES = (
    "original",
    "horizontal_flip",
    "brightness_0.8",
    "contrast_1.15",
    "rotation_+3",
)
VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"
RESULT_SCHEMA_VERSION = "1.0"
UNSUPPORTED_EXPLANATION_PATTERN = re.compile(
    r"\b(?:caused by|due to|structural integrity|structurally (?:sound|unsafe)|"
    r"structural damage|instability|safe|unsafe|collapse|future failure|"
    r"compliance|certif(?:y|ied|ication)|"
    r"material (?:is|appears)|\d+(?:\.\d+)?\s*(?:mm|cm|met(?:er|re)s?|inches?|feet|ft))\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PipelineConfig:
    detector_path: str = "models/crack_yolov8s_best.pt"
    confidence_threshold: float = 0.35
    variance_threshold: float = 0.03
    detector_confidence_floor: float = 0.01
    image_size: int = 640
    crop_expansion: float = 0.15
    device: str = "auto"
    tta_passes: int = 5

    def __post_init__(self) -> None:
        numeric_fields = {
            "confidence_threshold": self.confidence_threshold,
            "variance_threshold": self.variance_threshold,
            "detector_confidence_floor": self.detector_confidence_floor,
            "crop_expansion": self.crop_expansion,
        }
        for name, value in numeric_fields.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1].")
        if not 0.0 <= self.variance_threshold <= 0.5:
            raise ValueError("variance_threshold must be within [0, 0.5].")
        if not 0.0 <= self.detector_confidence_floor <= 1.0:
            raise ValueError("detector_confidence_floor must be within [0, 1].")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero.")
        if self.crop_expansion < 0.0:
            raise ValueError("crop_expansion must be non-negative.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        if self.tta_passes not in range(3, 16):
            raise ValueError("tta_passes must be within [3, 15].")

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class DetectionPass:
    confidence: float
    box: list[float] | None


_DETECTOR_CACHE: dict[str, Any] = {}
_VLM_CACHE: tuple[Any, Any] | None = None


def deployed_config() -> PipelineConfig:
    """Use the reproducible calibration artifact for normal CLI/UI inspection."""
    path = Path(__file__).parent / "models/gate_calibration.json"
    if not path.exists():
        return PipelineConfig()
    calibration = json.loads(path.read_text(encoding="utf-8"))
    cfg = PipelineConfig(variance_threshold=calibration["variance_threshold"],
                         confidence_threshold=calibration["confidence_threshold"],
                         tta_passes=calibration["tta_passes"])
    with Path(cfg.detector_path).open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    if checksum != calibration['detector_sha256']:
        raise ValueError("Gate calibration does not match detector weights; recalibration required.")
    return cfg


def load_image(image: str | os.PathLike[str] | Image.Image) -> Image.Image:
    """Load, orient, and normalize an image to non-empty RGB pixels."""
    if image is None:
        raise ValueError("An image is required.")
    try:
        if isinstance(image, Image.Image):
            loaded = image.copy()
        else:
            path = Path(image)
            if not path.is_file():
                raise ValueError(f"Image does not exist: {path}")
            with Image.open(path) as opened:
                loaded = opened.copy()
        loaded = ImageOps.exif_transpose(loaded).convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read image: {exc}") from exc
    if loaded.width <= 0 or loaded.height <= 0:
        raise ValueError("Image dimensions must be greater than zero.")
    return loaded


def deterministic_tta(image: Image.Image, n: int = 5) -> list[tuple[str, Image.Image]]:
    """Return the exact, ordered five-pass deterministic augmentation sequence."""
    rgb = load_image(image)
    if n not in range(1, 16):
        raise ValueError("TTA pass count must be within [1, 15].")
    variants = [
        (TTA_NAMES[0], rgb.copy()),
        (TTA_NAMES[1], ImageOps.mirror(rgb)),
        (TTA_NAMES[2], ImageEnhance.Brightness(rgb).enhance(0.8)),
        (TTA_NAMES[3], ImageEnhance.Contrast(rgb).enhance(1.15)),
        (
            TTA_NAMES[4],
            rgb.rotate(
                3.0,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=(128, 128, 128),
            ),
        ),
    ]
    variants.extend([
        ("vertical_flip", ImageOps.flip(rgb)),
        ("scale_0.85", rgb.resize((round(rgb.width * .85), round(rgb.height * .85)))),
        ("color_0.5", ImageEnhance.Color(rgb).enhance(.5)),
        ("brightness_1.2", ImageEnhance.Brightness(rgb).enhance(1.2)),
        ("rotation_-3", rgb.rotate(-3, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))),
        ("scale_1.15", rgb.resize((round(rgb.width * 1.15), round(rgb.height * 1.15)))),
        ("contrast_0.85", ImageEnhance.Contrast(rgb).enhance(.85)),
        ("rotation_+7", rgb.rotate(7, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))),
        ("color_1.5", ImageEnhance.Color(rgb).enhance(1.5)),
        ("rotation_-7", rgb.rotate(-7, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))),
    ])
    return variants[:n]


def classify_risk(
    detected: bool,
    calibrated_confidence: float,
    tta_stddev: float,
    confidence_threshold: float,
    variance_threshold: float,
) -> str:
    """Classify hallucination risk using strict prompt-defined boundaries."""
    if not detected:
        return "N/A"
    high_boundary = variance_threshold * 1.5
    if tta_stddev > high_boundary and not math.isclose(
        tta_stddev, high_boundary, rel_tol=0.0, abs_tol=1e-12
    ):
        return "High"
    if tta_stddev > variance_threshold and not math.isclose(
        tta_stddev, variance_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        return "Medium"
    if calibrated_confidence < confidence_threshold:
        return "Medium"
    return "Low"


def _load_detector(config: PipelineConfig) -> Any:
    path = str(Path(config.detector_path).resolve())
    if path not in _DETECTOR_CACHE:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {path}")
        from ultralytics import YOLO

        _DETECTOR_CACHE[path] = YOLO(path)
    return _DETECTOR_CACHE[path]


def _predict_top(
    detector: Any, image: Image.Image, config: PipelineConfig
) -> DetectionPass:
    device = config.resolved_device()
    predict_options: dict[str, Any] = {
        # Preserve the PIL RGB contract. Ultralytics interprets raw ndarray
        # input as OpenCV-style BGR, which would silently swap color channels.
        "source": image,
        "imgsz": config.image_size,
        "conf": config.detector_confidence_floor,
        "device": device,
        "verbose": False,
    }
    if device == "cuda":
        predict_options["half"] = True
    result = detector.predict(
        **predict_options,
    )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return DetectionPass(confidence=0.0, box=None)
    index = int(boxes.conf.argmax().item())
    confidence = float(boxes.conf[index].item())
    box = [float(value) for value in boxes.xyxy[index].tolist()]
    return DetectionPass(confidence=confidence, box=box)


Predictor = Callable[[Any, Image.Image, PipelineConfig], DetectionPass]


def _predict_all(detector: Any, image: Image.Image, config: PipelineConfig) -> list[DetectionPass]:
    result = detector.predict(source=image, imgsz=config.image_size,
                              conf=config.detector_confidence_floor,
                              device=config.resolved_device(),
                              half=config.resolved_device() == "cuda", verbose=False)[0]
    if result.boxes is None:
        return []
    return sorted([DetectionPass(float(c), _validated_box(b))
                   for c, b in zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist())],
                  key=lambda p: -p.confidence)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def inverse_box(box: Sequence[float], name: str, original_size: tuple[int, int], augmented_size: tuple[int, int]) -> list[float]:
    """Map all four corners back to the original image before matching."""
    w, h = original_size
    points = [(box[x], box[y]) for x, y in ((0, 1), (0, 3), (2, 1), (2, 3))]
    if name == "horizontal_flip":
        points = [(w-x, y) for x, y in points]
    elif name == "vertical_flip":
        points = [(x, h-y) for x, y in points]
    elif name.startswith("scale_"):
        points = [(x*w/augmented_size[0], y*h/augmented_size[1]) for x, y in points]
    elif name.startswith("rotation_"):
        angle = math.radians(float(name.split("_")[1]))
        c, s = math.cos(angle), math.sin(angle)
        points = [(c*(x-w/2)-s*(y-h/2)+w/2, s*(x-w/2)+c*(y-h/2)+h/2) for x, y in points]
    xs, ys = zip(*points)
    return [max(0., min(xs)), max(0., min(ys)), min(float(w), max(xs)), min(float(h), max(ys))]


def match_detections(base: list[DetectionPass], candidates: list[DetectionPass], min_iou: float = .3) -> list[float]:
    """Greedy highest-IoU one-to-one association; a missing match contributes zero."""
    edges = sorted(((box_iou(a.box, b.box), i, j) for i, a in enumerate(base)
                    for j, b in enumerate(candidates)), reverse=True)
    scores = [0.] * len(base)
    used_base, used_candidate = set(), set()
    for overlap, i, j in edges:
        if overlap >= min_iou and i not in used_base and j not in used_candidate:
            scores[i] = candidates[j].confidence
            used_base.add(i)
            used_candidate.add(j)
    return scores


def run_all_detections(image: Image.Image, config: PipelineConfig, detector: Any,
                       predictor: Any = None) -> dict[str, Any]:
    """Share image inference while keeping a distinct matched TTA sequence per base box."""
    predict = predictor or _predict_all
    augmented = deterministic_tta(image, config.tta_passes)
    all_passes = []
    for name, frame in augmented:
        raw = predict(detector, frame, config)
        if isinstance(raw, DetectionPass):
            raw = [raw]
        valid = []
        for p in raw:
            if not math.isfinite(p.confidence) or not 0 <= p.confidence <= 1:
                raise ValueError("Detector confidence must be finite and within [0, 1].")
            if p.box is not None and p.confidence >= config.detector_confidence_floor:
                valid.append(DetectionPass(p.confidence, inverse_box(_validated_box(p.box), name, image.size, frame.size)))
        all_passes.append(valid)
    base = sorted(all_passes[0], key=lambda p: -p.confidence)
    matched = [[p.confidence for p in base]] + [match_detections(base, candidates) for candidates in all_passes[1:]]
    regions = []
    for i, p in enumerate(base):
        scores = [row[i] for row in matched]
        # Reuse the audited decision construction without another inference call.
        seq = iter(DetectionPass(c, p.box if c else None) for c in scores)
        region = run_detection_with_uncertainty(image, config=config, detector=detector,
                                                predictor=lambda *_: next(seq))  # noqa: B023 - consumed synchronously before the next iteration
        region["detection_id"] = i
        regions.append(region)
    return {"detections": regions, "pass_box_counts": [len(p) for p in all_passes],
            "legacy_top_confidences": [max((p.confidence for p in rows), default=0.) for rows in all_passes]}


def _validated_box(box: Sequence[float]) -> list[float]:
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError("Detection box must contain four finite xyxy values.")
    values = [float(value) for value in box]
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("Detection box has zero or negative area.")
    return values


def run_detection_with_uncertainty(
    image: str | os.PathLike[str] | Image.Image,
    *,
    config: PipelineConfig | None = None,
    detector: Any | None = None,
    predictor: Predictor | None = None,
) -> dict[str, Any]:
    """Run exactly five detector passes and return confidence stability data.

    The compatibility field ``variance`` contains population standard deviation,
    not mathematical variance or Bayesian posterior uncertainty.
    """
    cfg = config or deployed_config()
    rgb = load_image(image)
    active_detector = detector if detector is not None else _load_detector(cfg)
    predict = predictor or _predict_top

    passes: list[DetectionPass] = []
    for _, augmented in deterministic_tta(rgb, cfg.tta_passes):
        prediction = predict(active_detector, augmented, cfg)
        confidence = float(prediction.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Detector confidence must be finite and within [0, 1], got {confidence!r}"
            )
        box = (
            _validated_box(prediction.box)
            if confidence > 0.0 and prediction.box is not None
            else None
        )
        passes.append(
            DetectionPass(confidence=confidence if box is not None else 0.0, box=box)
        )

    confidences = [item.confidence for item in passes]
    calibrated = statistics.fmean(confidences)
    stddev = statistics.pstdev(confidences)
    base = passes[0]
    detected = base.box is not None
    risk = classify_risk(
        detected,
        calibrated,
        stddev,
        cfg.confidence_threshold,
        cfg.variance_threshold,
    )
    confidence_passed = calibrated >= cfg.confidence_threshold
    uncertainty_passed = stddev <= cfg.variance_threshold
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "detected": detected,
        "confidence": calibrated,
        "raw_confidence": base.confidence,
        "variance": stddev,
        "box": base.box,
        "hallucination_risk": risk,
        "tta_confidences": confidences,
        "tta_names": [name for name, _ in deterministic_tta(rgb, cfg.tta_passes)],
        "uncertainty_metric": "population_standard_deviation_of_tta_confidence",
        "thresholds": {
            "confidence": cfg.confidence_threshold,
            "variance": cfg.variance_threshold,
            "high_risk_variance": round(cfg.variance_threshold * 1.5, 12),
            "detector_floor": cfg.detector_confidence_floor,
        },
        "decision_trace": {
            "base_detection_present": detected,
            "confidence_passed": confidence_passed,
            "uncertainty_passed": uncertainty_passed,
            "risk": risk,
            "gate_permitted": detected and risk == "Low",
        },
    }


def expand_box(
    box: Sequence[float], image_size: tuple[int, int], expansion: float = 0.15
) -> tuple[int, int, int, int]:
    """Expand an xyxy box on every side, clamp it, and reject empty crops."""
    x1, y1, x2, y2 = _validated_box(box)
    width, height = image_size
    dx = (x2 - x1) * expansion
    dy = (y2 - y1) * expansion
    expanded = (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)),
        min(height, math.ceil(y2 + dy)),
    )
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        raise ValueError("Expanded detection box is outside the image.")
    return expanded


def _load_vlm(config: PipelineConfig) -> tuple[Any, Any]:
    global _VLM_CACHE
    if _VLM_CACHE is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = config.resolved_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            VLM_MODEL_ID,
            revision=VLM_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL_ID, revision=VLM_REVISION)
        _VLM_CACHE = (model, tokenizer)
    return _VLM_CACHE


VlmLoader = Callable[[PipelineConfig], tuple[Any, Any]]


def _one_factual_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        raise ValueError("VLM returned an empty explanation.")
    first = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    if first[-1] not in ".!?":
        first += "."
    if UNSUPPORTED_EXPLANATION_PATTERN.search(first):
        raise ValueError(
            "VLM explanation contained unsupported diagnostic or measurement language."
        )
    return first


def _generate_explanation(
    crop: Image.Image, config: PipelineConfig, loader: VlmLoader
) -> str:
    model, tokenizer = loader(config)
    prompt = (
        "Describe only the visible crack evidence in this crop in exactly one factual sentence. "
        "Do not guess causes, measurements, hidden damage, material composition, structural integrity, "
        "future failure, safety, or engineering compliance. State only visible extent, direction, branching, "
        "or apparent severity when directly justified by the pixels. This is an observation, not a diagnosis."
    )
    encoded = model.encode_image(crop)
    answer = model.answer_question(encoded, prompt, tokenizer)
    return _one_factual_sentence(answer)


def annotate_image(image: Image.Image, box: Sequence[float] | None) -> Image.Image:
    annotated = load_image(image)
    if box is not None:
        draw = ImageDraw.Draw(annotated)
        line_width = max(3, round(min(annotated.size) / 150))
        draw.rectangle(
            tuple(float(value) for value in box), outline="red", width=line_width
        )
    return annotated


def _inspect_single(
    image: str | os.PathLike[str] | Image.Image,
    *,
    config: PipelineConfig | None = None,
    detector: Any | None = None,
    predictor: Predictor | None = None,
    vlm_loader: VlmLoader | None = None,
    detection_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect one image; this is the only pipeline function permitted to call the VLM."""
    started = time.perf_counter()
    cfg = config or deployed_config()
    rgb = load_image(image)
    fingerprint = hashlib.sha256()
    fingerprint.update(rgb.width.to_bytes(8, "big"))
    fingerprint.update(rgb.height.to_bytes(8, "big"))
    fingerprint.update(rgb.tobytes())
    result = detection_result.copy() if detection_result is not None else run_detection_with_uncertainty(
        rgb,
        config=cfg,
        detector=detector,
        predictor=predictor,
    )
    result.update(
        {
            "normalized_image_sha256": fingerprint.hexdigest(),
            "models": {
                "detector": Path(cfg.detector_path).name,
                "vlm": VLM_MODEL_ID,
                "vlm_revision": VLM_REVISION,
            },
            "runtime": {
                "device": cfg.resolved_device(),
                "detector_dtype": "float16"
                if cfg.resolved_device() == "cuda"
                else "float32",
                "vlm_dtype": "float16"
                if cfg.resolved_device() == "cuda"
                else "float32",
            },
            "vlm_called": False,
            "vlm_succeeded": False,
            "vlm_error": None,
            "crop_box": None,
        }
    )

    if not result["detected"]:
        result["explanation"] = NO_DEFECT_MESSAGE
        gate_log = "VLM SKIPPED: NO DETECTION"
        gate_reason = "The original detector pass produced no valid region."
    elif result["hallucination_risk"] != "Low":
        result["explanation"] = WITHHELD_MESSAGE
        gate_log = f"VLM SKIPPED: {result['hallucination_risk'].upper()} RISK"
        gate_reason = "Explanation withheld because calibrated confidence or TTA stability did not satisfy the Low-risk gate."
    else:
        result["vlm_called"] = True
        gate_log = "VLM CALLED"
        gate_reason = "A base detection exists and both confidence and TTA stability satisfy the Low-risk gate."
        try:
            crop_box = expand_box(result["box"], rgb.size, cfg.crop_expansion)
            crop = rgb.crop(crop_box)
            result["crop_box"] = list(crop_box)
            result["explanation"] = _generate_explanation(
                crop, cfg, vlm_loader or _load_vlm
            )
            local_box = [result["box"][0]-crop_box[0], result["box"][1]-crop_box[1],
                         result["box"][2]-crop_box[0], result["box"][3]-crop_box[1]]
            result["grounding"] = verify_grounding(result["explanation"], local_box, crop.size, result["confidence"])
            if not result["grounding"]["passed"]:
                raise ValueError("Grounding verifier: " + "; ".join(result["grounding"]["reasons"]))
            result["vlm_succeeded"] = True
        except Exception as exc:  # noqa: BLE001 - the safety boundary must fail closed for every VLM failure
            result["explanation"] = VLM_ERROR_MESSAGE
            result["vlm_error"] = f"{type(exc).__name__}: {exc}"

    result["gate_log"] = gate_log
    result["gate_reason"] = gate_reason
    result["processing_time"] = time.perf_counter() - started
    result["annotated_image"] = annotate_image(rgb, result["box"])
    print(gate_log, flush=True)
    return result


def verify_grounding(text: str, box: Sequence[float], image_size: tuple[int, int], confidence: float) -> dict[str, Any]:
    """Check explicit geometric claims in the coordinate frame seen by the VLM.

    This is a contradiction detector, not a semantic truth certificate. Appearance,
    branching and severity are not inferable from box geometry and remain unverified.
    """
    x1, y1, x2, y2 = _validated_box(box)
    w, h = image_size
    attributes = {"width": (x2-x1)/w, "height": (y2-y1)/h,
                  "area": (x2-x1)*(y2-y1)/(w*h), "confidence": confidence,
                  "center_x": (x1+x2)/(2*w), "center_y": (y1+y2)/(2*h),
                  "aspect_ratio": (x2-x1)/(y2-y1)}
    lower = text.lower()
    reasons, checked = [], []
    if UNSUPPORTED_EXPLANATION_PATTERN.search(text):
        reasons.append("unsupported diagnostic or measurement language")
    predicates = {
        "left": attributes["center_x"] < .4,
        "right": attributes["center_x"] > .6,
        "top": attributes["center_y"] < .4,
        "bottom": attributes["center_y"] > .6,
        "center": .35 <= attributes["center_x"] <= .65 and .35 <= attributes["center_y"] <= .65,
        "horizontal": attributes["aspect_ratio"] >= 1.5,
        "vertical": attributes["aspect_ratio"] <= 1/1.5,
        "tiny": attributes["area"] <= .05,
    }
    for word, consistent in predicates.items():
        if re.search(r"\b" + word + r"\b", lower):
            checked.append(word)
            if not consistent:
                reasons.append(f"{word} contradicts measured bounding region")
    for match in re.finditer(r"\b(width|height|area|confidence)\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%", lower):
        attribute, amount = match.groups()
        checked.append(attribute)
        if abs(float(amount)/100 - attributes[attribute]) > .05:
            reasons.append(f"{attribute} percentage differs by more than 5 percentage points")
    if re.search(r"\b(?:no crack|no defect|crack.free)\b", lower):
        checked.append("presence")
        reasons.append("absence claim conflicts with the detected region")
    if re.search(r"\b(?:not|never|neither)\b", lower) and checked:
        reasons.append("negated geometric claim requires human review")
    return {"passed": not reasons, "reasons": reasons, "attributes": attributes,
            "checked_claims": checked, "scope": "geometric contradictions only; appearance and causality not certified"}


def inspect(image: str | os.PathLike[str] | Image.Image, *, config: PipelineConfig | None = None,
            detector: Any | None = None, predictor: Predictor | None = None,
            vlm_loader: VlmLoader | None = None, all_predictor: Any = None,
            detection_batch: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gate and explain every base detection; legacy scalar predictor remains a test adapter.

    detection_batch permits evaluation to replay actual cached detector passes through
    the same language boundary. It must come from run_all_detections with this config.
    """
    started = time.perf_counter()
    cfg = config or deployed_config()
    rgb = load_image(image)
    if predictor is not None:
        result = _inspect_single(rgb, config=cfg, detector=detector, predictor=predictor, vlm_loader=vlm_loader)
        result["detections"] = [_json_result(result)] if result["detected"] else []
        return result
    active = detector if detector is not None else _load_detector(cfg)
    batch = detection_batch if detection_batch is not None else run_all_detections(rgb, cfg, active, all_predictor)
    if not batch["detections"]:
        result = _inspect_single(rgb, config=cfg, detector=active,
                                 predictor=lambda *_: DetectionPass(0., None), vlm_loader=vlm_loader)
        result["detections"] = []
        return result
    regions = [_inspect_single(rgb, config=cfg, detector=active, vlm_loader=vlm_loader,
                               detection_result=region) for region in batch["detections"]]
    result = regions[0].copy()
    result["detections"] = [_json_result(region) for region in regions]
    result["pass_box_counts"] = batch["pass_box_counts"]
    result["hallucination_risk"] = max((r["hallucination_risk"] for r in regions), key={"Low": 0, "Medium": 1, "High": 2}.get)
    result["vlm_called"] = any(r["vlm_called"] for r in regions)
    result["vlm_succeeded"] = any(r["vlm_succeeded"] for r in regions)
    result["explanation"] = "\n".join(f"Region {i+1} ({r['hallucination_risk']}): {r['explanation']}" for i, r in enumerate(regions))
    result["gate_log"] = f"{len(regions)} REGIONS: {sum(r['vlm_called'] for r in regions)} VLM CALLS"
    result["gate_reason"] = "Each region has an independent gate; image risk is the highest region risk. Scalar confidence and box describe the top region."
    annotated = rgb.copy()
    for i, r in enumerate(regions):
        annotated = annotate_image(annotated, r["box"])
        ImageDraw.Draw(annotated).text(tuple(r["box"][:2]), str(i+1), fill="yellow")
    result["annotated_image"] = annotated
    result["processing_time"] = time.perf_counter() - started
    return result


def _json_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "annotated_image"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run hallucination-aware crack inspection."
    )
    parser.add_argument("image_path", help="Path to an input image")
    parser.add_argument("--detector", default=PipelineConfig.detector_path)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=deployed_config().confidence_threshold,
    )
    parser.add_argument(
        "--variance-threshold", type=float, default=deployed_config().variance_threshold
    )
    parser.add_argument("--save-annotated", type=Path)
    parser.add_argument(
        "--save-json", type=Path, help="Save the structured result without image pixels"
    )
    args = parser.parse_args(argv)
    try:
        config = PipelineConfig(
            detector_path=args.detector,
            confidence_threshold=args.confidence_threshold,
            variance_threshold=args.variance_threshold,
        )
        result = inspect(args.image_path, config=config)
        if args.save_annotated:
            args.save_annotated.parent.mkdir(parents=True, exist_ok=True)
            result["annotated_image"].save(args.save_annotated)
        json_result = _json_result(result)
        if args.save_json:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_json.write_text(
                json.dumps(json_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(json.dumps(json_result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI converts all failures into actionable JSON
        print(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), flush=True
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
