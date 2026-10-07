"""Separate detector evidence from language reliability in the public interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def operating_point() -> dict:
    root = Path(__file__).resolve().parent
    artifact = json.loads((root / "models/detection_operating_point.json").read_text())
    model = json.loads((root / "models/MODEL_SOURCE.json").read_text())
    if artifact["detector_sha256"] != model["sha256"]:
        raise ValueError("Detection calibration does not match the pinned detector.")
    for key in ("raw_confidence_threshold", "localization_threshold"):
        if not 0.01 <= artifact[key] <= 1:
            raise ValueError("Invalid detection operating point.")
    return artifact


def claim_status(region: dict) -> str:
    if region.get("vlm_succeeded"):
        return "Description passed automated checks"
    error = region.get("vlm_error") or ""
    if (
        "Grounding verifier:" in error
        or "unsupported diagnostic or measurement" in error
    ):
        return "Unsupported language claim blocked"
    if region.get("vlm_called"):
        return "Description unavailable (model error)"
    return "AI description withheld (language gate not passed)"


def language_gate_reason(region: dict) -> str:
    """Explain the actual language gate without reclassifying crack presence."""
    thresholds = region.get("thresholds", {})
    reasons = []
    scores = region.get("tta_confidences", [])
    if scores:
        missing = sum(score == 0 for score in scores)
        if missing:
            reasons.append(f"region not matched in {missing}/{len(scores)} image checks")
    limit = thresholds.get("variance")
    spread = region.get("variance")
    if limit is not None and spread is not None and spread > limit:
        reasons.append(
            f"score variation {spread * 100:.1f} percentage points exceeds "
            f"the language limit of {limit * 100:.1f}"
        )
    confidence_limit = thresholds.get("confidence")
    confidence = region.get("confidence")
    if confidence_limit is not None and confidence is not None and confidence < confidence_limit:
        reasons.append(f"mean score {confidence:.0%} is below the language limit {confidence_limit:.0%}")
    return "; ".join(reasons) or "language gate not passed"


def summarize(
    image: Image.Image, result: dict[str, Any], policy: dict | None = None
) -> dict:
    policy = policy if policy is not None else operating_point()
    regions = result.get("detections", [])
    present = any(
        r["raw_confidence"] >= policy["raw_confidence_threshold"] for r in regions
    )
    selected = [
        (i, r)
        for i, r in enumerate(regions, 1)
        if r["raw_confidence"] >= policy["localization_threshold"]
    ]
    # Show a bounded preview of weaker proposals without obscuring stronger boxes.
    uncertain = sorted(
        ((i, r) for i, r in enumerate(regions, 1)
         if policy["raw_confidence_threshold"] <= r["raw_confidence"] < policy["localization_threshold"]),
        key=lambda item: -item[1]["raw_confidence"],
    )[:5]
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width = max(2, round(min(image.size) / 200))
    observations = []
    for number, r in uncertain + selected:
        # Draw stronger regions last so weaker proposals cannot hide them.
        permitted = r["raw_confidence"] >= policy["localization_threshold"]
        color = "#e43c3c" if permitted else "#eeeeee"
        x1, y1, x2, y2 = r["box"]
        if permitted:
            draw.rectangle(r["box"], outline=color, width=width)
        else:
            for x in range(round(x1), round(x2), 12):
                draw.line((x, y1, min(x + 6, x2), y1), fill=color, width=width)
                draw.line((x, y2, min(x + 6, x2), y2), fill=color, width=width)
            for y in range(round(y1), round(y2), 12):
                draw.line((x1, y, x1, min(y + 6, y2)), fill=color, width=width)
                draw.line((x2, y, x2, min(y + 6, y2)), fill=color, width=width)
        label = f"R{number} | {r['raw_confidence']:.0%}"
        bounds = draw.textbbox((x1, y1), label)
        draw.rectangle(bounds, fill="#111111")
        draw.text((x1, y1), label, fill=color)
        cx, cy = (x1 + x2) / (2 * image.width), (y1 + y2) / (2 * image.height)
        horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        vertical = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
        text = (
            f"Region {number}: {'crack candidate' if permitted else 'weak candidate; location uncertain'} (detector score {r['raw_confidence']:.0%}). "
            f"Box at {vertical} {horizontal}, spanning {(x2 - x1) / image.width:.0%} of image width "
            f"and {(y2 - y1) / image.height:.0%} of height. "
        )
        if r.get("vlm_succeeded"):
            text += r["explanation"]
        observations.append(text)
    if selected:
        heading = "Crack detector: " + f"{len(selected)} region" + (
            "s" if len(selected) != 1 else ""
        )
        detail = ""
        observations.insert(0, "The detector found crack evidence. The locations below are measured from its boxes, independently of AI-generated descriptions.")
    elif present:
        heading = "Possible crack signal; location uncertain"
        detail = "The image passes the crack-presence threshold, but no region passes the location threshold. Review the photograph."
    else:
        heading = "No confident crack signal"
        detail = "No candidate passes the calibrated crack-presence threshold. This does not prove the surface is crack-free."
    if selected or uncertain:
        detail += (
            f" Solid red: {len(selected)} candidate(s) at or above the {policy['localization_threshold']:.1%} location threshold. "
            f"Dashed white: {len(uncertain)} weaker candidate(s) below it (up to five shown). "
            "Colors reflect detector scores, not AI-description approval. Scores are not probabilities."
        )
        detail = detail.strip()
    hidden = len(regions) - len(selected) - len(uncertain)
    if hidden:
        observations.append(
            f"{hidden} weaker candidate(s) kept in Detailed evidence, omitted from the overlay."
        )
    if not selected:
        observations.insert(0, detail)
    withheld = [(i, r) for i, r in selected if not r.get("vlm_called") and not r.get("vlm_succeeded")]
    if withheld:
        observations.append(
            f"AI description withheld for {len(withheld)} region(s) because the language gate did not pass. "
            "The detector observations above remain available. See Detailed evidence for the exact reasons."
        )
    failed = [(i, r) for i, r in selected if r.get("vlm_called") and not r.get("vlm_succeeded")]
    for number, r in failed:
        observations.append(f"Region {number}: {claim_status(r)}.")
    blocked = sum("claim blocked" in claim_status(r) for r in regions)
    if blocked:
        observations.append(
            f"{blocked} unsupported language claim(s) blocked. A rejected description does not invalidate the crack detection."
        )
    return {
        "image": annotated,
        "observation": "\n\n".join(observations),
        "heading": heading,
        "detail": detail,
        "present": present,
        "selected_ids": [i for i, r in selected],
        "uncertain_ids": [i for i, r in uncertain],
        "hidden_count": hidden,
        "blocked_claims": blocked,
        "policy": policy,
    }
