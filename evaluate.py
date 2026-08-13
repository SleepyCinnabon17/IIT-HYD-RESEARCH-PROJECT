"""Run the five-image acceptance matrix and save reproducible gate evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from pipeline import PipelineConfig, inspect, run_detection_with_uncertainty


def _serializable(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "annotated_image"}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {
            "artifact_manifest.json",
            "execution.log",
        }:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append(
            {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": digest}
        )
    manifest = {
        "algorithm": "SHA-256",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def run_evaluation(test_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig()
    images = sorted(test_dir.glob("*.jpg"))
    if len(images) != 5:
        raise RuntimeError(
            f"Expected exactly five top-level JPG fixtures, found {len(images)}."
        )

    rows: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []
    for image_path in images:
        expected = "clear" if image_path.name.startswith("clear") else "ambiguous"
        result = inspect(image_path, config=config)
        result["annotated_image"].save(output_dir / image_path.name)
        serialized = {
            "image": image_path.name,
            "expected_category": expected,
            **_serializable(result),
        }
        _write_json(output_dir / f"{image_path.stem}.json", serialized)
        full_results.append(serialized)
        rows.append(
            {
                "Image": image_path.name,
                "Expected Category": expected,
                "Raw Confidence": result["raw_confidence"],
                "Calibrated Confidence": result["confidence"],
                "TTA Variance": result["variance"],
                "Risk": result["hallucination_risk"],
                "VLM Called": result["vlm_called"],
                "VLM Succeeded": result["vlm_succeeded"],
                "Explanation": result["explanation"],
                "Processing Time": result["processing_time"],
            }
        )

    _write_json(output_dir / "test_results.json", full_results)
    with (output_dir / "test_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    low_results = [item for item in full_results if item["hallucination_risk"] == "Low"]
    high_results = [
        item for item in full_results if item["hallucination_risk"] == "High"
    ]
    no_detection = [item for item in full_results if not item["detected"]]
    clear_results = [
        item for item in full_results if item["expected_category"] == "clear"
    ]
    if sum(item["hallucination_risk"] == "Low" for item in clear_results) < 2:
        raise RuntimeError("Acceptance failed: most clear fixtures are not Low risk.")
    if not any(item["vlm_succeeded"] for item in low_results):
        raise RuntimeError(
            "Acceptance failed: no real Low-risk VLM explanation succeeded."
        )
    if not high_results or any(item["vlm_called"] for item in high_results):
        raise RuntimeError(
            "Acceptance failed: High-risk withholding is not demonstrated."
        )
    if not no_detection or any(item["vlm_called"] for item in no_detection):
        raise RuntimeError(
            "Acceptance failed: no-detection short-circuit is not demonstrated."
        )

    explained = next(item for item in low_results if item["vlm_succeeded"])
    withheld = high_results[0]
    no_defect = no_detection[0]
    for alias, item in (
        ("explained_example", explained),
        ("withheld_example", withheld),
        ("no_detection_example", no_defect),
    ):
        shutil.copy2(output_dir / item["image"], output_dir / f"{alias}.jpg")
        _write_json(output_dir / f"{alias}.json", item)

    loader_calls = 0

    def forbidden_loader(_config: PipelineConfig) -> tuple[Any, Any]:
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("VLM loader crossed a forbidden gate")

    spy_started = time.perf_counter()
    spy_result = inspect(
        test_dir / withheld["image"], config=config, vlm_loader=forbidden_loader
    )
    spy_record = {
        "image": withheld["image"],
        "risk": spy_result["hallucination_risk"],
        "vlm_loader_calls": loader_calls,
        "vlm_called": spy_result["vlm_called"],
        "explanation": spy_result["explanation"],
        "elapsed": time.perf_counter() - spy_started,
        "passed": loader_calls == 0 and not spy_result["vlm_called"],
    }
    _write_json(output_dir / "gate_spy.json", spy_record)
    if not spy_record["passed"]:
        raise RuntimeError(
            "Acceptance failed: real-detector High-risk spy crossed the VLM boundary."
        )

    determinism_records: list[dict[str, Any]] = []
    for name in (explained["image"], withheld["image"]):
        first = run_detection_with_uncertainty(test_dir / name, config=config)
        second = run_detection_with_uncertainty(test_dir / name, config=config)
        deltas = [
            abs(a - b)
            for a, b in zip(
                first["tta_confidences"], second["tta_confidences"], strict=True
            )
        ]
        passed = (
            max(deltas, default=0.0) <= 1e-9
            and first["hallucination_risk"] == second["hallucination_risk"]
        )
        determinism_records.append(
            {
                "image": name,
                "first_tta_confidences": first["tta_confidences"],
                "second_tta_confidences": second["tta_confidences"],
                "absolute_deltas": deltas,
                "first_risk": first["hallucination_risk"],
                "second_risk": second["hallucination_risk"],
                "tolerance": 1e-9,
                "passed": passed,
            }
        )
    _write_json(output_dir / "determinism.json", determinism_records)
    if not all(item["passed"] for item in determinism_records):
        raise RuntimeError(
            "Acceptance failed: repeated detector decisions were not deterministic."
        )

    summary = {
        "status": "PASS",
        "confidence_threshold": config.confidence_threshold,
        "variance_threshold": config.variance_threshold,
        "clear_low_count": sum(
            item["hallucination_risk"] == "Low" for item in clear_results
        ),
        "clear_count": len(clear_results),
        "real_explanation_example": explained["image"],
        "real_withheld_example": withheld["image"],
        "no_detection_example": no_defect["image"],
        "vlm_attempt_count": sum(item["vlm_called"] for item in full_results),
        "vlm_success_count": sum(item["vlm_succeeded"] for item in full_results),
        "language_guard_blocked_images": [
            item["image"]
            for item in full_results
            if item["vlm_called"] and not item["vlm_succeeded"] and item["vlm_error"]
        ],
        "real_detector_gate_spy_passed": spy_record["passed"],
        "determinism_passed": all(item["passed"] for item in determinism_records),
    }
    _write_json(output_dir / "acceptance_summary.json", summary)
    write_artifact_manifest(output_dir)
    print(json.dumps(summary, indent=2))
    return full_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("test_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/final"))
    args = parser.parse_args()
    run_evaluation(args.test_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
