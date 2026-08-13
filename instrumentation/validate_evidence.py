"""Verify expanded evidence provenance, structure, gate invariants, and integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_PIPELINE_SHA256 = (
    "9c222780f0730fb43129411c14071a397bbc8564a9dffd5dc3e027769f82c423"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home()
        / ".cache"
        / "hallucination-aware-visual-inspection"
        / "crack-seg",
    )
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence/expanded"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    evidence_dir = args.evidence_dir
    output = args.output or evidence_dir / "validation_summary.json"
    results: list[dict[str, Any]] = json.loads(
        (evidence_dir / "results.json").read_text(encoding="utf-8")
    )
    selections: list[dict[str, Any]] = json.loads(
        (evidence_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []

    require(len(results) == 29, "Expected 29 combined result rows.", failures)
    require(
        sum(item["set"] == "original" for item in results) == 5,
        "Expected five original rows.",
        failures,
    )
    require(
        sum(item["set"] == "expanded" for item in results) == 24,
        "Expected 24 expanded rows.",
        failures,
    )
    require(len(selections) == 24, "Expected 24 selection records.", failures)
    require(
        sum(item["split"] == "test" for item in selections) == 12,
        "Expected 12 test-split selections.",
        failures,
    )
    require(
        sum(item["split"] == "val" for item in selections) == 12,
        "Expected 12 validation-split selections.",
        failures,
    )
    require(
        sum(not item["label_nonempty"] for item in selections) == 1,
        "Expected exactly one empty-label selection.",
        failures,
    )
    require(
        len({item["sha256"] for item in selections}) == 24,
        "Selection contains duplicate image hashes.",
        failures,
    )

    for item in selections:
        copied = Path("test_images/expanded") / item["image"]
        source = args.dataset_root / item["source_path_relative"]
        require(source.is_file(), f"Dataset source missing: {source}", failures)
        require(copied.is_file(), f"Selected copy missing: {copied}", failures)
        if source.is_file():
            require(
                sha256(source) == item["sha256"],
                f"Dataset source hash mismatch: {item['image']}",
                failures,
            )
        if copied.is_file():
            require(
                sha256(copied) == item["sha256"],
                f"Selected copy hash mismatch: {item['image']}",
                failures,
            )

    annotated_hashes: dict[str, str] = {}
    for item in results:
        image = item["image"]
        require(
            len(item["detector_pass_seconds"]) == 5,
            f"Expected five pass timings: {image}",
            failures,
        )
        expected_call = bool(item["detected"] and item["hallucination_risk"] == "Low")
        require(
            item["vlm_called"] == expected_call,
            f"Gate-call invariant failed: {image}",
            failures,
        )
        require(
            item["vlm_called"] == (item["vlm_call_seconds"] is not None),
            f"VLM timer invariant failed: {image}",
            failures,
        )
        require(
            all(
                float(item[field]) >= 0.0
                for field in (
                    "base_detector_seconds",
                    "tta_5_pass_seconds",
                    "inspect_total_seconds",
                )
            ),
            f"Negative timing found: {image}",
            failures,
        )
        annotated = evidence_dir / "annotated" / image
        require(annotated.is_file(), f"Annotated evidence missing: {image}", failures)
        if annotated.is_file():
            annotated_hashes[image] = sha256(annotated)

    test_results = Path("TEST_RESULTS.md")
    sensitivity = Path("SENSITIVITY.md")
    require(test_results.is_file(), "TEST_RESULTS.md is missing.", failures)
    require(sensitivity.is_file(), "SENSITIVITY.md is missing.", failures)
    if test_results.is_file():
        require(
            sum(
                line.startswith("| `")
                for line in test_results.read_text(encoding="utf-8").splitlines()
            )
            == 29,
            "TEST_RESULTS.md does not contain 29 image rows.",
            failures,
        )
    if sensitivity.is_file():
        require(
            sum(
                line.startswith("| `")
                for line in sensitivity.read_text(encoding="utf-8").splitlines()
            )
            == 29,
            "SENSITIVITY.md does not contain 29 image rows.",
            failures,
        )

    pipeline_hash = sha256(Path("pipeline.py"))
    require(
        pipeline_hash == EXPECTED_PIPELINE_SHA256,
        "pipeline.py differs from its pre-instrumentation SHA-256.",
        failures,
    )
    artifact_paths = [
        Path("pipeline.py"),
        test_results,
        sensitivity,
        evidence_dir / "results.json",
        evidence_dir / "selection_manifest.json",
        Path("evidence/expanded_execution.log"),
    ]
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "pipeline_unchanged": pipeline_hash == EXPECTED_PIPELINE_SHA256,
        "pipeline_sha256": pipeline_hash,
        "result_rows": len(results),
        "original_rows": sum(item["set"] == "original" for item in results),
        "expanded_rows": sum(item["set"] == "expanded" for item in results),
        "selection": {
            "test": sum(item["split"] == "test" for item in selections),
            "val": sum(item["split"] == "val" for item in selections),
            "empty_label": sum(not item["label_nonempty"] for item in selections),
        },
        "gate_counts": {
            "low_vlm_called": sum(
                item["hallucination_risk"] == "Low" and item["vlm_called"]
                for item in results
            ),
            "medium_high_withheld": sum(
                item["hallucination_risk"] in {"Medium", "High"}
                and not item["vlm_called"]
                for item in results
            ),
            "no_detection_skipped": sum(
                not item["detected"] and not item["vlm_called"] for item in results
            ),
        },
        "annotated_count": len(annotated_hashes),
        "annotated_sha256": annotated_hashes,
        "artifact_sha256": {
            str(path).replace("\\", "/"): sha256(path)
            for path in artifact_paths
            if path.is_file()
        },
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "annotated_sha256"},
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
