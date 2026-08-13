"""Additive latency and gate instrumentation around the unchanged inspect entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean, median
from typing import Any
from unittest.mock import patch

import pipeline
from pipeline import PipelineConfig

ORIGINAL_METADATA = {
    "clear_01.jpg": ("original", "clear crack", "Crack-Seg test split"),
    "clear_02.jpg": ("original", "clear crack", "locked original fixture"),
    "clear_03.jpg": ("original", "clear crack", "Crack-Seg test split"),
    "ambiguous_01.jpg": ("original", "ambiguous shadow", "locked original fixture"),
    "ambiguous_02.jpg": ("original", "synthetic background", "locked original fixture"),
}

ORIGINAL_GROUND_TRUTH = {
    "ambiguous_01.jpg": False,
    "ambiguous_02.jpg": False,
    "clear_01.jpg": True,
    "clear_02.jpg": True,
    "clear_03.jpg": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_additional_images(
    dataset_root: Path, output_dir: Path
) -> list[dict[str, Any]]:
    """Select 12 test cracks, 11 val cracks, and the one val background deterministically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    original_hashes = {
        "2a6ed35d9c831614a3b22cba05c7133623fd750de2a794b75c163f9e10c3833d",
        "02beb610e7a256f91c9ab42593d87e4f85db13b75d896ddb65af95cc1b52a82a",
    }
    seen_hashes = set(original_hashes)
    selections: list[dict[str, Any]] = []

    def add_from_split(split: str, target_count: int, *, require_empty: bool) -> None:
        added = 0
        for source in sorted((dataset_root / "images" / split).glob("*")):
            label = dataset_root / "labels" / split / f"{source.stem}.txt"
            empty = not label.is_file() or label.stat().st_size == 0
            if empty != require_empty:
                continue
            digest = sha256(source)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            destination = output_dir / f"{split}_{source.name}"
            shutil.copy2(source, destination)
            selections.append(
                {
                    "image": destination.name,
                    "source_filename": source.name,
                    "split": split,
                    "expected_category": "background (empty dataset label)"
                    if empty
                    else "clear crack (dataset label)",
                    "label_nonempty": not empty,
                    "sha256": digest,
                    "source_path_relative": f"images/{split}/{source.name}",
                    "label_path_relative": f"labels/{split}/{label.name}",
                }
            )
            added += 1
            if added == target_count:
                return
        raise RuntimeError(
            f"Could not select {target_count} qualifying unique images from {split}."
        )

    add_from_split("test", 12, require_empty=False)
    add_from_split("val", 11, require_empty=False)
    add_from_split("val", 1, require_empty=True)
    return selections


@contextmanager
def timed_generation(timing: dict[str, float | None]) -> Iterator[None]:
    original = pipeline._generate_explanation

    def wrapper(*args: Any, **kwargs: Any) -> str:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timing["vlm_call_seconds"] = time.perf_counter() - started

    with patch.object(pipeline, "_generate_explanation", wrapper):
        yield


def instrumented_inspect(
    image: Path, config: PipelineConfig, detector: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    pass_times: list[float] = []
    first_started: float | None = None
    fifth_finished: float | None = None

    def timed_predictor(
        active_detector: Any, augmented: Any, active_config: PipelineConfig
    ) -> Any:
        nonlocal first_started, fifth_finished
        if first_started is None:
            first_started = time.perf_counter()
        started = time.perf_counter()
        prediction = pipeline._predict_top(active_detector, augmented, active_config)
        pass_times.append(time.perf_counter() - started)
        if len(pass_times) == 5:
            fifth_finished = time.perf_counter()
        return prediction

    timing: dict[str, float | None] = {"vlm_call_seconds": None}
    total_started = time.perf_counter()
    with timed_generation(timing):
        result = pipeline.inspect(
            image, config=config, detector=detector, predictor=timed_predictor
        )
    external_total = time.perf_counter() - total_started
    if len(pass_times) != 5 or first_started is None or fifth_finished is None:
        raise RuntimeError(
            f"Expected five timed detector passes for {image.name}, got {len(pass_times)}."
        )
    timing.update(
        {
            "base_detector_seconds": pass_times[0],
            "tta_5_pass_seconds": fifth_finished - first_started,
            "detector_pass_seconds": pass_times,
            "inspect_total_seconds": external_total,
            "pipeline_processing_seconds": result["processing_time"],
        }
    )
    return result, timing


def markdown_table(results: list[dict[str, Any]]) -> str:
    header = (
        "| Image | Set | Expected category | Raw conf. | Calibrated conf. | TTA stddev | Risk | VLM called | "
        "VLM succeeded | Base detector (s) | 5-pass TTA (s) | VLM (s) | Total inspect (s) | Explanation |\n"
        "|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|"
    )
    rows = [header]
    for item in results:
        vlm_time = (
            "—"
            if item["vlm_call_seconds"] is None
            else f"{item['vlm_call_seconds']:.3f}"
        )
        explanation = item["explanation"].replace("|", "\\|").replace("\n", " ")
        rows.append(
            f"| `{item['image']}` | {item['set']} | {item['expected_category']} | "
            f"{item['raw_confidence']:.4f} | {item['confidence']:.4f} | {item['variance']:.4f} | "
            f"{item['hallucination_risk']} | {item['vlm_called']} | {item['vlm_succeeded']} | "
            f"{item['base_detector_seconds']:.3f} | {item['tta_5_pass_seconds']:.3f} | {vlm_time} | "
            f"{item['inspect_total_seconds']:.3f} | {explanation} |"
        )
    return "\n".join(rows)


def latency_row(label: str, values: list[float]) -> str:
    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return (
        f"| {label} | {len(ordered)} | {fmean(ordered):.3f} | "
        f"{median(ordered):.3f} | {p95:.3f} |"
    )


def write_test_results(
    path: Path, results: list[dict[str, Any]], config: PipelineConfig
) -> None:
    low = sum(item["hallucination_risk"] == "Low" for item in results)
    withheld = sum(item["hallucination_risk"] in {"Medium", "High"} for item in results)
    no_detection = sum(not item["detected"] for item in results)
    original = [item for item in results if item["set"] == "original"]
    expanded = [item for item in results if item["set"] == "expanded"]
    original_pattern = (
        sum(item["hallucination_risk"] == "Low" for item in original),
        sum(item["hallucination_risk"] in {"Medium", "High"} for item in original),
        sum(not item["detected"] for item in original),
    )
    expanded_pattern = (
        sum(item["hallucination_risk"] == "Low" for item in expanded),
        sum(item["hallucination_risk"] in {"Medium", "High"} for item in expanded),
        sum(not item["detected"] for item in expanded),
    )
    consistent = expanded_pattern[0] > 0 and (
        expanded_pattern[1] > 0 or expanded_pattern[2] > 0
    )
    base_times = [float(item["base_detector_seconds"]) for item in results]
    tta_times = [float(item["tta_5_pass_seconds"]) for item in results]
    vlm_times = [
        float(item["vlm_call_seconds"])
        for item in results
        if item["vlm_call_seconds"] is not None
    ]
    cold_vlm_time = vlm_times[0]
    warm_vlm_times = vlm_times[1:]
    total_times = [float(item["inspect_total_seconds"]) for item in results]
    called_total_times = [
        float(item["inspect_total_seconds"]) for item in results if item["vlm_called"]
    ]
    skipped_total_times = [
        float(item["inspect_total_seconds"])
        for item in results
        if not item["vlm_called"]
    ]
    background = [
        item for item in expanded if item["expected_category"].startswith("background")
    ]
    background_observation = (
        "The sole empty-label background frame was detected and classified Low risk, "
        "so it invoked the VLM; this is a concrete false-positive warning for human review."
        if background and background[0]["detected"]
        else "The sole empty-label background frame produced no detection and skipped the VLM."
    )
    latency_rows = "\n".join(
        [
            latency_row("Base detector inference", base_times),
            latency_row("Full five-pass TTA", tta_times),
            f"| VLM cold start (first call, includes model load) | 1 | {cold_vlm_time:.3f} | — | — |",
            latency_row("VLM warm inference (calls 2 onward)", warm_vlm_times),
            latency_row("Total inspect (all images)", total_times),
            latency_row("Total inspect (VLM called)", called_total_times),
            latency_row("Total inspect (VLM skipped)", skipped_total_times),
        ]
    )
    original_predictions = {item["image"]: bool(item["detected"]) for item in original}
    true_positive = sum(
        ORIGINAL_GROUND_TRUTH[name] and original_predictions[name]
        for name in ORIGINAL_GROUND_TRUTH
    )
    false_positive = sum(
        not ORIGINAL_GROUND_TRUTH[name] and original_predictions[name]
        for name in ORIGINAL_GROUND_TRUTH
    )
    true_negative = sum(
        not ORIGINAL_GROUND_TRUTH[name] and not original_predictions[name]
        for name in ORIGINAL_GROUND_TRUTH
    )
    false_negative = sum(
        ORIGINAL_GROUND_TRUTH[name] and not original_predictions[name]
        for name in ORIGINAL_GROUND_TRUTH
    )
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / len(ORIGINAL_GROUND_TRUTH)
    text = f"""# Expanded Test Results

This is additive evidence collected between Pass 5 and Pass 6. It does not replace the locked Pass 5 evidence. Every row was executed through the unchanged `inspect()` entrypoint. Stage timings were collected by external wrappers around the detector predictor and VLM generator; no production decision logic was modified.

## Evaluation Configuration

- Shipped thresholds observed at execution: confidence `{config.confidence_threshold}`, TTA standard-deviation `{config.variance_threshold}`.
- Threshold chronology: standard deviation `0.15` was the initial Pass 4/early Pass 5 value; Pass 5 tuning changed and locked it at `0.03` before the authoritative acceptance and this expanded run. Pass 6 did not change it.
- Detector passes: exactly five per image.
- Additional source: the already-integrated Ultralytics Crack-Seg test/validation splits only.
- Combined rows: {len(results)} ({len(original)} original + {len(expanded)} additional).
- The Crack-Seg archive contains one empty-label validation frame; it is included as the additional background case.

## Results

{markdown_table(results)}

## Summary

- Low risk / VLM called: **{low}**
- Medium or High risk / explanation withheld: **{withheld}**
- No detection / VLM skipped: **{no_detection}**
- Original five pattern `(Low, withheld, no-detection)`: **{original_pattern}**
- Additional set pattern `(Low, withheld, no-detection)`: **{expanded_pattern}**
- Pattern consistency: **{"Directionally yes" if consistent else "No"}** — the expanded set {"repeats both Low/called and Medium-or-High/withheld behavior" if consistent else "does not independently exercise both gate directions"}. It did not repeat a no-detection case.
- Background observation: **{background_observation}**

## Latency KPI Summary

All values are wall-clock seconds from this CPU-only run. VLM timing includes lazy model loading on the first call.

| Stage | n | Mean | Median | P95 |
|---|---:|---:|---:|---:|
{latency_rows}

## Small-Sample Gate Accuracy

**n=5, illustrative only, not a statistically powered evaluation.**

The positive class is `crack present = yes`; the pipeline prediction is its unchanged `detected` output.

| Metric | Result |
|---|---:|
| True positives | {true_positive} |
| False positives | {false_positive} |
| True negatives | {true_negative} |
| False negatives | {false_negative} |
| Precision | {precision:.2%} |
| Recall | {recall:.2%} |
| Accuracy | {accuracy:.2%} |
"""
    path.write_text(text, encoding="utf-8")


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
    parser.add_argument("--test-dir", type=Path, default=Path("test_images"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence/expanded"))
    parser.add_argument("--markdown", type=Path, default=Path("TEST_RESULTS.md"))
    args = parser.parse_args()

    evidence_dir = args.evidence_dir
    selected_dir = args.test_dir / "expanded"
    annotated_dir = evidence_dir / "annotated"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    selections = select_additional_images(args.dataset_root, selected_dir)
    (evidence_dir / "selection_manifest.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8"
    )

    config = PipelineConfig()
    detector = pipeline._load_detector(config)
    work: list[tuple[Path, dict[str, Any]]] = []
    for name, (set_name, expected, source) in ORIGINAL_METADATA.items():
        work.append(
            (
                args.test_dir / name,
                {"set": set_name, "expected_category": expected, "source": source},
            )
        )
    for item in selections:
        work.append(
            (
                selected_dir / item["image"],
                {
                    "set": "expanded",
                    "expected_category": item["expected_category"],
                    "source": f"Crack-Seg {item['split']} split",
                },
            )
        )

    results: list[dict[str, Any]] = []
    for index, (image, metadata) in enumerate(work, start=1):
        print(f"[{index}/{len(work)}] {image.name}", flush=True)
        result, timing = instrumented_inspect(image, config, detector)
        result["annotated_image"].save(annotated_dir / image.name)
        record = {
            "image": image.name,
            "input_sha256": sha256(image),
            **metadata,
            **{key: value for key, value in result.items() if key != "annotated_image"},
            **timing,
        }
        results.append(record)
        (evidence_dir / "results.partial.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    (evidence_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_test_results(args.markdown, results, config)
    print(
        json.dumps(
            {"status": "PASS", "rows": len(results), "markdown": str(args.markdown)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
