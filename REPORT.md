# Final Technical Report

## Objective and Outcome

The project implements and validates a hallucination-aware crack-inspection assistant. Its primary safety property is not merely that a detector and language model run, but that uncertain detector behavior prevents language generation. The final system demonstrates a real explanation path, a real withheld path with independently proven VLM non-invocation, and a no-detection short circuit.

Final status: **COMPLETE WITH DOCUMENTED LIMITATIONS**.

## Implementation

`pipeline.py` normalizes an input image to RGB, executes exactly five deterministic YOLOv8 segmentation passes, summarizes top-detection confidence stability, classifies hallucination risk, and conditionally calls Moondream2 on a bounded crop. `inspect(image)` is the only public route that may invoke the VLM. `app.py` is a thin Gradio presentation adapter over that entrypoint; it does not implement a separate gate.

The five passes are ordered as original, horizontal flip, brightness 0.8, contrast 1.15, and rotation +3 degrees. Missing pass detections contribute confidence 0.0. The original pass alone establishes `detected` and supplies the box; augmented passes inform confidence stability only. This avoids using augmented detections without inverse box transforms.

## Uncertainty and Gate

The compatibility result field `variance` is population standard deviation of the five confidence values (`ddof=0`). It is an empirical TTA-based stability measure, not a full Bayesian posterior uncertainty estimate.

Final thresholds are:

```text
confidence_threshold = 0.35
variance_threshold = 0.03
```

The VLM is permitted if and only if the original pass has a valid box and risk is Low. Medium and High cases return `Explanation withheld — flagged for human review`. No-detection cases return `No defect detected.` Risk boundaries use strict comparisons, so equality remains in the lower band.

For a Low case, the original box is expanded 15% on each side, clamped, and cropped. Only that crop reaches Moondream. The generation prompt requests one visible, non-causal observation; output is normalized to one sentence and checked for diagnostic, causal, safety, compliance, material, unsupported measurement, and observed structural-language patterns. A VLM load, generation, crop, or language-guard failure returns an explicit fail-closed message instead of fabricated fallback text.

## Models and Reproducibility

The detector is YOLOv8s segmentation from `OpenSistemas/YOLOv8-crack-seg`, file `yolov8s/weights/best.pt`, pinned revision `910c33b45c040fd8b1cb93eae3fb08ec2ca8b956`. The local 23,861,027-byte checkpoint matches SHA-256 `aceae4a99a4e7903a78d11336afc71b4cb47318888d87279922687bfeb16637d` and is licensed AGPL-3.0.

The VLM is `vikhyatk/moondream2`, revision `2024-08-26`, resolved commit `92d3d73b6fd61ab84d9fe093a9c7fd8c04bf2c0d`. Its cached safetensors weight file matches SHA-256 `4bf7aed8ba4325d23fa7cd348d795a27f3b272682536f08aca4cdd62cde79293`; license is Apache-2.0. The pinned model uses `trust_remote_code` with Transformers 4.44.0.

The validated machine used Windows 11, Python 3.11.9, an Intel i5-1135G7, 15.79 GB RAM, and CPU fp32 because CUDA was unavailable. Automatic CUDA selection remains supported in code.

## Evidence and Results

The authoritative five-image acceptance run classified all three clear fixtures Low, the real ambiguous shadow High, and the synthetic background N/A/no detection. The real VLM succeeded on two clear cases. The third Low case invoked the VLM but its unsupported language was blocked and failed closed. A loader spy proved no VLM call for the High case. Two repeated images had exactly equal five-confidence sequences and unchanged risk labels.

The approved 29-image combined evidence consists of five originals and 24 additional images selected deterministically from the already-integrated Crack-Seg test/validation splits: 12 test and 12 validation images, including the split's sole empty-label frame. Results were:

| Gate outcome | Count |
|---|---:|
| Low / VLM called | 13 |
| Medium or High / VLM withheld | 15 |
| No detection / VLM skipped | 1 |

The expansion independently repeated both Low/called and Medium-or-High/withheld directions. It did not add another no-detection result. Crucially, the empty-label validation frame was detected, classified Low, and explained, documenting a concrete semantic false positive.

Manual ground truth for the original five yielded TP=3, FP=1, TN=1, FN=0: precision 75%, recall 100%, accuracy 80%. **n=5, illustrative only, not a statistically powered evaluation.**

The requested isolated sensitivity grid varied standard-deviation thresholds 0.13, 0.15, 0.17 at confidence 0.35 and confidence thresholds 0.30, 0.35, 0.40 at standard deviation 0.15. No label changed within either narrow grid. However, 15/29 risk labels differ between the shipped 0.03 threshold and requested 0.15 reference; narrow-grid stability must not be read as general threshold insensitivity.

CPU latency results were:

| Stage | n | Mean (s) | Median (s) | P95 (s) |
|---|---:|---:|---:|---:|
| Base detector | 29 | 0.509 | 0.389 | 0.930 |
| Full five-pass TTA | 29 | 2.019 | 1.878 | 3.452 |
| VLM cold start, including load | 1 | 49.149 | — | — |
| Warm VLM inference | 12 | 24.040 | 22.492 | 31.697 |
| Total inspect, all images | 29 | 13.719 | 2.419 | 35.085 |

## Fallbacks and Substitutions

- CPU fp32 was used because no CUDA device existed; no reduced-precision fallback was required.
- Pretrained detector validation passed, so no detector retraining occurred.
- One intended web texture was rate-limited and replaced by a deterministic synthetic no-crack fixture, explicitly labeled synthetic.
- Two unsuitable clear fixtures were replaced through a recorded, deterministic audit of a fixed held-out Crack-Seg test pool; former cases were retained as difficult examples.
- Transformers, Hugging Face Hub, Gradio, and Pillow were pinned to a mutually compatible stack for the selected Moondream revision.
- Unsupported VLM output is blocked rather than edited into a more favorable answer.

## Limitations

1. **Small, selected evidence.** Threshold tuning and core acceptance used five selected images. The 24-image expansion comes from the same integrated Crack-Seg source. Neither is an independent, statistically powered evaluation, and candidate selection can inflate apparent demonstration performance.
2. **Thresholds are demonstration-level.** The 0.03 stability threshold separates the locked examples but is not calibrated across structures, cameras, weather, substrates, or operating environments. The 15/29 label change relative to 0.15 shows its material effect.
3. **Stable errors remain possible.** TTA measures prediction stability, not correctness. The empty-label Low-risk false positive and original shadow false positive show that a detector can be consistently wrong.
4. **Detector scope is narrow.** Only the top region is retained. Small, multiple, occluded, non-surface, or out-of-distribution cracks may be missed, and masks are not used by the gate.
5. **No severity or causality assessment.** The system does not infer structural integrity, crack depth, cause, progression, compliance, or safety. A red box and fluent sentence must not be interpreted as engineering certification.
6. **Language safety is incomplete.** Prompting and a lexical guard reduce known unsupported claims but cannot prove semantic grounding or cover every paraphrase. Guarded failures intentionally reduce availability.
7. **Resource and latency constraints.** CPU fp32 VLM calls are slow and memory intensive; the first call includes a substantial lazy-load cost. Concurrent requests are serialized in the UI.
8. **Reproducibility boundaries.** GPU kernels and library/platform changes can introduce small numeric differences. Model caches and external hosting availability are operational dependencies.
9. **Supply chain and licensing.** The pickle-based `.pt` checkpoint can execute code while loading and is AGPL-3.0. Moondream uses pinned remote code. Distribution and network use require appropriate security and license review.
10. **Dataset labels are not perfect ground truth.** An empty annotation is treated as background for the reported warning, but dataset omissions remain possible. Broader manual expert labeling is needed.

## Verification and Artifacts

Automated tests cover deterministic augmentation, statistics, missing detections, risk boundaries, invalid input, crop clamping, VLM call/no-call paths, one-sentence processing, unsupported-language failures, CLI structures, and UI adaptation. Machine-readable evidence preserves input/model identifiers, thresholds, decision traces, timings, logs, annotations, source hashes, and integrity checks.

Primary artifacts are `pipeline.py`, `app.py`, `README.md`, `REPORT.md`, `DECISIONS.md`, `requirements.txt`, `TEST_RESULTS.md`, `SENSITIVITY.md`, `models/`, `test_images/`, `tests/`, and `evidence/`.

## Conclusion

The project satisfies its central criterion: the gate has been executed in both directions, uncertain/no-detection cases bypass the VLM, real Low cases cross it, and evidence independently verifies non-invocation. Its limitations prevent an unqualified `COMPLETE` designation. The appropriate final status is **COMPLETE WITH DOCUMENTED LIMITATIONS**.
