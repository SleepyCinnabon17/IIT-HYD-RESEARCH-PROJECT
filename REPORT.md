# Report

<!-- V3 RESULTS START -->
## V3 operating-point evaluation - September 21, 2026

**Release decision:** ship the calibrated box overlay and separate detection/language statuses. Keep image-level presence at 0.01. The 0.015 presence trial below was rejected: fresh accuracy improved slightly but the older external benchmark and sensitivity regressed. Deployed image-level metrics therefore remain the Before values. No detector weights or language gate were weakened.

| Metric | Before (V2 policy on same images) | Trial (presence) / deployed (overlay) | Delta |
|---|---:|---:|---:|
| fresh_test, n=200: precision | 67.15% | 71.07% | +3.92 pp |
| fresh_test, n=200: recall | 92.00% | 86.00% | -6.00 pp |
| fresh_test, n=200: f1 | 77.64% | 77.83% | +0.19 pp |
| fresh_test, n=200: accuracy | 73.50% | 75.50% | +2.00 pp |
| v2_evaluation, n=200: precision | 99.50% | 99.50% | +0.00 pp |
| v2_evaluation, n=200: recall | 100.00% | 100.00% | +0.00 pp |
| v2_evaluation, n=200: f1 | 99.75% | 99.75% | +0.00 pp |
| v2_evaluation, n=200: accuracy | 99.50% | 99.50% | +0.00 pp |
| v2_external, n=200: precision | 69.63% | 70.25% | +0.62 pp |
| v2_external, n=200: recall | 94.00% | 85.00% | -9.00 pp |
| v2_external, n=200: f1 | 80.00% | 76.92% | -3.08 pp |
| v2_external, n=200: accuracy | 76.50% | 74.50% | -2.00 pp |
| Box localization precision, IoU >=0.5 | 12.62% | 68.28% | +55.66 pp |
| Box localization recall, IoU >=0.5 | 91.24% | 78.88% | -12.35 pp |
| Box localization f1, IoU >=0.5 | 22.17% | 73.20% | +51.03 pp |
| Drawn boxes / Crack-Seg image | 9.07 | 1.45 | -7.62 |

These are measured detector operating-point changes, not retrained weights or a claim that all hallucinations are identifiable. Every candidate at the original 0.01 floor still receives its independent five-pass TTA gate and remains in the audit. Only the overlay uses the new deployed threshold; image-level presence retains the old sensitive threshold. The existing language gate and grounding checks remain unchanged.

**Frozen trial thresholds:** crack presence `0.015`; displayed localization `0.305`. Presence maximizes accuracy on a NEW balanced 200-image SDNET calibration split subject to >=85% recall when feasible. The initial target was infeasible (81% baseline calibration recall); a documented calibration-only amendment limits sensitivity loss to 5 percentage points instead. Localization maximizes box F1 on the original disjoint 111-image Crack-Seg calibration split subject to >=75% ground-truth-box recall. Ties favor recall, then a lower threshold. Test scores never choose thresholds. See the complete curves and hashed manifest in `evidence/v3/`.

**Independent checks:** a NEW 200-image SDNET test split (100 cracked / 100 non-cracked), the unchanged previous 200-image SDNET benchmark, and the unchanged 200-image Crack-Seg benchmark. All 400 new images had fresh detector inference. Prior 400 test-image detector outputs were replayed exactly; no new VLM accuracy result is claimed. Seed 20260921; selected bytes are disjoint from prior evaluation/calibration and known Crack-Seg training images. SDNET is now calibration data, so call these domain-adapted results, not validation on an untouched external domain. Parent-scene independence and unseen pretraining exposure remain unverified. Crack-Seg still has only one negative and its accuracy is not a useful general false-positive estimate.

### Confidence intervals and tradeoffs

| Dataset | After metric | Estimate | Conservative 95% CI |
|---|---|---:|---|
| fresh_test | precision | 71.07% | [62.11%, 79.08%] |
| fresh_test | recall | 86.00% | [76.38%, 92.80%] |
| fresh_test | f1 | 77.83% | [68.51%, 85.39%] |
| fresh_test | accuracy | 75.50% | [64.89%, 84.12%] |
| v2_evaluation | precision | 99.50% | [99.49%, 99.99%] |
| v2_evaluation | recall | 100.00% | [97.82%, 100.00%] |
| v2_evaluation | f1 | 99.75% | [98.65%, 100.00%] |
| v2_evaluation | accuracy | 99.50% | [97.33%, 99.99%] |
| v2_external | precision | 70.25% | [61.23%, 78.34%] |
| v2_external | recall | 85.00% | [75.21%, 92.06%] |
| v2_external | f1 | 76.92% | [67.50%, 84.65%] |
| v2_external | accuracy | 74.50% | [63.79%, 83.30%] |

fresh_test: paired accuracy change +2.00 pp (95% paired bootstrap [-2.00, +6.00] pp); FP 45 -> 35, FN 8 -> 14.

v2_evaluation: paired accuracy change +0.00 pp (95% paired bootstrap [+0.00, +0.00] pp); FP 1 -> 1, FN 0 -> 0.

v2_external: paired accuracy change -2.00 pp (95% paired bootstrap [-5.50, +1.50] pp); FP 41 -> 36, FN 6 -> 15.

Narrower overlays sacrifice some box recall; omitted candidates remain in Detailed evidence. An image can have a possible crack signal without a reliable location to draw. Solid red boxes mean the detector passes the existing stability gate; dashed white boxes mean candidate detections with uncertain language support. A blocked language claim is separate from detector evidence. Scores are not calibrated probabilities, and automated grounding cannot verify every visual assertion. No structural safety assessment is provided.

### Reproduction

```powershell
python evaluate_v3.py prepare
python evaluate_v3.py infer
python evaluate_v3.py calibrate
python evaluate_v3.py report
```

Use the existing Crack-Seg and SDNET dataset paths documented for V2. `infer` verifies image hashes and caches one original detector pass per image, sufficient for this raw-score operating-point study. It does not substitute a one-pass language gate: deployed inference still performs five passes for every base candidate. Historical inference source and provenance are preserved; the current report source hash is recorded separately. The fresh test and old benchmark estimates are descriptive patch-level results, not guarantees on user road photographs.

<!-- V3 RESULTS END -->

| Metric | Old (n=5/29) | New (n=200 + external n=200) | Delta |
|---|---|---|---|
| Precision | 75.00% (n=5) | Crack-Seg 99.50%; SDNET 69.63% | In-domain vs pilot +24.50 pp; domain gap -29.87 pp |
| Recall | 100.00% (n=5) | Crack-Seg 100.00%; SDNET 94.00% | In-domain vs pilot +0.00 pp; domain gap -6.00 pp |
| F1 | 85.71% (n=5) | Crack-Seg 99.75%; SDNET 80.00% | In-domain vs pilot +14.04 pp; domain gap -19.75 pp |
| Accuracy | 80.00% (n=5) | Crack-Seg 99.50%; SDNET 76.50% | In-domain vs pilot +19.50 pp; domain gap -23.00 pp |
| Regions handled/image | 1.000 (old algorithm replay, n=200) | 9.075 | +8.075 |
| Correctly localized regions/image | 0.880 (old algorithm replay, n=200) | 1.145 | +0.265 |
| Grounding hallucination recall | 20.00% (word-list replay, n=30) | 100.00% | +80.00 pp |
| TTA-std threshold | 0.030000 | 0.016384 | -0.013616 |
| Stability ROC-AUC / PR-AUC (AP) | Not measured | 0.3788 / 0.1430 | Newly measured |

## V2 Evaluation — measured results and limits

**Task 1 is not fully satisfiable with this checkpoint and dataset.** The source has 3,717 training, 200 validation and 112 test images. Only one image has an empty label, in validation, and that image was already used in the n=29 pilot. Validation also participated in checkpoint selection. The new 200-image benchmark is a seeded, stratified sample of non-training partitions (199 positive, 1 negative), **not 200 newly untouched held-out images**. No training images or invented negatives were added to meet the count. The pilot is superseded for descriptive benchmark reporting, but its historical records remain below.

The remaining 111 unique non-training images calibrate the gate and are disjoint from the 200 scoring images. One exact duplicate was excluded before sampling. Selection uses labels, seed 20260907 and no detector scores. Bootstrap CIs condition on the observed class proportions; the lone in-domain negative cannot establish useful real-world false-positive performance, regardless of narrow aggregate intervals. A genuinely fresh balanced test set requires additional annotations/data or retraining a checkpoint on a newly partitioned corpus.

External validation uses 200 randomly sampled SDNET2018 patches (100 cracked, 100 non-cracked). [The original authors](https://digitalcommons.usu.edu/all_datasets/48/) publish SDNET2018 under CC BY 4.0: Maguire, Dorafshan & Thomas (2018), Utah State University, DOI 10.15142/T3TD19. The official binary endpoint returned HTTP 403; the public Kaggle mirror is recorded in the manifest, while the original license takes precedence over the mirror's CC0 tag. This is a separate source dataset, and selected exact-byte training overlaps are excluded. The checkpoint's full source-image lineage is unavailable: **never-seen status cannot be guaranteed against resized/cropped derivatives or pretraining exposure**. SDNET patches can share parent photographs, so image-level intervals can understate scene-level uncertainty.

### Detection metrics

Positive prediction means at least one original-pass box at the unchanged 0.01 inference floor, matching the pilot's `detected` definition. These are image-level crack-presence metrics, not segmentation IoU or language factuality. JSON includes 5,000 stratified image bootstrap resamples; these degenerate to zero width for an all-positive predictor. The primary intervals below instead propagate simultaneous 97.5% exact binomial intervals for sensitivity and specificity through each metric at the fixed sampled class prevalence (Bonferroni gives at least 95% joint coverage under independent images). This also provides a nondegenerate F1 interval. Class prevalence differs sharply between datasets; the domain gap includes this prevalence effect.

| Dataset | Metric | Estimate | Conservative 95% CI | Full CI width |
|---|---|---:|---|---:|
| evaluation | precision | 99.50% | [99.49%, 99.99%] | 0.50% |
| evaluation | recall | 100.00% | [97.82%, 100.00%] | 2.18% |
| evaluation | f1 | 99.75% | [98.65%, 100.00%] | 1.35% |
| evaluation | accuracy | 99.50% | [97.33%, 99.99%] | 2.66% |
| external | precision | 69.63% | [62.11%, 76.58%] | 14.47% |
| external | recall | 94.00% | [86.36%, 98.09%] | 11.73% |
| external | f1 | 80.00% | [72.26%, 86.01%] | 13.76% |
| external | accuracy | 76.50% | [66.84%, 84.05%] | 17.21% |

evaluation: TP=199, FP=1, TN=0, FN=0. At confidence >=0.35, precision/recall/F1 = 99.49%/97.49%/98.48%.

Exact binomial 95% intervals (image-level independence assumed): precision [97.25%, 99.99%]; recall [98.16%, 100.00%]; specificity [0.00%, 97.50%]; accuracy [97.25%, 99.99%].

external: TP=94, FP=41, TN=59, FN=6. At confidence >=0.35, precision/recall/F1 = 87.72%/50.00%/63.69%.

Exact binomial 95% intervals (image-level independence assumed): precision [61.13%, 77.24%]; recall [87.40%, 97.77%]; specificity [48.71%, 68.74%]; accuracy [70.00%, 82.19%].

### Calibration and false-explanation proxy

Trust means a unique detection matches a ground-truth polygon's enclosing box at IoU >=0.5, with greedy one-to-one matching. It is an operational localization proxy, not a human label of language truth and not the old risk label. Calibration contains 771 detections, 126 trustworthy. Stability alone uses score `-TTA_std`; ROC-AUC=0.3788, PR-AUC (average precision)=0.1430. AUC-PR is reported as average precision rather than trapezoidal area.

Selection maximizes recall of trustworthy regions subject to <=5% empirical false detections among Low gates; lower false fraction and then smaller threshold break ties. The confidence floor for Low remains 0.35. ROC/PR arrays and every candidate operating point are in `evidence/v2/calibration_curve.json`. Chosen std threshold: **0.016384**. The deployable artifact is `models/gate_calibration.json`; default CLI/UI inspection checks its detector hash before use. Medium/High retain the existing 1.5x std boundary, which is a policy convention, not a calibrated probability.

| Sample / operating point | Low regions | False Low | False-explanation proxy | True-Low recall | One-sided 95% binomial upper bound |
|---|---:|---:|---:|---:|---:|
| legacy_top_calibration | 75 | 6 | 8.00% | 69.70% | 15.18% |
| old | 68 | 7 | 10.29% | 48.41% | 18.47% |
| chosen | 42 | 2 | 4.76% | 31.75% | 14.24% |
| legacy_top_evaluation | 133 | 10 | 7.52% | 69.89% | 12.42% |
| evaluation_old | 108 | 8 | 7.41% | 43.67% | 12.97% |
| evaluation_chosen | 60 | 3 | 5.00% | 24.89% | 12.42% |

`legacy_top_*` replays the original highest-confidence-per-pass algorithm at 0.03. `old`/`evaluation_old` apply 0.03 to the new matched per-region sequences, isolating the threshold change. The denominators differ between top-only and all-region policies.

| AUC population | Regions | Trustworthy prevalence | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|---:|---:|
| calibration_all_auc | 771 | 16.34% | 0.3788 | 0.1430 |
| calibration_confidence_eligible_auc | 137 | 76.64% | 0.7113 | 0.8935 |
| evaluation_all_auc | 1815 | 12.62% | 0.3458 | 0.0950 |
| evaluation_confidence_eligible_auc | 255 | 72.55% | 0.7098 | 0.8788 |

![Calibration ROC and PR curves](evidence/v2/calibration_curves.png)

Stability alone ranks trust worse than chance across all floor-level proposals. Very low-confidence false proposals can also have very low variance. The confidence-conditioned AUC is stronger, supporting the combined confidence-and-stability gate rather than treating TTA std as a calibrated trust probability.

**No <=5% at 95% confidence guarantee is claimed.** Selection-set binomial bounds are descriptive after threshold search, boxes cluster within images, and the evaluation includes model-selection exposure. Even an independent zero-error sample needs at least 59 accepted independent cases for a one-sided exact 95% upper bound <=5%. The stretch KPI is not achieved. Calibration is never fitted on SDNET2018.

### Per-detection coverage

Five image-level passes are shared for efficiency. Every base box at the 0.01 floor gets its own matched confidence sequence, risk, crop and language decision. Boxes are inverse-transformed before greedy highest-IoU matching (minimum IoU 0.3), each augmented box is used at most once, and missing matches contribute zero. Secondary regions appear in JSON, numbered annotations and the UI. Image risk is the highest region risk; scalar confidence/box fields retain top-region compatibility.

| Set | Old handled/image | New handled/image | Multi-detection images |
|---|---:|---:|---:|
| evaluation | 1.000 | 9.075 | 183/200 |
| external | 0.675 | 3.160 | 102/200 |

Correctly localized regions on the 200-image benchmark increase from 176 (0.880/image) to 229 (1.145/image), against 251 labeled regions. Processing all predictions does not imply all are correct. SDNET has image labels, so external per-region correctness is unavailable.

### Grounding verifier

The verifier uses crop-relative bounding-box center, width, height, area, aspect ratio and TTA-mean confidence. Explicit location/orientation and percentage claims are checked before surfacing. The old diagnostic blocklist remains in force. Contradictions fail closed. Box orientation is only a geometric proxy; branching, appearance, severity, vague paraphrases and all general semantic hallucinations are not certified. The checker records which claims were checked, including an empty list when none were measurable.

The fixed 30-output challenge set contains 15 supported and 15 hallucinated outputs across three geometry templates. Old guard: precision 100.00%, recall 20.00%, F1 33.33%. New verifier: precision 100.00%, recall 100.00%, F1 100.00%. These are real executions on authored adversarial text, **not estimates of natural Moondream hallucination prevalence**; template correlation and narrow vocabulary limit generalization. All prompts, labels, measurements and verdicts are in `grounding_benchmark.json`.

### TTA diversity and pass count

The existing five passes were already diverse in kind: original, horizontal flip, brightness 0.8, contrast 1.15 and rotation +3 degrees. The new audit adds vertical flip, scale 0.85/1.15, color 0.5/1.5, brightness 1.2, contrast 0.85 and rotations -3/+7/-7. Deterministic transforms are reproducible perturbations, not independent posterior samples. Color/contrast can be weak on gray or uniform surfaces. Normalized pixel-change measurements for every transform are in `results.json`.

Thirty randomly sampled images (245 base detections) use a nested 15-pass sequence. Every N is compared on the same regions, with missing matches retained as zero. The reference is N=15, not the true uncertainty.

| N | Mean std | Mean absolute error vs N=15 | Relative MAE vs N=15 |
|---:|---:|---:|---:|
| 3 | 0.02496 | 0.03556 | 63.26% |
| 5 | 0.03864 | 0.02248 | 39.98% |
| 7 | 0.04858 | 0.01772 | 31.51% |
| 8 | 0.04644 | 0.01717 | 30.53% |
| 10 | 0.05137 | 0.01095 | 19.48% |
| 15 | 0.05622 | 0.00000 | 0.00% |

First tested N below 10% relative MAE (excluding the reference itself): **not reached before N=15**. Production retains N=5 because this is the evaluated/calibrated operating point; the curve does not by itself establish a better gate at larger N. This is a latency/calibration choice, not a claim that five passes have converged. **The convergence target is not achieved.** Increasing production N requires recalibrating and scoring the entire gate at that N; N=15 agreeing with itself is not evidence of convergence.

![TTA stability curve](evidence/v2/tta_curve.png)

### Real external language-stage execution

All 200 external images passed through `inspect()` using replayed, hashed real detector results and the real pinned Moondream2 loader. There were 5 region-level VLM calls across 5 images, 3 surfaced explanations and 2 failed-closed outputs. Negative images incorrectly gated Low: 0. Language-stage wall time (including loading) was 405.6s; cached detector times are recorded separately. No stub VLM was used for this evaluation.

### Reproduction and artifacts

Use the pinned dependencies in `requirements.txt`. Dataset files are ignored by Git. Crack-Seg is the existing cached [official archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip); extract it under `~/.cache/hallucination-aware-visual-inspection/` or override `--crack-root`. Selected image and label hashes are in the manifest. Override `--external-root` for other locations. SDNET's downloaded archive must match the pinned checksum before extraction.

```powershell
.venv/Scripts/python.exe evaluate_v2.py download
.venv/Scripts/python.exe evaluate_v2.py prepare
.venv/Scripts/python.exe evaluate_v2.py detect
.venv/Scripts/python.exe evaluate_v2.py summarize
.venv/Scripts/python.exe evaluate_v2.py external-vlm
.venv/Scripts/python.exe evaluate_v2.py deploy
.venv/Scripts/python.exe evaluate_v2.py report
.venv/Scripts/python.exe -m pytest -q
```

`prepare` recreates the seeded split; `detect` resumes only matching provenance and refuses mixed caches. Use a new `--output` directory after changing inference code/configuration. Calibration and results are derived from cached actual detector outputs. `external-vlm` resumes only matching calibration. Tests use fake predictors/VLMs to isolate boundary behavior, distinct from the real evaluation records.

Split manifest SHA-256: `49eea8323f1761e72135045d26b5dec38bb2ba5489342f0d8bac3056054c14a1`. Machine-readable results: `evidence/v2/results.json`; detector/model/config provenance: `inference_provenance.json`.

The additional external overlap screen compared 200 SDNET patches with 3717 training images using a 63-bit grayscale DCT hash, including horizontal mirrors. There were 0 candidates at Hamming distance <=4; the minimum distance was 12. This is supporting evidence, not proof against crops or unknown pretraining. Reproduce with `python evaluate_v2.py overlap`; full nearest-neighbor records are in `external_overlap_audit.json`.

The exact inference source snapshot is retained in `evidence/v2/source/pipeline.py`. Cache resume permits comment/formatting-only changes when parsed executable ASTs match; semantic changes require a new output directory. The original executed-source hash remains in inference provenance.

Final pytest verification: **66 passed in 59.98s**. The saved log is `evidence/v2/pytest_final.log`.

Artifact integrity audit: **PASS**, checking all 511 input hashes, external gate boundaries, installed calibration and preserved pilot sections. See `final_integrity_check.json`.

---

## Pilot Evaluation (superseded)

The following is preserved verbatim from the pre-v2 report, including its historical status statements.

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
