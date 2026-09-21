# Sensitivity

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

# Threshold Sensitivity

This is an isolated, post-hoc analysis. It does not edit `pipeline.py`, rerun the VLM, or alter the locked Pass 5 evidence. Detector and five-pass TTA outputs are fixed; only the deterministic risk classifier is recomputed because confidence thresholds do not affect detector inference.

> **Threshold chronology:** standard deviation `0.15` was the initial Pass 4/early Pass 5 value. Pass 5 tuning changed and locked the shipped value at `0.03` before authoritative acceptance; Pass 6 did not change it. The `0.13/0.15/0.17` and `0.30/0.35/0.40` grid below is the exact separately requested sensitivity reference, with the other parameter held at `0.15` or `0.35` as specified. No production threshold was changed by this analysis.

## Per-Image Risk Labels

| Image | V=0.13 (C=0.35) | V=0.15 (C=0.35) | V=0.17 (C=0.35) | C=0.30 (V=0.15) | C=0.35 (V=0.15) | C=0.40 (V=0.15) |
|---|---|---|---|---|---|---|
| `clear_01.jpg` | Low | Low | Low | Low | Low | Low |
| `clear_02.jpg` | Low | Low | Low | Low | Low | Low |
| `clear_03.jpg` | Low | Low | Low | Low | Low | Low |
| `ambiguous_01.jpg` | Low | Low | Low | Low | Low | Low |
| `ambiguous_02.jpg` | N/A | N/A | N/A | N/A | N/A | N/A |
| `test_1616.rf.c868709931a671796794fdbb95352c5a.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1675.rf.e3aa3f8d28d0247ef0284dd46dacc29f.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1686.rf.809fb1b51c607e5cf787e44ef4ddd7b8.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1706.rf.011d213c21ec78896c36728dcbc156f5.jpg` | Medium | Medium | Medium | Medium | Medium | Medium |
| `test_1716.rf.85ea38b36008beaa72c5d8541f734eb0.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1722.rf.38b38f2e833309a4f35bfbf0432dffff.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1794.rf.7a03ca09d05e9e2941f768bc8570cb54.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1804.rf.7e39a73f7b0d1c4bdf8094006e0cf495.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1813.rf.79f1af872f76dd8b46df4a83ed5f54c6.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1818.rf.7e52c1687bfb625e86c54c9d71a66d99.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1819.rf.d2d41865c85e1019dc3e8b9daf73c434.jpg` | Low | Low | Low | Low | Low | Low |
| `test_1839.rf.3495cbba8ee9a583ed0dff8d0e5fc0f7.jpg` | Medium | Medium | Medium | Medium | Medium | Medium |
| `val_1604.rf.7229a9adfa1c9ec285d55c965172ea32.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1605.rf.53a5ea427ceda0b3d6abbc79c64efc36.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1610.rf.3271e8c1058a3a701de2d96b621e9080.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1619.rf.5a42efe0c04aca69843e73fcc3a1a1bc.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1622.rf.da42af8068237feeeaa4db6f8628ccb2.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1629.rf.3ecf969381ffe53960ebb20a47013488.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1631.rf.e30824b70361db2a9e2eb7863269ce86.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1637.rf.e941aa1b84b3d14c3ea9f48867b6676a.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1647.rf.3bab20d4aa37f4fe965fbf2cfe1622fb.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1649.rf.11669e5fb853ededb95501a2bc7e31b9.jpg` | Low | Low | Low | Low | Low | Low |
| `val_1658.rf.38d44ec0dfa198d017ba9cb0ada57f15.jpg` | Low | Low | Low | Low | Low | Low |
| `val_3513.rf.782300f38d0a008b3340e54b643e713e.jpg` | Low | Low | Low | Low | Low | Low |

## Risk Counts

| Configuration | Low | Medium | High | N/A |
|---|---:|---:|---:|---:|
| V=0.13 (C=0.35) | 26 | 2 | 0 | 1 |
| V=0.15 (C=0.35) | 26 | 2 | 0 | 1 |
| V=0.17 (C=0.35) | 26 | 2 | 0 | 1 |
| C=0.30 (V=0.15) | 26 | 2 | 0 | 1 |
| C=0.35 (V=0.15) | 26 | 2 | 0 | 1 |
| C=0.40 (V=0.15) | 26 | 2 | 0 | 1 |

## Interpretation

Across 29 images, **0** change risk label somewhere over variance thresholds 0.13–0.17 and **0** change somewhere over confidence thresholds 0.30–0.40. This indicates how many samples lie near the requested gate boundary; unchanged samples are stable within this narrow reference range. These counts characterize this curated small sample only and do not establish universal calibration. Separately, **15 of 29** labels differ between the shipped variance threshold 0.03 results and the requested 0.15 reference. Therefore, internal stability of the requested grid must not be interpreted as insensitivity to the shipped threshold choice.
