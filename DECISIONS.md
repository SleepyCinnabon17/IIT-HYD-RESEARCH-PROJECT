# Engineering Decisions

## 2026-08-13 16:09 +05:30 — Use an isolated Python 3.11 environment

- **Decision:** Use `.venv` with Python 3.11.9 and invoke `.venv\\Scripts\\python.exe` explicitly.
- **Reason:** The system `py` launcher defaults to Python 3.13, while the selected ML dependencies are verified in Python 3.11.
- **Alternative considered:** Install into the global Python environment.
- **Result:** All core imports pass and `pip check` reports no broken requirements.
- **Impact:** Commands must activate the virtual environment or use its Python executable.

## 2026-08-13 16:09 +05:30 — Use CPU fp32 execution

- **Decision:** Configure the implementation for CPU and fp32 on this machine while retaining automatic CUDA selection for portable use.
- **Reason:** No CUDA-capable device is available and the installed PyTorch runtime reports no CUDA support.
- **Alternative considered:** Require GPU execution.
- **Result:** Environment setup succeeds without a GPU dependency.
- **Impact:** Detector and especially VLM inference may be slow; models must be lazy-loaded and cached sequentially.

## 2026-08-13 16:09 +05:30 — Select the YOLOv8s crack-segmentation checkpoint

- **Decision:** Use `OpenSistemas/YOLOv8-crack-seg/yolov8s/weights/best.pt`, pinned to revision `910c33b45c040fd8b1cb93eae3fb08ec2ca8b956`.
- **Reason:** The 23.9 MB small checkpoint is a practical CPU-oriented balance between the nano and larger variants.
- **Alternative considered:** Nano for speed or medium/large/xlarge for accuracy.
- **Result:** The targeted checkpoint downloaded successfully and matches SHA-256 `aceae4a99a4e7903a78d11336afc71b4cb47318888d87279922687bfeb16637d`.
- **Impact:** The detector remains subject to genuine inference validation in Pass 3. The AGPL-3.0 license and pickle-based `.pt` format must remain documented.

## 2026-08-13 16:09 +05:30 — Resolve gating and metric ambiguities conservatively

- **Decision:** Call the VLM only for a valid original-pass detection classified Low risk. Medium, High, and no-detection paths bypass it. Preserve the output key `variance` but calculate population standard deviation.
- **Reason:** This interpretation satisfies the safety objective and the approved execution plan while resolving inconsistent wording in the master prompt.
- **Alternative considered:** Permit Medium-risk explanations or compute mathematical variance.
- **Result:** The public contracts and upcoming tests have one unambiguous rule.
- **Impact:** The implementation will fail closed and explicitly document that `variance` is a compatibility field containing TTA confidence standard deviation.

## 2026-08-13 16:24 +05:30 — Accept the pretrained detector after real inference

- **Decision:** Continue with the pinned pretrained YOLOv8s segmentation model; do not trigger fallback training.
- **Reason:** The model loaded as a one-class segmentation model, returned boxes and masks, and detected the obvious road crack at confidence `0.759285`, exceeding the approximate `0.30` validation target.
- **Alternative considered:** Train `yolov8n-seg` for at most eight epochs.
- **Result:** The detector baseline completed for all five evaluation images on CPU.
- **Impact:** Pass 4 can implement the TTA pipeline. Visual review showed false positives on wires, architectural edges, and pavement shadows, supporting the need for uncertainty gating.

## 2026-08-13 16:24 +05:30 — Replace a rate-limited image with a reproducible synthetic fixture

- **Decision:** Use one deterministic synthetic mottled/stained surface as `ambiguous_02.jpg` after repeated Wikimedia HTTP 429 responses.
- **Reason:** Further retries would violate the project's blocker/fallback principle.
- **Alternative considered:** Continue retrying the intended CC0 asphalt texture download.
- **Result:** The fixed-seed fixture contains no deliberate crack and produced no detections at the `0.01` detector floor.
- **Impact:** It is explicitly labeled synthetic and cannot be presented as real-world evidence; the real pavement-shadow photograph remains the primary ambiguous example.

## 2026-08-13 16:24 +05:30 — Use crack-focused derivatives for two clear cases

- **Decision:** Crop two licensed photographs and retain their originals under `test_images/source_images/`.
- **Reason:** Visual review found that the uncropped images' highest-confidence boxes followed a wire or roof edge rather than the visible crack.
- **Alternative considered:** Keep misleading full-scene images because their confidence exceeded the target.
- **Result:** The focused fixtures better test crack behavior while retaining provenance and reproducibility.
- **Impact:** Evaluation remains demonstration-only; `clear_03` still exposes a strong architectural-edge false positive and is a difficult clear case.

## 2026-08-13 16:33 +05:30 — Pin the VLM-compatible Transformers stack

- **Decision:** Pin Transformers `4.44.0`, Hugging Face Hub `0.36.2`, Gradio `5.49.1`, and Pillow `11.3.0`.
- **Reason:** The pinned Moondream2 revision declares Transformers `4.44.0`; Transformers `5.15.0` failed to construct its custom Phi model. Gradio 6 required Hugging Face Hub 1.x, which is incompatible with Transformers 4.44.
- **Alternative considered:** Keep the newest packages and let all Low-risk cases fail closed.
- **Result:** An empty-weight construction probe created the pinned `Moondream` class with 1,867,982,709 meta parameters and verified `encode_image` and `answer_question`; `pip check` and all tests pass.
- **Impact:** The environment uses an older compatible UI stack. Full VLM weights and real generation remain for the Pass 5 end-to-end validation.

## 2026-08-13 16:33 +05:30 — Preserve default thresholds until the tuning pass

- **Decision:** Keep `CONF_THRESHOLD=0.35` and `VARIANCE_THRESHOLD=0.15` in Pass 4 despite the initial real TTA matrix classifying the pavement-shadow case Low risk.
- **Reason:** Pass 5 explicitly owns full-matrix threshold tuning and evidence comparison.
- **Alternative considered:** Silently change the values during implementation.
- **Result:** The implementation is testable against the specified defaults and the tuning need is recorded transparently.
- **Impact:** Pass 5 must tune only these two thresholds and rerun all images before acceptance.

## 2026-08-13 16:42 +05:30 — Add machine-verifiable decision provenance

- **Decision:** Version the result schema and include normalized-image SHA-256, model identifiers, thresholds, uncertainty-metric identity, structured decision checks, a human-readable gate reason, and optional CLI JSON saving.
- **Reason:** The original result fields showed the outcome but did not make each decision independently auditable or easily reproducible.
- **Alternative considered:** Leave provenance only in console logs and project documentation.
- **Result:** Every inspection result now explains the exact gate inputs and decision, while `--save-json` preserves it without image objects.
- **Impact:** Evidence consumers can verify which input, models, thresholds, and rule produced an explanation or refusal.

## 2026-08-13 16:42 +05:30 — Add fail-fast validation and preflight diagnostics

- **Decision:** Validate configuration and detector boxes before inference results are serialized, return CLI failures as JSON, and provide `diagnostics.py` for non-inference environment and checkpoint checks.
- **Reason:** Invalid thresholds, non-finite boxes, malformed metadata, and checkpoint drift could otherwise yield misleading output or tracebacks.
- **Alternative considered:** Rely on downstream libraries to reject invalid state.
- **Result:** Twenty-nine automated tests, lint, formatting, compilation, dependency checks, checkpoint hashing, and a real no-detection CLI smoke test pass.
- **Impact:** The project has a lightweight evaluator-friendly preflight command. It deliberately does not replace real detector/VLM inference validation.

## 2026-08-13 16:50 +05:30 — Correct the detector color input contract before tuning

- **Decision:** Pass PIL RGB images directly to Ultralytics instead of converting them to raw NumPy arrays.
- **Reason:** A controlled comparison proved that Ultralytics interprets array inputs as OpenCV BGR. The prior RGB array changed `clear_03` top confidence from `0.289337` to `0.681581` and changed its box.
- **Alternative considered:** Manually reverse RGB arrays into contiguous BGR arrays.
- **Result:** PIL and file-path inference now agree exactly, while deterministic TTA remains in PIL space.
- **Impact:** The Pass 4 TTA baseline is superseded and cannot be used for threshold tuning; Pass 5 reruns the complete corrected matrix.

## 2026-08-13 16:56 +05:30 — Replace unsuitable clear fixtures using a held-out deterministic audit

- **Decision:** Replace `clear_01` and `clear_03` with the two highest-ranked distinct images from a fixed 20-image pool in the official Crack-Seg test split; retain the former fixtures under `difficult_cases/`.
- **Reason:** With corrected RGB handling, threshold-only tuning could not classify most clear cases Low while rejecting the shadow false positive. Visual inspection showed the former cases had fragmented or irrelevant top detections.
- **Alternative considered:** Tune thresholds around mislabeled or poorly localized examples, or cherry-pick without recording the selection pool.
- **Result:** The selection pool, hashes, labels, five-pass vectors, ranking formula, and all candidate results are stored in `evidence/official_test_candidate_audit.json`.
- **Impact:** The demo set includes three obvious, detector-appropriate clear cracks; limitations still include evaluation-set selection and small-sample tuning.

## 2026-08-13 16:56 +05:30 — Tune only the TTA standard-deviation threshold

- **Decision:** Keep `CONF_THRESHOLD=0.35` and change `VARIANCE_THRESHOLD` from `0.15` to `0.03`.
- **Reason:** In the corrected matrix, all three clear cases have TTA standard deviation at or below `0.018741`, while the real pavement-shadow false positive is `0.051846`. A `0.03` threshold makes the shadow High risk because it exceeds the `1.5 ×` boundary of `0.045`.
- **Alternative considered:** Increase the confidence threshold, which would not separate the former clear and ambiguous cases reliably.
- **Result:** The chosen value provides the required explain/withhold separation with one threshold change.
- **Impact:** The value is demonstration-level tuning on a five-image set and is not a universal calibration result.

## 2026-08-13 17:03 +05:30 — Use the required CPU-fp32 VLM path and guard its language

- **Decision:** Retain Moondream2 CPU fp32 after a successful real load, and reject generated sentences containing causal, diagnostic, safety, compliance, material, or unsupported measurement claims.
- **Reason:** The machine had only about 2.8 GB free physical memory, but the pinned 3,736,040,266-byte weight file still loaded through the required fp32 path using virtual memory. Prompting alone cannot guarantee compliant language.
- **Alternative considered:** Preemptively lower precision or accept any first generated sentence.
- **Result:** A real Low-risk inspection called the pinned VLM and generated a one-sentence visible observation in 52.9 seconds; unsupported-language unit cases fail closed.
- **Impact:** No precision fallback is needed. VLM text remains an observation and is blocked rather than rewritten if it crosses the safety-language boundary.

## 2026-08-13 17:05 +05:30 — Expand the language guard after observing a real unsupported claim

- **Decision:** Treat “structural damage” and “instability” as prohibited generated language and fail closed without rewriting the sentence.
- **Reason:** During the first five-image acceptance run, Moondream described a visible crack and then inferred “potential structural damage or instability,” despite the grounding prompt.
- **Alternative considered:** Keep the output because its detector input was Low risk, or silently remove the unsupported clause.
- **Result:** The exact observed failure is now a regression test and the authoritative evidence run is repeated.
- **Impact:** Detector reliability controls whether the VLM may speak; an independent language guard controls whether its generated sentence may be released.

## 2026-08-13 17:30 +05:30 — Freeze expanded evidence without retuning

- **Decision:** Preserve production thresholds at confidence `0.35` and TTA standard deviation `0.03`; add 24 deterministically selected Crack-Seg test/validation images as evidence only.
- **Reason:** The user approved the Pass 5 gate evidence and explicitly prohibited further tuning, retraining, or new data sources.
- **Alternative considered:** Retune after observing expanded results or add external negative examples.
- **Result:** The 29-image matrix contains 13 Low/called, 15 Medium-or-High/withheld, and one no-detection result. The sole empty-label Crack-Seg frame was a Low-risk false positive.
- **Impact:** The false positive and same-source selection are documented limitations; the expansion reinforces gate direction but does not establish general detector accuracy.

## 2026-08-13 17:36 +05:30 — Keep requested sensitivity analysis isolated

- **Decision:** Recompute risk labels post hoc at requested variance thresholds `0.13/0.15/0.17` and confidence thresholds `0.30/0.35/0.40`, without editing `pipeline.py` or rerunning the VLM.
- **Reason:** Risk classification is deterministic from frozen detector/TTA outputs, and the requested reference grid differs from the shipped threshold.
- **Alternative considered:** Change production configuration to `0.15` for the experiment.
- **Result:** Zero labels changed within either requested narrow range, while 15 of 29 differ between shipped `0.03` and reference `0.15`.
- **Impact:** Documentation must distinguish narrow-grid stability from sensitivity to the actual shipped threshold.

## 2026-08-13 17:48 +05:30 — Report small-sample detector accuracy separately from gate acceptance

- **Decision:** Compare unchanged `detected` outputs against user-supplied labels for the original five and report the full confusion matrix.
- **Reason:** Gate direction and detector correctness are different properties.
- **Alternative considered:** Treat Low risk as the positive prediction or extrapolate accuracy to all 29 dataset labels.
- **Result:** TP=3, FP=1, TN=1, FN=0; precision 75%, recall 100%, accuracy 80%.
- **Impact:** Results carry the caption `n=5, illustrative only, not a statistically powered evaluation` and are not a deployment claim.

## 2026-08-13 18:00 +05:30 — Keep the UI as a presentation-only adapter

- **Decision:** Build `app.py` around `pipeline.inspect()` with no duplicated thresholds or gate logic, serialize UI work at concurrency one, and show risk, refusal/explanation, metrics, five-pass values, and explicit gate evidence.
- **Reason:** One decision implementation prevents UI/pipeline divergence and conservative queuing limits CPU/RAM contention.
- **Alternative considered:** Implement a separate UI inference path or add interactive threshold controls.
- **Result:** The interface builds without model loading and automated adapter tests prove it calls the public entrypoint.
- **Impact:** Users cannot silently alter the validated production thresholds in the UI; the UI remains intentionally minimal.

## Final status — Complete with documented limitations

- **Decision:** Classify the final project as `COMPLETE WITH DOCUMENTED LIMITATIONS`.
- **Reason:** All minimum architecture, gate-direction, evidence, CLI, UI, test, and documentation requirements are implemented, but small selected evidence, demonstrated false positives, threshold dependence, CPU latency, and model supply-chain risks prevent an unqualified completion claim.
- **Result:** Final limitations and reproducibility boundaries are consolidated in `REPORT.md` and summarized in `README.md`.
- **Impact:** The artifact is suitable as an auditable demonstration of hallucination-aware gating, not as a production structural-inspection system.
