# Final Artifact Checklist

Verified on 2026-08-13 in the documented Python 3.11 CPU environment.

## Core Implementation

- [x] `pipeline.py`: public `inspect()` entrypoint, five deterministic detector passes, uncertainty classification, strict VLM gate, crop, language guard, CLI.
- [x] `app.py`: Gradio upload UI, annotated image, risk indicator, metrics, gate evidence, explanation/withholding display.
- [x] `diagnostics.py`: non-inference preflight and checkpoint integrity checks.
- [x] `evaluate.py` and `instrumentation/`: locked acceptance runner and additive evidence tooling.
- [x] `tests/`: 38 automated tests passing.

## Models and Configuration

- [x] Detector checkpoint present at `models/crack_yolov8s_best.pt`.
- [x] Detector SHA-256 verified as `aceae4a99a4e7903a78d11336afc71b4cb47318888d87279922687bfeb16637d`.
- [x] Detector provenance recorded in `models/MODEL_SOURCE.json`.
- [x] VLM provenance and cache metadata recorded in `models/VLM_SOURCE.json`.
- [x] Final thresholds frozen at confidence `0.35` and TTA standard deviation `0.03`.
- [x] No retraining performed; pretrained validation passed.

## Gate Acceptance and Evidence

- [x] Low-risk real cases invoke the VLM and can produce a grounded sentence.
- [x] Medium/High cases withhold explanation and do not invoke the VLM.
- [x] No-detection cases return `No defect detected.` and do not invoke the VLM.
- [x] VLM loader spy independently proves non-invocation on a withheld case.
- [x] Determinism evidence preserves repeated TTA vectors and gate decisions.
- [x] Explained, withheld, and no-detection annotated examples are present in `evidence/final/`.
- [x] Expanded 29-image results, source hashes, timings, and 29 annotated outputs are present in `evidence/expanded/`.
- [x] `TEST_RESULTS.md` includes gate counts, latency KPIs, and illustrative n=5 accuracy.
- [x] `SENSITIVITY.md` records the isolated requested threshold grid and shipped-threshold comparison.

## User Interfaces

- [x] CLI real smoke test exits 0 and emits structured no-detection evidence.
- [x] Gradio builds without model loading in an automated test.
- [x] Gradio real server launch returns HTTP 200 with the expected title.
- [x] Upload adapter calls `pipeline.inspect()` rather than duplicating gate logic.
- [x] UI displays a red-box annotation when a box exists, metrics, risk, explanation/refusal, and gate log.
- [x] UI queue is serialized at concurrency one for CPU/RAM safety.

## Documentation and Quality

- [x] `README.md` contains setup, usage, architecture, models, thresholds, validation, and limitations.
- [x] `REPORT.md` contains implementation, evidence, fallbacks, interpretation, and explicit limitations.
- [x] `DECISIONS.md` records environment, model, gate, tuning, evidence, UI, and status decisions.
- [x] `requirements.txt` contains the verified pinned environment.
- [x] Ruff, compilation, 38 tests, and `pip check` pass.
- [x] `pipeline.py` remains byte-identical to approved Pass 5, SHA-256 `9c222780f0730fb43129411c14071a397bbc8564a9dffd5dc3e027769f82c423`.

## Completion Status

**COMPLETE WITH DOCUMENTED LIMITATIONS**

The gate and required interfaces are complete and evidenced. The qualification reflects small selected evaluation sets, a documented stable false positive, demonstration-level threshold tuning, CPU/VLM latency, incomplete semantic guarantees, and model supply-chain/licensing constraints.
