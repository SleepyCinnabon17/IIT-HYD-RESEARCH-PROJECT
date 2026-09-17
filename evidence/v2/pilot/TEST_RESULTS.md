# Expanded Test Results

This is additive evidence collected between Pass 5 and Pass 6. It does not replace the locked Pass 5 evidence. Every row was executed through the unchanged `inspect()` entrypoint. Stage timings were collected by external wrappers around the detector predictor and VLM generator; no production decision logic was modified.

## Evaluation Configuration

- Shipped thresholds observed at execution: confidence `0.35`, TTA standard-deviation `0.03`.
- Threshold chronology: standard deviation `0.15` was the initial Pass 4/early Pass 5 value; Pass 5 tuning changed and locked it at `0.03` before the authoritative acceptance and this expanded run. Pass 6 did not change it.
- Detector passes: exactly five per image.
- Additional source: the already-integrated Ultralytics Crack-Seg test/validation splits only.
- Combined rows: 29 (5 original + 24 additional).
- The Crack-Seg archive contains one empty-label validation frame; it is included as the additional background case.

## Results

| Image | Set | Expected category | Raw conf. | Calibrated conf. | TTA stddev | Risk | VLM called | VLM succeeded | Base detector (s) | 5-pass TTA (s) | VLM (s) | Total inspect (s) | Explanation |
|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|
| `clear_01.jpg` | original | clear crack | 0.8954 | 0.8986 | 0.0042 | Low | True | True | 2.420 | 3.475 | 49.149 | 52.656 | The crack is vertical and extends from the top left to the bottom right of the image. |
| `clear_02.jpg` | original | clear crack | 0.7593 | 0.7419 | 0.0187 | Low | True | True | 0.829 | 2.677 | 31.697 | 35.085 | A large crack in the pavement, with the crack extending from the left side of the image towards the right. |
| `clear_03.jpg` | original | clear crack | 0.8173 | 0.8095 | 0.0110 | Low | True | False | 0.496 | 2.284 | 22.600 | 24.920 | Explanation unavailable — VLM failed closed; human review required |
| `ambiguous_01.jpg` | original | ambiguous shadow | 0.4774 | 0.4658 | 0.0518 | High | False | False | 0.374 | 2.059 | — | 2.102 | Explanation withheld — flagged for human review |
| `ambiguous_02.jpg` | original | synthetic background | 0.0000 | 0.0000 | 0.0000 | N/A | False | False | 0.389 | 1.889 | — | 1.942 | No defect detected. |
| `test_1616.rf.c868709931a671796794fdbb95352c5a.jpg` | expanded | clear crack (dataset label) | 0.6512 | 0.6588 | 0.0168 | Low | True | True | 0.434 | 2.086 | 21.937 | 24.060 | The crack in the wall is quite large and extends horizontally across the image. |
| `test_1675.rf.e3aa3f8d28d0247ef0284dd46dacc29f.jpg` | expanded | clear crack (dataset label) | 0.7671 | 0.7081 | 0.0759 | High | False | False | 0.345 | 1.808 | — | 1.839 | Explanation withheld — flagged for human review |
| `test_1686.rf.809fb1b51c607e5cf787e44ef4ddd7b8.jpg` | expanded | clear crack (dataset label) | 0.7865 | 0.7848 | 0.0314 | Medium | False | False | 0.389 | 1.988 | — | 2.020 | Explanation withheld — flagged for human review |
| `test_1706.rf.011d213c21ec78896c36728dcbc156f5.jpg` | expanded | clear crack (dataset label) | 0.0853 | 0.2310 | 0.1688 | High | False | False | 0.365 | 1.777 | — | 1.817 | Explanation withheld — flagged for human review |
| `test_1716.rf.85ea38b36008beaa72c5d8541f734eb0.jpg` | expanded | clear crack (dataset label) | 0.8292 | 0.8080 | 0.0284 | Low | True | True | 0.395 | 1.888 | 21.294 | 23.222 | The crack is located in the center of the image and extends horizontally across the frame. |
| `test_1722.rf.38b38f2e833309a4f35bfbf0432dffff.jpg` | expanded | clear crack (dataset label) | 0.5909 | 0.5471 | 0.0785 | High | False | False | 0.350 | 1.690 | — | 1.722 | Explanation withheld — flagged for human review |
| `test_1794.rf.7a03ca09d05e9e2941f768bc8570cb54.jpg` | expanded | clear crack (dataset label) | 0.3492 | 0.4327 | 0.1255 | High | False | False | 0.428 | 1.951 | — | 1.982 | Explanation withheld — flagged for human review |
| `test_1804.rf.7e39a73f7b0d1c4bdf8094006e0cf495.jpg` | expanded | clear crack (dataset label) | 0.7006 | 0.6912 | 0.0455 | High | False | False | 0.385 | 1.937 | — | 1.972 | Explanation withheld — flagged for human review |
| `test_1813.rf.79f1af872f76dd8b46df4a83ed5f54c6.jpg` | expanded | clear crack (dataset label) | 0.8284 | 0.8157 | 0.0216 | Low | True | True | 0.378 | 1.834 | 21.952 | 23.824 | The crack is located in the middle of the image, running horizontally across the center of the wall. |
| `test_1818.rf.7e52c1687bfb625e86c54c9d71a66d99.jpg` | expanded | clear crack (dataset label) | 0.7799 | 0.7855 | 0.0217 | Low | True | True | 0.725 | 2.275 | 21.636 | 23.945 | The crack in the rock is quite large and runs horizontally across the image. |
| `test_1819.rf.d2d41865c85e1019dc3e8b9daf73c434.jpg` | expanded | clear crack (dataset label) | 0.5693 | 0.6046 | 0.0555 | High | False | False | 0.338 | 1.878 | — | 1.910 | Explanation withheld — flagged for human review |
| `test_1839.rf.3495cbba8ee9a583ed0dff8d0e5fc0f7.jpg` | expanded | clear crack (dataset label) | 0.1348 | 0.1714 | 0.1036 | High | False | False | 0.369 | 1.680 | — | 1.712 | Explanation withheld — flagged for human review |
| `val_1604.rf.7229a9adfa1c9ec285d55c965172ea32.jpg` | expanded | clear crack (dataset label) | 0.8977 | 0.8913 | 0.0205 | Low | True | True | 0.341 | 1.689 | 21.941 | 23.668 | The crack is located on the right side of the image and extends horizontally across the frame. |
| `val_1605.rf.53a5ea427ceda0b3d6abbc79c64efc36.jpg` | expanded | clear crack (dataset label) | 0.7932 | 0.7780 | 0.0478 | High | False | False | 0.405 | 1.725 | — | 1.755 | Explanation withheld — flagged for human review |
| `val_1610.rf.3271e8c1058a3a701de2d96b621e9080.jpg` | expanded | clear crack (dataset label) | 0.6602 | 0.6816 | 0.0369 | Medium | False | False | 0.326 | 1.702 | — | 1.731 | Explanation withheld — flagged for human review |
| `val_1619.rf.5a42efe0c04aca69843e73fcc3a1a1bc.jpg` | expanded | clear crack (dataset label) | 0.7517 | 0.7622 | 0.0189 | Low | True | True | 0.323 | 1.567 | 23.161 | 24.762 | The crack is long and narrow, extending from the top left to the bottom right of the image. |
| `val_1622.rf.da42af8068237feeeaa4db6f8628ccb2.jpg` | expanded | clear crack (dataset label) | 0.5691 | 0.6020 | 0.0523 | High | False | False | 0.606 | 2.385 | — | 2.419 | Explanation withheld — flagged for human review |
| `val_1629.rf.3ecf969381ffe53960ebb20a47013488.jpg` | expanded | clear crack (dataset label) | 0.5820 | 0.6197 | 0.0926 | High | False | False | 0.930 | 3.452 | — | 3.484 | Explanation withheld — flagged for human review |
| `val_1631.rf.e30824b70361db2a9e2eb7863269ce86.jpg` | expanded | clear crack (dataset label) | 0.8116 | 0.7906 | 0.0251 | Low | True | True | 0.332 | 1.873 | 25.653 | 27.559 | The crack extends horizontally across the image, with visible signs of wear and tear, and it appears to be a significant one. |
| `val_1637.rf.e941aa1b84b3d14c3ea9f48867b6676a.jpg` | expanded | clear crack (dataset label) | 0.6734 | 0.6809 | 0.0105 | Low | True | True | 0.459 | 1.954 | 28.997 | 30.985 | The crack in the image is quite large and extends horizontally across the image. |
| `val_1647.rf.3bab20d4aa37f4fe965fbf2cfe1622fb.jpg` | expanded | clear crack (dataset label) | 0.7120 | 0.6847 | 0.0863 | High | False | False | 0.487 | 1.793 | — | 1.825 | Explanation withheld — flagged for human review |
| `val_1649.rf.11669e5fb853ededb95501a2bc7e31b9.jpg` | expanded | clear crack (dataset label) | 0.6117 | 0.6556 | 0.0564 | High | False | False | 0.314 | 1.663 | — | 1.693 | Explanation withheld — flagged for human review |
| `val_1658.rf.38d44ec0dfa198d017ba9cb0ada57f15.jpg` | expanded | clear crack (dataset label) | 0.8615 | 0.8704 | 0.0073 | Low | True | True | 0.329 | 1.693 | 25.226 | 26.953 | The crack is located in the center of the image and extends diagonally from the top left to the bottom right. |
| `val_3513.rf.782300f38d0a008b3340e54b643e713e.jpg` | expanded | background (empty dataset label) | 0.8144 | 0.8123 | 0.0260 | Low | True | True | 0.503 | 1.874 | 22.384 | 24.296 | The crack is located on the right side of the image and extends horizontally across the frame. |

## Summary

- Low risk / VLM called: **13**
- Medium or High risk / explanation withheld: **15**
- No detection / VLM skipped: **1**
- Original five pattern `(Low, withheld, no-detection)`: **(3, 1, 1)**
- Additional set pattern `(Low, withheld, no-detection)`: **(10, 14, 0)**
- Pattern consistency: **Directionally yes** — the expanded set repeats both Low/called and Medium-or-High/withheld behavior. It did not repeat a no-detection case.
- Background observation: **The sole empty-label background frame was detected and classified Low risk, so it invoked the VLM; this is a concrete false-positive warning for human review.**

## Latency KPI Summary

All values are wall-clock seconds from this CPU-only run. VLM timing includes lazy model loading on the first call.

| Stage | n | Mean | Median | P95 |
|---|---:|---:|---:|---:|
| Base detector inference | 29 | 0.509 | 0.389 | 0.930 |
| Full five-pass TTA | 29 | 2.019 | 1.878 | 3.452 |
| VLM cold start (first call, includes model load) | 1 | 49.149 | — | — |
| VLM warm inference (calls 2 onward) | 12 | 24.040 | 22.492 | 31.697 |
| Total inspect (all images) | 29 | 13.719 | 2.419 | 35.085 |
| Total inspect (VLM called) | 13 | 28.149 | 24.762 | 52.656 |
| Total inspect (VLM skipped) | 16 | 1.995 | 1.874 | 3.484 |

## Small-Sample Gate Accuracy

**n=5, illustrative only, not a statistically powered evaluation.**

The positive class is `crack present = yes`; the pipeline prediction is its unchanged `detected` output.

| Metric | Result |
|---|---:|
| True positives | 3 |
| False positives | 1 |
| True negatives | 1 |
| False negatives | 0 |
| Precision | 75.00% |
| Recall | 100.00% |
| Accuracy | 80.00% |
