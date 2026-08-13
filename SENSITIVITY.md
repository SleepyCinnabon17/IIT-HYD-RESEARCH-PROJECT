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
