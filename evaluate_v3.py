"""V3 detection operating point: python evaluate_v3.py prepare|infer|calibrate|report.
Calibration and new test samples are fixed before inference; v2 tests never select thresholds.
"""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

import evaluate_v2 as v2
import pipeline

OUT = Path("evidence/v3")
SEED = 20260921


def prepare():
    old = v2.read(v2.OUT / "split_manifest.json")
    seen = {r["sha256"] for r in old["rows"]}
    seen.update(v2.digest(p) for p in (v2.ROOT / "images/train").glob("*"))
    pools = {True: [], False: []}
    for p in sorted(v2.EXTERNAL.rglob("*.jpg")):
        if p.parent.name.upper() in {
            "CD",
            "UD",
            "CW",
            "UW",
            "CP",
            "UP",
            "CRACKED",
            "NON-CRACKED",
            "UNCRACKED",
        }:
            pools[p.parent.name.upper().startswith("C")].append(p)
    rng = random.Random(SEED)
    rows = []
    for positive, paths in pools.items():
        rng.shuffle(paths)
        chosen = []
        for p in paths:
            sha = v2.digest(p)
            if sha in seen:
                continue
            seen.add(sha)
            chosen.append(
                {
                    "path": str(p.resolve()),
                    "relative_path": p.relative_to(v2.EXTERNAL).as_posix(),
                    "sha256": sha,
                    "positive": positive,
                }
            )
            if len(chosen) == 200:
                break
        assert len(chosen) == 200
        for i, r in enumerate(chosen):
            r.update(role="calibration" if i < 100 else "fresh_test", id=len(rows))
            rows.append(r)
    v2.save(
        OUT / "manifest.json",
        {
            "seed": SEED,
            "rows": rows,
            "v2_manifest_sha256": v2.digest(v2.OUT / "split_manifest.json"),
            "limitations": [
                "Patch-level split; parent-scene independence is not established.",
                "SDNET is now used for calibration, so this is domain-adapted evaluation, not unseen-domain validation.",
                "Image labels do not validate individual boxes or segmentation.",
            ],
        },
    )


def infer():
    import torch

    torch.set_num_threads(2)
    cfg = pipeline.PipelineConfig(device="cpu")
    provenance = {
        "config": asdict(cfg),
        "weights_sha256": v2.digest(Path(cfg.detector_path)),
        "manifest_sha256": v2.digest(OUT / "manifest.json"),
        "pipeline_sha256": v2.digest(Path("pipeline.py")),
        "evaluation_sha256": v2.digest(Path(__file__)),
    }
    path = OUT / "inference_provenance.json"
    if path.exists():
        previous = v2.read(path)
        for key in ("config", "weights_sha256", "manifest_sha256", "pipeline_sha256"):
            assert previous[key] == provenance[key], "Stale inference cache"
    else:
        v2.save(path, provenance)
        source = OUT / "source" / "evaluate_v3_inference.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(Path(__file__).read_bytes())
    detector = pipeline._load_detector(cfg)
    for row in v2.read(OUT / "manifest.json")["rows"]:
        target = OUT / "detections" / f"{row['id']:04}.json"
        if target.exists():
            assert v2.read(target)["sha256"] == row["sha256"], "Mismatched cached image"
            continue
        assert v2.digest(Path(row["path"])) == row["sha256"]
        with Image.open(row["path"]) as im:
            regions = pipeline._predict_all(detector, pipeline.load_image(im), cfg)
        v2.save(
            target,
            {
                "sha256": row["sha256"],
                "detections": [
                    {"raw_confidence": r.confidence, "box": r.box} for r in regions
                ],
            },
        )
        print(f"{row['id'] + 1}/400 {row['role']}", flush=True)


def examples(role):
    return [
        (r, v2.read(OUT / "detections" / f"{r['id']:04}.json")["detections"])
        for r in v2.read(OUT / "manifest.json")["rows"]
        if r["role"] == role
    ]


def calibrate():
    rows = examples("calibration")
    y = [r["positive"] for r, d in rows]
    scores = np.array(
        [max((d["raw_confidence"] for d in ds), default=0) for r, ds in rows]
    )
    # Predeclared objective: maximize balanced-sample accuracy, retain >=85% sensitivity.
    curve = []
    for threshold in np.round(np.arange(0.01, 0.351, 0.005), 3):
        m = v2.metrics(y, scores >= threshold)
        curve.append({"threshold": float(threshold), **m})
    baseline_recall = curve[0]["recall"]
    recall_floor = 0.85 if baseline_recall >= 0.85 else max(0.0, baseline_recall - 0.05)
    eligible = [m for m in curve if m["recall"] >= recall_floor]
    chosen = max(eligible, key=lambda m: (m["accuracy"], m["recall"], -m["threshold"]))
    location_curve = []
    old = v2.read(v2.OUT / "split_manifest.json")
    localization = [
        (r, v2.read(v2.OUT / "detections" / f"{r['id']:04}.json")["detections"])
        for r in old["rows"]
        if r["role"] == "calibration"
    ]
    for t in np.round(np.arange(0.01, 0.801, 0.005), 3):
        selected = [
            ([d for d in ds if d["raw_confidence"] >= t], r["boxes"])
            for r, ds in localization
        ]
        tp = sum(sum(v2.region_truth(ds, b)) for ds, b in selected)
        n = sum(len(ds) for ds, b in selected)
        gt = sum(len(b) for ds, b in selected)
        location_curve.append(
            {
                "threshold": float(t),
                "precision": tp / n if n else 0,
                "recall": tp / gt,
                "f1": 2 * tp / (n + gt),
                "tp": tp,
                "predictions": n,
                "gt": gt,
            }
        )
    location = max(
        (m for m in location_curve if m["recall"] >= 0.75),
        key=lambda m: (m["f1"], m["recall"], -m["threshold"]),
    )
    v2.save(OUT / "localization_curve.json", location_curve)
    v2.save(OUT / "calibration_curve.json", curve)
    v2.save(
        OUT / "operating_point.json",
        {
            "raw_confidence_threshold": chosen["threshold"],
            "localization_threshold": location["threshold"],
            "localization_calibration": location,
            "localization_selection": "Maximum box F1 on Crack-Seg calibration subject to >=75% GT recall; not chosen on test images.",
            "selection": "Maximum calibration accuracy with recall >=85% if feasible; otherwise no more than 5 percentage points below baseline recall. Ties favor recall then lower threshold.",
            "recall_target_feasible": baseline_recall >= 0.85,
            "calibration_baseline_recall": baseline_recall,
            "recall_floor": recall_floor,
            "protocol_amendment": "Initial >=85% recall target was infeasible on calibration (81% at floor). Fallback declared before scoring the fresh test; no test labels used.",
            "calibration": chosen,
            "manifest_sha256": v2.digest(OUT / "manifest.json"),
            "detector_sha256": v2.digest(Path(pipeline.PipelineConfig().detector_path)),
            "confidence_guarantee": None,
        },
    )
    print(json.dumps(chosen, indent=2))


def report():
    artifact = v2.read(OUT / "operating_point.json")
    threshold = artifact["raw_confidence_threshold"]
    sets = {"fresh_test": examples("fresh_test")}
    old = v2.read(v2.OUT / "split_manifest.json")
    for role in ("evaluation", "external"):
        sets["v2_" + role] = [
            (r, v2.read(v2.OUT / "detections" / f"{r['id']:04}.json")["detections"])
            for r in old["rows"]
            if r["role"] == role
        ]
    results = {"operating_point": artifact, "datasets": {}}
    for name, rows in sets.items():
        y = [r["positive"] for r, d in rows]
        entry = {}
        for label, t in [("before", 0.01), ("after", threshold)]:
            pred = [any(d["raw_confidence"] >= t for d in ds) for r, ds in rows]
            entry[label] = v2.bootstrap_metrics(y, pred)
            entry[label]["boxes_per_image"] = sum(
                sum(
                    d["raw_confidence"]
                    >= (
                        0.01
                        if label == "before"
                        else artifact["localization_threshold"]
                    )
                    for d in ds
                )
                for r, ds in rows
            ) / len(rows)
            if name == "v2_evaluation":
                selected = [
                    (
                        [
                            d
                            for d in ds
                            if d["raw_confidence"]
                            >= (
                                0.01
                                if label == "before"
                                else artifact["localization_threshold"]
                            )
                        ],
                        r["boxes"],
                    )
                    for r, ds in rows
                ]
                tp = sum(sum(v2.region_truth(ds, boxes)) for ds, boxes in selected)
                n = sum(len(ds) for ds, b in selected)
                total = sum(len(b) for ds, b in selected)
                entry[label]["localization"] = {
                    "tp": tp,
                    "predictions": n,
                    "gt": total,
                    "precision": tp / n if n else 0,
                    "recall": tp / total,
                    "f1": 2 * tp / (n + total),
                }
        rng = np.random.default_rng(SEED)
        before = np.array([bool(ds) for r, ds in rows]) == y
        after = (
            np.array(
                [any(d["raw_confidence"] >= threshold for d in ds) for r, ds in rows]
            )
            == y
        )
        delta = after.astype(float) - before.astype(float)
        draws = [
            delta[rng.integers(0, len(rows), len(rows))].mean() for _ in range(5000)
        ]
        entry["accuracy_delta"] = {
            "estimate": float(delta.mean()),
            "paired_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
        results["datasets"][name] = entry
    v2.save(OUT / "results.json", results)
    v2.save(
        OUT / "report_provenance.json",
        {
            "evaluation_sha256": v2.digest(Path(__file__)),
            "view_sha256": v2.digest(Path("inspection_view.py")),
            "results_sha256": v2.digest(OUT / "results.json"),
        },
    )
    deployment = dict(artifact)
    deployment["raw_confidence_threshold"] = 0.01
    deployment["presence_trial_threshold"] = artifact["raw_confidence_threshold"]
    deployment["release_decision"] = (
        "Keep baseline presence threshold: trial regressed on old external benchmark and reduced recall on both external tests. Deploy localization filtering and separated language status only."
    )
    v2.save(Path("models/detection_operating_point.json"), deployment)
    write_docs(results)
    print(json.dumps(results, indent=2))


def write_docs(results):
    artifact = results["operating_point"]

    def pct(v):
        return f"{100 * v:.2f}%"

    body = "## V3 operating-point evaluation - September 21, 2026\n\n"
    body += "**Release decision:** ship the calibrated box overlay and separate detection/language statuses. Keep image-level presence at 0.01. The 0.015 presence trial below was rejected: fresh accuracy improved slightly but the older external benchmark and sensitivity regressed. Deployed image-level metrics therefore remain the Before values. No detector weights or language gate were weakened.\n\n"
    body += "| Metric | Before (V2 policy on same images) | Trial (presence) / deployed (overlay) | Delta |\n|---|---:|---:|---:|\n"
    for name, entry in results["datasets"].items():
        for metric in ("precision", "recall", "f1", "accuracy"):
            old, new = entry["before"][metric], entry["after"][metric]
            body += f"| {name}, n=200: {metric} | {pct(old)} | {pct(new)} | {(new - old) * 100:+.2f} pp |\n"
    entry = results["datasets"]["v2_evaluation"]
    for metric in ("precision", "recall", "f1"):
        old, new = (
            entry["before"]["localization"][metric],
            entry["after"]["localization"][metric],
        )
        body += f"| Box localization {metric}, IoU >=0.5 | {pct(old)} | {pct(new)} | {(new - old) * 100:+.2f} pp |\n"
    old, new = entry["before"]["boxes_per_image"], entry["after"]["boxes_per_image"]
    body += f"| Drawn boxes / Crack-Seg image | {old:.2f} | {new:.2f} | {new - old:+.2f} |\n\n"
    body += "These are measured detector operating-point changes, not retrained weights or a claim that all hallucinations are identifiable. Every candidate at the original 0.01 floor still receives its independent five-pass TTA gate and remains in the audit. Only the overlay uses the new deployed threshold; image-level presence retains the old sensitive threshold. The existing language gate and grounding checks remain unchanged.\n\n"
    body += f"**Frozen trial thresholds:** crack presence `{artifact['raw_confidence_threshold']:.3f}`; displayed localization `{artifact['localization_threshold']:.3f}`. Presence maximizes accuracy on a NEW balanced 200-image SDNET calibration split subject to >=85% recall when feasible. The initial target was infeasible (81% baseline calibration recall); a documented calibration-only amendment limits sensitivity loss to 5 percentage points instead. Localization maximizes box F1 on the original disjoint 111-image Crack-Seg calibration split subject to >=75% ground-truth-box recall. Ties favor recall, then a lower threshold. Test scores never choose thresholds. See the complete curves and hashed manifest in `evidence/v3/`.\n\n"
    body += "**Independent checks:** a NEW 200-image SDNET test split (100 cracked / 100 non-cracked), the unchanged previous 200-image SDNET benchmark, and the unchanged 200-image Crack-Seg benchmark. All 400 new images had fresh detector inference. Prior 400 test-image detector outputs were replayed exactly; no new VLM accuracy result is claimed. Seed 20260921; selected bytes are disjoint from prior evaluation/calibration and known Crack-Seg training images. SDNET is now calibration data, so call these domain-adapted results, not validation on an untouched external domain. Parent-scene independence and unseen pretraining exposure remain unverified. Crack-Seg still has only one negative and its accuracy is not a useful general false-positive estimate.\n\n"
    body += "### Confidence intervals and tradeoffs\n\n| Dataset | After metric | Estimate | Conservative 95% CI |\n|---|---|---:|---|\n"
    for name, entry in results["datasets"].items():
        for metric, ci in entry["after"]["simultaneous_rate_ci95"].items():
            body += f"| {name} | {metric} | {pct(entry['after'][metric])} | [{pct(ci['low'])}, {pct(ci['high'])}] |\n"
    for name, entry in results["datasets"].items():
        delta = entry["accuracy_delta"]
        lo, hi = delta["paired_bootstrap_ci95"]
        body += f"\n{name}: paired accuracy change {delta['estimate'] * 100:+.2f} pp (95% paired bootstrap [{lo * 100:+.2f}, {hi * 100:+.2f}] pp); FP {entry['before']['fp']} -> {entry['after']['fp']}, FN {entry['before']['fn']} -> {entry['after']['fn']}.\n"
    body += "\nNarrower overlays sacrifice some box recall; omitted candidates remain in Detailed evidence. An image can have a possible crack signal without a reliable location to draw. Solid red boxes mean the detector passes the existing stability gate; dashed white boxes mean candidate detections with uncertain language support. A blocked language claim is separate from detector evidence. Scores are not calibrated probabilities, and automated grounding cannot verify every visual assertion. No structural safety assessment is provided.\n\n"
    body += "### Reproduction\n\n```powershell\npython evaluate_v3.py prepare\npython evaluate_v3.py infer\npython evaluate_v3.py calibrate\npython evaluate_v3.py report\n```\n\nUse the existing Crack-Seg and SDNET dataset paths documented for V2. `infer` verifies image hashes and caches one original detector pass per image, sufficient for this raw-score operating-point study. It does not substitute a one-pass language gate: deployed inference still performs five passes for every base candidate. Historical inference source and provenance are preserved; the current report source hash is recorded separately. The fresh test and old benchmark estimates are descriptive patch-level results, not guarantees on user road photographs.\n\n"
    block = "<!-- V3 RESULTS START -->\n" + body + "<!-- V3 RESULTS END -->\n\n"
    for filename in ("REPORT.md", "TEST_RESULTS.md", "SENSITIVITY.md"):
        path = Path(filename)
        text = path.read_text(encoding="utf-8")
        if "<!-- V3 RESULTS START -->" in text:
            start = text.index("<!-- V3 RESULTS START -->")
            end = text.index("<!-- V3 RESULTS END -->") + len("<!-- V3 RESULTS END -->")
            text = text[:start] + text[end:].lstrip("\n")
        title, rest = text.split("\n", 1)
        path.write_text(title + "\n\n" + block + rest.lstrip("\n"), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "infer", "calibrate", "report"])
    globals()[parser.parse_args().step]()
