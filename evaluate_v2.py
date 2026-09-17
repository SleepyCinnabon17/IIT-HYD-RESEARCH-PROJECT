"""Reproducible v2 benchmark. See --help; no synthetic images substitute for data.

Run prepare, detect, summarize, then external-vlm. Detection caches contain real
inference and are fingerprinted against code, weights, configuration and input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

import pipeline

SEED = 20260907
OUT = Path("evidence/v2")
ROOT = Path.home() / ".cache/hallucination-aware-visual-inspection/crack-seg"
EXTERNAL = Path("data/sdnet2018")
SDNET_ARCHIVE_SHA256 = '7ac0fcf73fdc33555d852e8ad761ba42fdcdef97493016331c816a385575b5b8'


def download():
    """Acquire the public SDNET mirror; retain original CC BY attribution in manifest."""
    import zipfile

    import requests
    archive = EXTERNAL.parent / 'sdnet2018.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = 'https://www.kaggle.com/api/v1/datasets/download/prathameshgadekar/sdnet-2018'
        response = requests.get(url, stream=True, timeout=90)
        response.raise_for_status()
        temporary = archive.with_suffix('.partial')
        with temporary.open('wb') as stream:
            for block in response.iter_content(1024*1024):
                stream.write(block)
        temporary.replace(archive)
    destination = EXTERNAL.resolve()
    if digest(archive) != SDNET_ARCHIVE_SHA256:
        raise ValueError('SDNET mirror archive changed; review provenance before creating a new split.')
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive path escapes dataset directory')
        package.extractall(destination)
    print(json.dumps({'archive': str(archive), 'sha256': digest(archive)}))


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_boxes(path, size):
    boxes = []
    for line in path.read_text().splitlines():
        values = list(map(float, line.split()))[1:]
        if len(values) == 4:
            x, y, w, h = values
            boxes.append([(x-w/2)*size[0], (y-h/2)*size[1], (x+w/2)*size[0], (y+h/2)*size[1]])
        else:
            points = np.asarray(values).reshape(-1, 2)
            boxes.append([float(points[:, 0].min()*size[0]), float(points[:, 1].min()*size[1]),
                          float(points[:, 0].max()*size[0]), float(points[:, 1].max()*size[1])])
    return boxes


def overlap_audit():
    """Perceptual nearest-neighbor screen, without post-inference sample curation."""
    import cv2
    cv2.setNumThreads(1)
    def phash(image):
        pixels = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float32)
        coefficients = cv2.dct(pixels)[:8, :8].ravel()[1:]
        return coefficients > np.median(coefficients)
    training_paths = sorted((ROOT / 'images/train').glob('*'))
    train = []
    for path in training_paths:
        with Image.open(path) as image:
            train.append(phash(image))
    train = np.asarray(train)
    records = []
    for row in read(OUT / 'split_manifest.json')['rows']:
        if row['role'] != 'external':
            continue
        with Image.open(row['path']) as image:
            original = phash(image)
            mirrored = phash(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        distances = np.minimum(np.sum(train != original, axis=1), np.sum(train != mirrored, axis=1))
        nearest = int(np.argmin(distances))
        records.append({'id': row['id'], 'sha256': row['sha256'], 'nearest_training_path': str(training_paths[nearest]),
                        'hamming_distance_63_bits': int(distances[nearest])})
    save(OUT / 'external_overlap_audit.json', {'n': len(records), 'training_n': len(training_paths),
         'suspect_distance_le_4': sum(r['hamming_distance_63_bits'] <= 4 for r in records),
         'minimum_distance': min(r['hamming_distance_63_bits'] for r in records), 'records': records,
         'scope': '63-bit grayscale DCT perceptual hash, original and mirror. Screening only; no guarantee against crops, transformations or unseen pretraining. No sampled image removed based on this screen.'})
    print('Perceptual overlap screen complete', flush=True)


def prepare():
    rng = random.Random(SEED)
    audit = {}
    train_hashes = {digest(p) for p in sorted((ROOT / "images/train").glob("*"))}
    seen = set(train_hashes)
    excluded = []
    pools = {True: [], False: []}
    for split in ("train", "val", "test"):
        images = sorted((ROOT / "images" / split).glob("*"))
        labels = [ROOT / "labels" / split / (p.stem + ".txt") for p in images]
        assert all(p.exists() for p in labels), "Missing labels cannot be treated as negatives"
        audit[split] = {"images": len(images), "crack_free": sum(not p.read_text().strip() for p in labels)}
        if split == "train":
            continue
        for p, label in zip(images, labels):
            sha = digest(p)
            if sha in seen:
                excluded.append({'path': str(p), 'reason': 'training or prior pool exact-byte duplicate'})
                continue
            seen.add(sha)
            with Image.open(p) as im:
                boxes = label_boxes(label, im.size)
            pools[bool(boxes)].append({"path": str(p), "relative_path": p.relative_to(ROOT).as_posix(),
                                      "sha256": sha, "label_sha256": digest(label), "positive": bool(boxes),
                                      "boxes": boxes, "source_split": split, "dataset": "Crack-Seg"})
    # Stratified random selection from non-training partitions. The sole negative
    # was used by the pilot; this cannot be called a fresh held-out test set.
    positives = rng.sample(pools[True], len(pools[True]))
    negatives = rng.sample(pools[False], len(pools[False]))
    assert len(positives) >= 199 and negatives
    test = positives[:199] + negatives[:1]
    calibration = positives[199:] + negatives[1:]
    for row in test:
        row["role"] = "evaluation"
    for row in calibration:
        row["role"] = "calibration"
    external_pools = {True: [], False: []}
    for p in sorted(EXTERNAL.rglob("*.jpg")):
        parent = p.parent.name.upper()
        if parent not in {"CD", "UD", "CW", "UW", "CP", "UP", "CRACKED", "NON-CRACKED", "UNCRACKED"}:
            continue
        external_pools[parent.startswith("C")].append(p)
    external = []
    for positive in (True, False):
        added = 0
        for p in rng.sample(external_pools[positive], len(external_pools[positive])):
            sha = digest(p)
            if sha in seen:
                excluded.append({'path': str(p), 'reason': 'exact-byte overlap'})
                continue
            seen.add(sha)
            external.append({"path": str(p.resolve()), "relative_path": p.relative_to(EXTERNAL).as_posix(),
                             "sha256": sha, "positive": positive, "boxes": None,
                             "dataset": "SDNET2018", "role": "external", "source_split": p.parent.parent.name + '/' + p.parent.name})
            added += 1
            if added == 100:
                break
        assert added == 100
    rows = calibration + test + external
    assert len({r['sha256'] for r in rows}) == len(rows), "Duplicate selected images"
    audit_ids = set(rng.sample([r["sha256"] for r in test], 30))
    for i, row in enumerate(rows):
        row["id"] = i
        row["tta_audit"] = row["sha256"] in audit_ids
    manifest = {"seed": SEED, "audit": audit, "rows": rows, "excluded_duplicates": excluded,
                "stratification": "Crack-Seg 199 positive + sole negative; SDNET2018 100 positive + 100 negative",
                "limitations": ["Crack-Seg validation was available for checkpoint selection.",
                                "The sole negative was already evaluated in the n=29 pilot.",
                                "Exact-byte training overlap excluded; no guarantee against crops or near-duplicates.",
                                "SDNET2018 parent photographs may correlate; image-level CIs are conditional on sampled patches."],
                "external_source": "https://digitalcommons.usu.edu/all_datasets/48/",
                "external_download": "https://www.kaggle.com/api/v1/datasets/download/prathameshgadekar/sdnet-2018",
                "external_license": "CC BY 4.0 per original authors (mirror's CC0 label is not used)",
                "external_attribution": "Maguire, Dorafshan & Thomas (2018), SDNET2018, Utah State University, doi:10.15142/T3TD19",
                "external_archive_sha256": digest(EXTERNAL.parent / "sdnet2018.zip")}
    save(OUT / "split_manifest.json", manifest)
    print(json.dumps({"audit": audit, "roles": {s: sum(r['role'] == s for r in rows) for s in ('calibration', 'evaluation', 'external')}}))


def config(n=5):
    return pipeline.PipelineConfig(variance_threshold=.03, tta_passes=n, device="cpu")


def detect():
    import ast

    import torch
    torch.set_num_threads(4)
    manifest = read(OUT / "split_manifest.json")
    detector = pipeline._load_detector(config())
    fingerprint = {"pipeline_sha256": digest("pipeline.py"), "weights_sha256": digest(config().detector_path),
                   "manifest_sha256": digest(OUT / "split_manifest.json"), "config": asdict(config()),
                   "python": platform.python_version(), "torch": torch.__version__}
    provenance_path = OUT / "inference_provenance.json"
    source_path = OUT / 'source/pipeline.py'
    if provenance_path.exists():
        previous = read(provenance_path)
        comparison = fingerprint.copy()
        if source_path.exists() and digest(source_path) == previous['pipeline_sha256']:
            same_ast = ast.dump(ast.parse(source_path.read_text(encoding='utf-8'))) == ast.dump(ast.parse(Path('pipeline.py').read_text(encoding='utf-8')))
            if same_ast:
                comparison['pipeline_sha256'] = previous['pipeline_sha256']
        if previous != comparison:
            raise RuntimeError("Inference provenance changed. Use a new output directory; do not mix caches.")
    else:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(Path('pipeline.py').read_bytes())
        save(provenance_path, fingerprint)
    for i, row in enumerate(manifest['rows']):
        target = OUT / "detections" / f"{row['id']:04}.json"
        if target.exists():
            continue
        assert digest(row['path']) == row['sha256']
        im = pipeline.load_image(row['path'])
        start = time.perf_counter()
        batch = pipeline.run_all_detections(im, config(15 if row['tta_audit'] else 5), detector)
        save(target, {"id": row['id'], "sha256": row['sha256'], "seconds": time.perf_counter()-start, **batch})
        print(f"[{i+1}/{len(manifest['rows'])}] {row['dataset']} regions={len(batch['detections'])} seconds={time.perf_counter()-start:.2f}", flush=True)


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=bool), np.asarray(pred, dtype=bool)
    tp, fp, fn, tn = int(sum(y & pred)), int(sum(~y & pred)), int(sum(y & ~pred)), int(sum(~y & ~pred))
    return {"precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None, "accuracy": (tp+tn)/len(y),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y)}


def bootstrap_metrics(y, pred):
    """Stratified image bootstrap, fixed class prevalence; undefined draws excluded."""
    y, pred = np.array(y), np.array(pred)
    result = metrics(y, pred)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == k) for k in (False, True)]
    draws = []
    for _ in range(5000):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if len(g)])
        draws.append(metrics(y[indices], pred[indices]))
    result['ci95'] = {}
    for name in ('precision', 'recall', 'f1', 'accuracy'):
        values = [d[name] for d in draws if d[name] is not None]
        lo, hi = np.quantile(values, [.025, .975]) if values else (None, None)
        result['ci95'][name] = {"low": lo, "high": hi, "width": hi-lo if values else None}
    # Wilson interval prevents misleading zero-width intervals at perfect rates.
    from scipy.stats import binomtest
    result['exact_binomial_ci95'] = {}
    for name, k, n in [('precision', result['tp'], result['tp']+result['fp']),
                       ('recall', result['tp'], result['tp']+result['fn']),
                       ('specificity', result['tn'], result['tn']+result['fp']),
                       ('accuracy', result['tp']+result['tn'], len(y))]:
        ci = binomtest(k, n).proportion_ci() if n else None
        result['exact_binomial_ci95'][name] = [ci.low, ci.high] if ci else None
    # Simultaneous 97.5% class-rate intervals (Bonferroni) propagate to metrics
    # at the fixed sampled prevalence, including a nondegenerate F1 interval.
    positives, negatives = result['tp']+result['fn'], result['tn']+result['fp']
    result['simultaneous_rate_ci95'] = {}
    if positives and negatives:
        recall_ci = binomtest(result['tp'], positives).proportion_ci(confidence_level=.975)
        specificity_ci = binomtest(result['tn'], negatives).proportion_ci(confidence_level=.975)
        prevalence = positives/len(y)
        def derived(recall, specificity):
            tp = prevalence*recall
            fp = (1-prevalence)*(1-specificity)
            fn = prevalence*(1-recall)
            return {'precision': tp/(tp+fp) if tp+fp else 0., 'recall': recall,
                    'f1': 2*tp/(2*tp+fp+fn), 'accuracy': tp+(1-prevalence)*specificity}
        lower, upper = derived(recall_ci.low, specificity_ci.low), derived(recall_ci.high, specificity_ci.high)
        result['simultaneous_rate_ci95'] = {name: {'low': lower[name], 'high': upper[name], 'width': upper[name]-lower[name]} for name in lower}
    return result


def region_truth(regions, boxes):
    edges = sorted([(pipeline.box_iou(r['box'], b), i, j) for i, r in enumerate(regions) for j, b in enumerate(boxes)], reverse=True)
    labels, used = [False]*len(regions), set()
    for iou, i, j in edges:
        if iou >= .5 and not labels[i] and j not in used:
            labels[i] = True
            used.add(j)
    return labels


def gate_values(region, n=5):
    scores = region['tta_confidences'][:n]
    return float(np.mean(scores)), float(np.std(scores))


def operating(labels, means, stds, threshold):
    from scipy.stats import beta
    labels, means, stds = np.asarray(labels), np.asarray(means), np.asarray(stds)
    low = (means >= .35) & (stds <= threshold)
    bad, count = int(sum(low & ~labels)), int(sum(low))
    return {'threshold': float(threshold), 'low_count': count, 'false_low_count': bad,
            'false_explanation_rate_proxy': bad/count if count else None,
            'false_low_rate_among_untrustworthy': bad/int(sum(~labels)) if sum(~labels) else None,
            'true_low_recall': int(sum(low & labels))/int(sum(labels)) if sum(labels) else None,
            'one_sided_95_upper': float(beta.ppf(.95, bad+1, count-bad)) if count > bad else 1.}


def grounding_benchmark():
    # Explicit, fixed challenge set with both supported and contradicted claims.
    # Labels are authored independently of running the verifier. Not real VLM accuracy.
    cases = []
    fixtures = [([5, 5, 25, 15], .8, 'left', 'right', 'horizontal', 'vertical'),
                ([75, 60, 85, 95], .4, 'right', 'left', 'vertical', 'horizontal'),
                ([30, 40, 70, 60], .9, 'center', 'bottom', 'horizontal', 'vertical')]
    for box, confidence, position, wrong_position, direction, wrong_direction in fixtures:
        width, height = box[2]-box[0], box[3]-box[1]
        for text, hallucinated in [
            (f'The detected region is at the {position}.', False),
            (f'The bounding region is {direction}.', False),
            (f'The region has width {width}%.', False),
            (f'The region has height {height}%.', False),
            (f'The detector confidence is {confidence*100:.0f}%.', False),
            (f'The detected region is at the {wrong_position}.', True),
            (f'The bounding region is {wrong_direction}.', True),
            ('The region has area 95%.', True),
            ('There is no crack in the crop.', True),
            ('The structure is unsafe due to corrosion.', True),
        ]:
            verified = pipeline.verify_grounding(text, box, (100, 100), confidence)
            cases.append({'text': text, 'box': box, 'image_size': [100, 100], 'confidence': confidence,
                          'hallucinated': hallucinated, 'old_caught': bool(pipeline.UNSUPPORTED_EXPLANATION_PATTERN.search(text)),
                          'new_caught': not verified['passed'], 'verification': verified})
    result = {'n': len(cases), 'old': bootstrap_metrics([c['hallucinated'] for c in cases], [c['old_caught'] for c in cases]),
              'new': bootstrap_metrics([c['hallucinated'] for c in cases], [c['new_caught'] for c in cases]), 'cases': cases,
              'limitation': 'Authored geometry challenge set, not held-out natural VLM hallucination prevalence. Three box templates are correlated.'}
    save(OUT / 'grounding_benchmark.json', result)
    return result


def summarize():
    from importlib.metadata import version

    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    manifest = read(OUT / 'split_manifest.json')
    data = [(r, read(OUT / 'detections' / f"{r['id']:04}.json")) for r in manifest['rows']]
    summary = {'seed': SEED, 'manifest_sha256': digest(OUT / 'split_manifest.json'), 'metrics': {}}
    summary['environment'] = {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'cpu', 'torch_threads': 4,
                              'packages': {name: version(name) for name in ('torch', 'ultralytics', 'numpy', 'Pillow', 'transformers', 'scikit-learn', 'scipy', 'matplotlib')}}
    for role in ('evaluation', 'external'):
        subset = [(r, d) for r, d in data if r['role'] == role]
        y = [r['positive'] for r, d in subset]
        summary['metrics'][role] = bootstrap_metrics(y, [bool(d['detections']) for r, d in subset])
        summary['metrics'][role]['at_confidence_035'] = bootstrap_metrics(y, [any(reg['raw_confidence'] >= .35 for reg in d['detections']) for r, d in subset])
    summary['domain_gap'] = {m: summary['metrics']['external'][m]-summary['metrics']['evaluation'][m] for m in ('precision', 'recall', 'f1', 'accuracy')}
    labels, means, stds = [], [], []
    for row, d in data:
        if row['role'] != 'calibration':
            continue
        labels.extend(region_truth(d['detections'], row['boxes']))
        for reg in d['detections']:
            mean, std = gate_values(reg)
            means.append(mean)
            stds.append(std)
    thresholds = sorted(set([0., .03] + stds))
    curve = [operating(labels, means, stds, t) for t in thresholds]
    feasible = [p for p in curve if p['low_count'] and p['false_explanation_rate_proxy'] <= .05]
    chosen = max(feasible, key=lambda p: (p['true_low_recall'], -p['false_explanation_rate_proxy'], -p['threshold'])) if feasible else operating(labels, means, stds, -1.)
    auc = {'roc': float(roc_auc_score(labels, -np.array(stds))), 'pr_average_precision': float(average_precision_score(labels, -np.array(stds)))} if len(set(labels)) == 2 else {'roc': None, 'pr_average_precision': None}
    curves = {}
    if len(set(labels)) == 2:
        fpr, tpr, th = roc_curve(labels, -np.array(stds))
        precision, recall, prth = precision_recall_curve(labels, -np.array(stds))
        curves = {'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'score_thresholds': [float(t) if np.isfinite(t) else None for t in th]},
                  'pr': {'precision': precision.tolist(), 'recall': recall.tolist(), 'score_thresholds': prth.tolist()}}
    test_labels, test_means, test_stds = [], [], []
    for row, d in data:
        if row['role'] != 'evaluation':
            continue
        test_labels.extend(region_truth(d['detections'], row['boxes']))
        for reg in d['detections']:
            mean, std = gate_values(reg)
            test_means.append(mean)
            test_stds.append(std)
    summary['calibration'] = {'trust_label': 'one-to-one ground-truth box IoU >= 0.5; not language correctness',
                              'n_regions': len(labels), 'trustworthy_regions': sum(labels), 'auc': auc,
                              'chosen': chosen, 'old': operating(labels, means, stds, .03),
                              'evaluation_old': operating(test_labels, test_means, test_stds, .03),
                              'evaluation_chosen': operating(test_labels, test_means, test_stds, chosen['threshold']),
                              'selection': 'Maximize true-positive region recall subject to empirical false-Low fraction <=5%; smallest threshold wins ties. Calibration-only selection.',
                              'guarantee': 'None. Selection-set binomial bounds are descriptive, not valid post-selection guarantees; clustered boxes and model-selection exposure also apply.'}
    for role, y, mu, sigma in [('calibration', labels, means, stds), ('evaluation', test_labels, test_means, test_stds)]:
        for eligible_only in (False, True):
            mask = np.asarray(mu) >= .35 if eligible_only else np.ones(len(y), dtype=bool)
            selected_y, scores = np.asarray(y)[mask], -np.asarray(sigma)[mask]
            values = {'n': int(sum(mask)), 'positive_prevalence': float(np.mean(selected_y)) if len(selected_y) else None,
                      'roc': float(roc_auc_score(selected_y, scores)) if len(set(selected_y)) == 2 else None,
                      'pr_average_precision': float(average_precision_score(selected_y, scores)) if len(set(selected_y)) == 2 else None}
            summary['calibration'][role + ('_confidence_eligible_auc' if eligible_only else '_all_auc')] = values
    for role in ('calibration', 'evaluation'):
        legacy_labels, legacy_means, legacy_stds = [], [], []
        for row, d in data:
            if row['role'] != role or not d['detections']:
                continue
            legacy_labels.extend(region_truth(d['detections'][:1], row['boxes']))
            scores = d['legacy_top_confidences'][:5]
            legacy_means.append(float(np.mean(scores)))
            legacy_stds.append(float(np.std(scores)))
        summary['calibration']['legacy_top_' + role] = operating(legacy_labels, legacy_means, legacy_stds, .03)
    save(OUT / 'calibration_curve.json', {'operating_points': curve, **curves})
    save(OUT / 'calibration.json', {'variance_threshold': chosen['threshold'], 'confidence_threshold': .35, 'tta_passes': 5,
                                  'detector_sha256': digest(config().detector_path), 'manifest_sha256': digest(OUT / 'split_manifest.json'),
                                  'confidence_guarantee': None})
    coverage = {}
    for role in ('evaluation', 'external'):
        subset = [(r, d) for r, d in data if r['role'] == role]
        total = sum(len(d['detections']) for r, d in subset)
        old = sum(bool(d['detections']) for r, d in subset)
        coverage[role] = {'n': len(subset), 'old_handled': old, 'new_handled': total,
                          'old_mean_per_image': old/len(subset), 'new_mean_per_image': total/len(subset),
                          'multi_detection_images': sum(len(d['detections']) > 1 for r, d in subset)}
        if role == 'evaluation':
            coverage[role]['old_correct_localizations'] = sum(sum(region_truth(d['detections'][:1], r['boxes'])) for r, d in subset)
            coverage[role]['new_correct_localizations'] = sum(sum(region_truth(d['detections'], r['boxes'])) for r, d in subset)
            coverage[role]['ground_truth_regions'] = sum(len(r['boxes']) for r, d in subset)
    summary['coverage'] = coverage
    audit = [(r, d) for r, d in data if r['tta_audit']]
    sequences = [reg['tta_confidences'] for r, d in audit for reg in d['detections']]
    ref = np.array([np.std(s) for s in sequences])
    tta = []
    for n in (3, 5, 7, 8, 10, 15):
        estimate = np.array([np.std(s[:n]) for s in sequences])
        tta.append({'n': n, 'mean_std': float(estimate.mean()), 'mean_absolute_error_vs_15': float(np.abs(estimate-ref).mean()),
                    'relative_mae_vs_15': float(np.abs(estimate-ref).mean()/ref.mean()) if ref.mean() else None})
    summary['tta'] = {'audit_images': len(audit), 'regions': len(sequences), 'curve': tta,
                      'first_n_below_10_percent_relative_mae': next((x['n'] for x in tta if x['n'] < 15 and x['relative_mae_vs_15'] <= .1), None),
                      'production_n': 5, 'rationale': 'Retain five passes for the calibrated gate; increased N is an audited alternative, requiring independent recalibration before deployment. N=15 is a reference, not ground truth.'}
    diversity = []
    for row, d in audit:
        im = pipeline.load_image(row['path'])
        original = np.asarray(im).astype(float)
        diversity.append([float(np.abs(np.asarray(frame.resize(im.size)).astype(float)-original).mean()/255)
                          for _, frame in pipeline.deterministic_tta(im, 15)])
    summary['tta']['mean_pixel_change_by_pass'] = dict(zip([name for name, _ in pipeline.deterministic_tta(Image.new('RGB', (100,100)), 15)], np.mean(diversity, axis=0).tolist()))
    g = grounding_benchmark()
    summary['grounding'] = {k: g[k] for k in ('n', 'old', 'new', 'limitation')}
    save(OUT / 'results.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([x['n'] for x in tta], [x['mean_std'] for x in tta], 'o-', label='Mean matched-region std')
    ax.plot([x['n'] for x in tta], [x['mean_absolute_error_vs_15'] for x in tta], 'o-', label='MAE vs N=15')
    ax.set(xlabel='TTA passes', ylabel='Confidence units', title=f'TTA stability: {len(audit)} random images, {len(sequences)} regions')
    ax.legend()
    fig.savefig(OUT / 'tta_curve.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(curves['roc']['fpr'], curves['roc']['tpr'], label=f"ROC AUC={auc['roc']:.3f}")
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Stability-only ROC')
        axes[1].plot(curves['pr']['recall'], curves['pr']['precision'], label=f"AP={auc['pr_average_precision']:.3f}")
        axes[1].axhline(np.mean(labels), linestyle='--', color='gray', label='Trustworthy prevalence')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='Stability-only precision-recall')
        for ax in axes:
            ax.legend()
        fig.suptitle('Calibration regions; chosen gate additionally requires mean confidence >=0.35')
        fig.tight_layout()
        fig.savefig(OUT / 'calibration_curves.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(summary, indent=2))


def external_vlm():
    import torch
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(4)
    source = read('models/VLM_SOURCE.json')
    checkpoint = Path(hf_hub_download(source['repository'], source['source_filename'], revision=pipeline.VLM_REVISION))
    actual_sha = digest(checkpoint)
    assert actual_sha == source['sha256'], 'VLM checkpoint changed'
    assert checkpoint.parent.name == source['resolved_commit'], 'VLM revision changed'
    save(OUT / 'vlm_provenance.json', {'repository': source['repository'], 'resolved_commit': checkpoint.parent.name,
                                     'sha256': actual_sha, 'real_model': True, 'dtype': 'float32', 'device': 'cpu'})
    calibration = read(OUT / 'calibration.json')
    assert calibration['manifest_sha256'] == digest(OUT / 'split_manifest.json')
    assert calibration['detector_sha256'] == digest(config().detector_path)
    threshold = calibration['variance_threshold']
    cfg = replace(config(), variance_threshold=max(0., threshold))
    rows = [r for r in read(OUT / 'split_manifest.json')['rows'] if r['role'] == 'external']
    detector = pipeline._load_detector(cfg)
    for i, row in enumerate(rows):
        target = OUT / 'external_pipeline' / f"{row['id']:04}.json"
        if target.exists():
            assert read(target)['calibration_sha256'] == digest(OUT / 'calibration.json')
            continue
        batch = read(OUT / 'detections' / f"{row['id']:04}.json")
        assert batch['sha256'] == row['sha256'] == digest(row['path'])
        for region in batch['detections']:
            mean, std = gate_values(region)
            region['hallucination_risk'] = pipeline.classify_risk(True, mean, std, .35, threshold)
            region['thresholds'].update(variance=threshold, high_risk_variance=threshold*1.5)
            region['decision_trace'].update(uncertainty_passed=std <= threshold,
                                            risk=region['hallucination_risk'], gate_permitted=region['hallucination_risk'] == 'Low')
        torch.manual_seed(SEED + row['id'])
        random.seed(SEED + row['id'])
        np.random.seed(SEED + row['id'])
        result = pipeline.inspect(row['path'], config=cfg, detector=detector, detection_batch=batch)
        save(target, {'id': row['id'], 'positive': row['positive'], 'seed': SEED + row['id'], 'calibration_sha256': digest(OUT / 'calibration.json'),
                      'cached_detector_seconds': batch['seconds'], **pipeline._json_result(result)})
        print(f"EXTERNAL [{i+1}/{len(rows)}] VLM called={result['vlm_called']}", flush=True)
    records = [read(OUT / 'external_pipeline' / f"{r['id']:04}.json") for r in rows]
    save(OUT / 'external_pipeline_summary.json', {'n': len(records), 'real_vlm': True,
         'images_vlm_called': sum(r['vlm_called'] for r in records),
         'regions_vlm_called': sum(d['vlm_called'] for r in records for d in r['detections']),
         'regions_vlm_succeeded': sum(d['vlm_succeeded'] for r in records for d in r['detections']),
         'regions_failed_closed': sum(d['vlm_called'] and not d['vlm_succeeded'] for r in records for d in r['detections']),
         'geometry_blocks': sum('Grounding verifier' in (d.get('vlm_error') or '') for r in records for d in r['detections']),
         'word_list_blocks': sum('unsupported diagnostic or measurement' in (d.get('vlm_error') or '') for r in records for d in r['detections']),
         'negative_images_gated_low': sum(not r['positive'] and r['vlm_called'] for r in records),
         'language_stage_seconds': sum(r['processing_time'] for r in records)})


def report():
    """Regenerate v2 sections while retaining the complete original pilot documents."""
    result = read(OUT / 'results.json')
    external_path = OUT / 'external_pipeline_summary.json'
    external = read(external_path) if external_path.exists() else None
    m = result['metrics']
    def pct(x):
        return 'undefined' if x is None else f'{100*x:.2f}%'
    old = {'precision': .75, 'recall': 1., 'f1': 6/7, 'accuracy': .8}
    table = '| Metric | Old (n=5/29) | New (n=200 + external n=200) | Delta |\n|---|---|---|---|\n'
    for name, old_value in old.items():
        a, b = m['evaluation'][name], m['external'][name]
        table += f'| {name.title()} | {pct(old_value)} (n=5) | Crack-Seg {pct(a)}; SDNET {pct(b)} | In-domain vs pilot {100*(a-old_value):+.2f} pp; domain gap {100*(b-a):+.2f} pp |\n'
    c = result['calibration']
    cov = result['coverage']['evaluation']
    g = result['grounding']
    table += f"| Regions handled/image | {cov['old_mean_per_image']:.3f} (old algorithm replay, n=200) | {cov['new_mean_per_image']:.3f} | {cov['new_mean_per_image']-cov['old_mean_per_image']:+.3f} |\n"
    table += f"| Correctly localized regions/image | {cov['old_correct_localizations']/200:.3f} (old algorithm replay, n=200) | {cov['new_correct_localizations']/200:.3f} | {(cov['new_correct_localizations']-cov['old_correct_localizations'])/200:+.3f} |\n"
    table += f"| Grounding hallucination recall | {pct(g['old']['recall'])} (word-list replay, n=30) | {pct(g['new']['recall'])} | {100*(g['new']['recall']-g['old']['recall']):+.2f} pp |\n"
    table += f"| TTA-std threshold | 0.030000 | {c['chosen']['threshold']:.6f} | {c['chosen']['threshold']-.03:+.6f} |\n"
    table += f"| Stability ROC-AUC / PR-AUC (AP) | Not measured | {c['auc']['roc']:.4f} / {c['auc']['pr_average_precision']:.4f} | Newly measured |\n"
    body = """## V2 Evaluation — measured results and limits

**Task 1 is not fully satisfiable with this checkpoint and dataset.** The source has 3,717 training, 200 validation and 112 test images. Only one image has an empty label, in validation, and that image was already used in the n=29 pilot. Validation also participated in checkpoint selection. The new 200-image benchmark is a seeded, stratified sample of non-training partitions (199 positive, 1 negative), **not 200 newly untouched held-out images**. No training images or invented negatives were added to meet the count. The pilot is superseded for descriptive benchmark reporting, but its historical records remain below.

The remaining 111 unique non-training images calibrate the gate and are disjoint from the 200 scoring images. One exact duplicate was excluded before sampling. Selection uses labels, seed 20260907 and no detector scores. Bootstrap CIs condition on the observed class proportions; the lone in-domain negative cannot establish useful real-world false-positive performance, regardless of narrow aggregate intervals. A genuinely fresh balanced test set requires additional annotations/data or retraining a checkpoint on a newly partitioned corpus.

External validation uses 200 randomly sampled SDNET2018 patches (100 cracked, 100 non-cracked). [The original authors](https://digitalcommons.usu.edu/all_datasets/48/) publish SDNET2018 under CC BY 4.0: Maguire, Dorafshan & Thomas (2018), Utah State University, DOI 10.15142/T3TD19. The official binary endpoint returned HTTP 403; the public Kaggle mirror is recorded in the manifest, while the original license takes precedence over the mirror's CC0 tag. This is a separate source dataset, and selected exact-byte training overlaps are excluded. The checkpoint's full source-image lineage is unavailable: **never-seen status cannot be guaranteed against resized/cropped derivatives or pretraining exposure**. SDNET patches can share parent photographs, so image-level intervals can understate scene-level uncertainty.

### Detection metrics

Positive prediction means at least one original-pass box at the unchanged 0.01 inference floor, matching the pilot's `detected` definition. These are image-level crack-presence metrics, not segmentation IoU or language factuality. JSON includes 5,000 stratified image bootstrap resamples; these degenerate to zero width for an all-positive predictor. The primary intervals below instead propagate simultaneous 97.5% exact binomial intervals for sensitivity and specificity through each metric at the fixed sampled class prevalence (Bonferroni gives at least 95% joint coverage under independent images). This also provides a nondegenerate F1 interval. Class prevalence differs sharply between datasets; the domain gap includes this prevalence effect.

| Dataset | Metric | Estimate | Conservative 95% CI | Full CI width |\n|---|---|---:|---|---:|\n"""
    for role in ('evaluation', 'external'):
        for name in old:
            ci = m[role]['simultaneous_rate_ci95'][name]
            body += f"| {role} | {name} | {pct(m[role][name])} | [{pct(ci['low'])}, {pct(ci['high'])}] | {pct(ci['width'])} |\n"
    for role in ('evaluation', 'external'):
        r = m[role]
        body += f"\n{role}: TP={r['tp']}, FP={r['fp']}, TN={r['tn']}, FN={r['fn']}. At confidence >=0.35, precision/recall/F1 = {pct(r['at_confidence_035']['precision'])}/{pct(r['at_confidence_035']['recall'])}/{pct(r['at_confidence_035']['f1'])}.\n"
        body += '\nExact binomial 95% intervals (image-level independence assumed): '
        body += '; '.join(f"{name} [{pct(ci[0])}, {pct(ci[1])}]" for name, ci in r['exact_binomial_ci95'].items() if ci) + '.\n'
    body += f"""
### Calibration and false-explanation proxy

Trust means a unique detection matches a ground-truth polygon's enclosing box at IoU >=0.5, with greedy one-to-one matching. It is an operational localization proxy, not a human label of language truth and not the old risk label. Calibration contains {c['n_regions']} detections, {c['trustworthy_regions']} trustworthy. Stability alone uses score `-TTA_std`; ROC-AUC={c['auc']['roc']:.4f}, PR-AUC (average precision)={c['auc']['pr_average_precision']:.4f}. AUC-PR is reported as average precision rather than trapezoidal area.

Selection maximizes recall of trustworthy regions subject to <=5% empirical false detections among Low gates; lower false fraction and then smaller threshold break ties. The confidence floor for Low remains 0.35. ROC/PR arrays and every candidate operating point are in `evidence/v2/calibration_curve.json`. Chosen std threshold: **{c['chosen']['threshold']:.6f}**. The deployable artifact is `models/gate_calibration.json`; default CLI/UI inspection checks its detector hash before use. Medium/High retain the existing 1.5x std boundary, which is a policy convention, not a calibrated probability.

| Sample / operating point | Low regions | False Low | False-explanation proxy | True-Low recall | One-sided 95% binomial upper bound |\n|---|---:|---:|---:|---:|---:|\n"""
    for key in ('legacy_top_calibration', 'old', 'chosen', 'legacy_top_evaluation', 'evaluation_old', 'evaluation_chosen'):
        point = c[key]
        body += f"| {key} | {point['low_count']} | {point['false_low_count']} | {pct(point['false_explanation_rate_proxy'])} | {pct(point['true_low_recall'])} | {pct(point['one_sided_95_upper'])} |\n"
    body += "\n`legacy_top_*` replays the original highest-confidence-per-pass algorithm at 0.03. `old`/`evaluation_old` apply 0.03 to the new matched per-region sequences, isolating the threshold change. The denominators differ between top-only and all-region policies.\n"
    body += '\n| AUC population | Regions | Trustworthy prevalence | ROC-AUC | PR-AUC (AP) |\n|---|---:|---:|---:|---:|\n'
    for key in ('calibration_all_auc', 'calibration_confidence_eligible_auc', 'evaluation_all_auc', 'evaluation_confidence_eligible_auc'):
        values = c[key]
        body += f"| {key} | {values['n']} | {pct(values['positive_prevalence'])} | {values['roc']:.4f} | {values['pr_average_precision']:.4f} |\n"
    body += '\n![Calibration ROC and PR curves](evidence/v2/calibration_curves.png)\n'
    body += '\nStability alone ranks trust worse than chance across all floor-level proposals. Very low-confidence false proposals can also have very low variance. The confidence-conditioned AUC is stronger, supporting the combined confidence-and-stability gate rather than treating TTA std as a calibrated trust probability.\n'
    body += "\n**No <=5% at 95% confidence guarantee is claimed.** Selection-set binomial bounds are descriptive after threshold search, boxes cluster within images, and the evaluation includes model-selection exposure. Even an independent zero-error sample needs at least 59 accepted independent cases for a one-sided exact 95% upper bound <=5%. The stretch KPI is not achieved. Calibration is never fitted on SDNET2018.\n"
    body += '\n### Per-detection coverage\n\nFive image-level passes are shared for efficiency. Every base box at the 0.01 floor gets its own matched confidence sequence, risk, crop and language decision. Boxes are inverse-transformed before greedy highest-IoU matching (minimum IoU 0.3), each augmented box is used at most once, and missing matches contribute zero. Secondary regions appear in JSON, numbered annotations and the UI. Image risk is the highest region risk; scalar confidence/box fields retain top-region compatibility.\n\n'
    body += '| Set | Old handled/image | New handled/image | Multi-detection images |\n|---|---:|---:|---:|\n'
    for role, r in result['coverage'].items():
        body += f"| {role} | {r['old_mean_per_image']:.3f} | {r['new_mean_per_image']:.3f} | {r['multi_detection_images']}/{r['n']} |\n"
    body += f"\nCorrectly localized regions on the 200-image benchmark increase from {cov['old_correct_localizations']} ({cov['old_correct_localizations']/200:.3f}/image) to {cov['new_correct_localizations']} ({cov['new_correct_localizations']/200:.3f}/image), against {cov['ground_truth_regions']} labeled regions. Processing all predictions does not imply all are correct. SDNET has image labels, so external per-region correctness is unavailable.\n"
    body += f"""
### Grounding verifier

The verifier uses crop-relative bounding-box center, width, height, area, aspect ratio and TTA-mean confidence. Explicit location/orientation and percentage claims are checked before surfacing. The old diagnostic blocklist remains in force. Contradictions fail closed. Box orientation is only a geometric proxy; branching, appearance, severity, vague paraphrases and all general semantic hallucinations are not certified. The checker records which claims were checked, including an empty list when none were measurable.

The fixed 30-output challenge set contains 15 supported and 15 hallucinated outputs across three geometry templates. Old guard: precision {pct(g['old']['precision'])}, recall {pct(g['old']['recall'])}, F1 {pct(g['old']['f1'])}. New verifier: precision {pct(g['new']['precision'])}, recall {pct(g['new']['recall'])}, F1 {pct(g['new']['f1'])}. These are real executions on authored adversarial text, **not estimates of natural Moondream hallucination prevalence**; template correlation and narrow vocabulary limit generalization. All prompts, labels, measurements and verdicts are in `grounding_benchmark.json`.

### TTA diversity and pass count

The existing five passes were already diverse in kind: original, horizontal flip, brightness 0.8, contrast 1.15 and rotation +3 degrees. The new audit adds vertical flip, scale 0.85/1.15, color 0.5/1.5, brightness 1.2, contrast 0.85 and rotations -3/+7/-7. Deterministic transforms are reproducible perturbations, not independent posterior samples. Color/contrast can be weak on gray or uniform surfaces. Normalized pixel-change measurements for every transform are in `results.json`.

Thirty randomly sampled images ({result['tta']['regions']} base detections) use a nested 15-pass sequence. Every N is compared on the same regions, with missing matches retained as zero. The reference is N=15, not the true uncertainty.

| N | Mean std | Mean absolute error vs N=15 | Relative MAE vs N=15 |\n|---:|---:|---:|---:|\n"""
    for point in result['tta']['curve']:
        body += f"| {point['n']} | {point['mean_std']:.5f} | {point['mean_absolute_error_vs_15']:.5f} | {pct(point['relative_mae_vs_15'])} |\n"
    convergence = result['tta']['first_n_below_10_percent_relative_mae'] or 'not reached before N=15'
    body += f"\nFirst tested N below 10% relative MAE (excluding the reference itself): **{convergence}**. Production retains N=5 because this is the evaluated/calibrated operating point; the curve does not by itself establish a better gate at larger N. This is a latency/calibration choice, not a claim that five passes have converged. **The convergence target is not achieved.** Increasing production N requires recalibrating and scoring the entire gate at that N; N=15 agreeing with itself is not evidence of convergence.\n\n![TTA stability curve](evidence/v2/tta_curve.png)\n"
    body += '\n### Real external language-stage execution\n\n'
    if external:
        body += f"All {external['n']} external images passed through `inspect()` using replayed, hashed real detector results and the real pinned Moondream2 loader. There were {external['regions_vlm_called']} region-level VLM calls across {external['images_vlm_called']} images, {external['regions_vlm_succeeded']} surfaced explanations and {external['regions_failed_closed']} failed-closed outputs. Negative images incorrectly gated Low: {external['negative_images_gated_low']}. Language-stage wall time (including loading) was {external['language_stage_seconds']:.1f}s; cached detector times are recorded separately. No stub VLM was used for this evaluation.\n"
    else:
        body += '**Not yet completed; do not interpret detector-only external metrics as full-pipeline execution.**\n'
    body += """
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
"""
    body += f"\nSplit manifest SHA-256: `{digest(OUT / 'split_manifest.json')}`. Machine-readable results: `evidence/v2/results.json`; detector/model/config provenance: `inference_provenance.json`.\n"
    overlap_path = OUT / 'external_overlap_audit.json'
    if overlap_path.exists():
        overlap = read(overlap_path)
        body += f"\nThe additional external overlap screen compared {overlap['n']} SDNET patches with {overlap['training_n']} training images using a 63-bit grayscale DCT hash, including horizontal mirrors. There were {overlap['suspect_distance_le_4']} candidates at Hamming distance <=4; the minimum distance was {overlap['minimum_distance']}. This is supporting evidence, not proof against crops or unknown pretraining. Reproduce with `python evaluate_v2.py overlap`; full nearest-neighbor records are in `external_overlap_audit.json`.\n"
    body += '\nThe exact inference source snapshot is retained in `evidence/v2/source/pipeline.py`. Cache resume permits comment/formatting-only changes when parsed executable ASTs match; semantic changes require a new output directory. The original executed-source hash remains in inference provenance.\n'
    test_log = OUT / 'pytest_final.log'
    if test_log.exists():
        raw = test_log.read_bytes()
        text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
        test_summary = next((line.strip() for line in text.splitlines() if ' passed in ' in line), 'See pytest_final.log')
        body += f'\nFinal pytest verification: **{test_summary}**. The saved log is `evidence/v2/pytest_final.log`.\n'
    integrity_path = OUT / 'final_integrity_check.json'
    if integrity_path.exists():
        integrity = read(integrity_path)
        body += f"\nArtifact integrity audit: **{integrity['status']}**, checking all {integrity['images']} input hashes, external gate boundaries, installed calibration and preserved pilot sections. See `final_integrity_check.json`.\n"
    for name in ('REPORT.md', 'TEST_RESULTS.md', 'SENSITIVITY.md'):
        path = Path(name)
        archive = OUT / 'pilot' / name
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(path.read_bytes())
        pilot = archive.read_text(encoding='utf-8')
        path.write_text('# ' + name.replace('.md', '').replace('_', ' ').title() + '\n\n' + table + '\n' + body + '\n---\n\n## Pilot Evaluation (superseded)\n\nThe following is preserved verbatim from the pre-v2 report, including its historical status statements.\n\n' + pilot, encoding='utf-8')
    save(OUT / 'report_provenance.json', {'evaluate_v2_sha256': digest(__file__), 'pipeline_sha256': digest('pipeline.py'),
                                        'pilot_hashes': {name: digest(OUT / 'pilot' / name) for name in ('REPORT.md', 'TEST_RESULTS.md', 'SENSITIVITY.md')}})


def deploy():
    """Install the measured operating point after the real external run completes."""
    calibration = read(OUT / 'calibration.json')
    completed = read(OUT / 'external_pipeline_summary.json')
    assert completed['n'] >= 100 and completed['real_vlm']
    assert calibration['detector_sha256'] == digest(config().detector_path)
    assert calibration['manifest_sha256'] == digest(OUT / 'split_manifest.json')
    if calibration['variance_threshold'] < 0:
        raise RuntimeError('No feasible operating point: do not deploy an invalid threshold.')
    save('models/gate_calibration.json', calibration)
    print('Installed models/gate_calibration.json; no statistical confidence guarantee is asserted.')


def main():
    global OUT, ROOT, EXTERNAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['download', 'prepare', 'detect', 'summarize', 'external-vlm', 'grounding', 'report', 'overlap', 'deploy'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--crack-root', type=Path, default=ROOT)
    parser.add_argument('--external-root', type=Path, default=EXTERNAL)
    args = parser.parse_args()
    OUT, ROOT, EXTERNAL = args.output, args.crack_root, args.external_root
    {'download': download, 'prepare': prepare, 'detect': detect, 'summarize': summarize, 'external-vlm': external_vlm, 'grounding': grounding_benchmark, 'report': report, 'overlap': overlap_audit, 'deploy': deploy}[args.phase]()


if __name__ == '__main__':
    main()
