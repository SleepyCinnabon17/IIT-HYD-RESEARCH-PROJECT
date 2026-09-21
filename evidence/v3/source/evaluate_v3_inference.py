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

OUT = Path('evidence/v3')
SEED = 20260921


def prepare():
    old = v2.read(v2.OUT / 'split_manifest.json')
    seen = {r['sha256'] for r in old['rows']}
    seen.update(v2.digest(p) for p in (v2.ROOT / 'images/train').glob('*'))
    pools = {True: [], False: []}
    for p in sorted(v2.EXTERNAL.rglob('*.jpg')):
        if p.parent.name.upper() in {'CD','UD','CW','UW','CP','UP','CRACKED','NON-CRACKED','UNCRACKED'}:
            pools[p.parent.name.upper().startswith('C')].append(p)
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
            chosen.append({'path':str(p.resolve()),'relative_path':p.relative_to(v2.EXTERNAL).as_posix(),
                           'sha256':sha,'positive':positive})
            if len(chosen)==200:
                break
        assert len(chosen)==200
        for i,r in enumerate(chosen):
            r.update(role='calibration' if i<100 else 'fresh_test',id=len(rows))
            rows.append(r)
    v2.save(OUT/'manifest.json',{'seed':SEED,'rows':rows,'v2_manifest_sha256':v2.digest(v2.OUT/'split_manifest.json'),
        'limitations':['Patch-level split; parent-scene independence is not established.',
                      'SDNET is now used for calibration, so this is domain-adapted evaluation, not unseen-domain validation.',
                      'Image labels do not validate individual boxes or segmentation.']})


def infer():
    import torch
    torch.set_num_threads(2)
    cfg=pipeline.PipelineConfig(device='cpu')
    provenance={'config':asdict(cfg),'weights_sha256':v2.digest(Path(cfg.detector_path)),
                'manifest_sha256':v2.digest(OUT/'manifest.json'),'pipeline_sha256':v2.digest(Path('pipeline.py')),
                'evaluation_sha256':v2.digest(Path(__file__))}
    path=OUT/'inference_provenance.json'
    if path.exists():
        assert v2.read(path)==provenance,'Stale inference cache'
    v2.save(path,provenance)
    detector=pipeline._load_detector(cfg)
    for row in v2.read(OUT/'manifest.json')['rows']:
        target=OUT/'detections'/f"{row['id']:04}.json"
        if target.exists():
            continue
        assert v2.digest(Path(row['path']))==row['sha256']
        with Image.open(row['path']) as im:
            regions=pipeline._predict_all(detector,pipeline.load_image(im),cfg)
        v2.save(target,{'sha256':row['sha256'],'detections':[{'raw_confidence':r.confidence,'box':r.box} for r in regions]})
        print(f"{row['id']+1}/400 {row['role']}",flush=True)


def examples(role):
    return [(r,v2.read(OUT/'detections'/f"{r['id']:04}.json")['detections'])
            for r in v2.read(OUT/'manifest.json')['rows'] if r['role']==role]


def calibrate():
    rows=examples('calibration')
    y=[r['positive'] for r,d in rows]
    scores=np.array([max((d['raw_confidence'] for d in ds),default=0) for r,ds in rows])
    # Predeclared objective: maximize balanced-sample accuracy, retain >=85% sensitivity.
    curve=[]
    for threshold in np.round(np.arange(.01,.351,.005),3):
        m=v2.metrics(y,scores>=threshold)
        curve.append({'threshold':float(threshold),**m})
    eligible=[m for m in curve if m['recall']>=.85]
    chosen=max(eligible,key=lambda m:(m['accuracy'],m['recall'],-m['threshold']))
    v2.save(OUT/'calibration_curve.json',curve)
    v2.save(OUT/'operating_point.json',{'raw_confidence_threshold':chosen['threshold'],
        'selection':'Maximum calibration accuracy subject to recall >=85%; ties favor recall then lower threshold.',
        'calibration':chosen,'manifest_sha256':v2.digest(OUT/'manifest.json'),
        'detector_sha256':v2.digest(Path(pipeline.PipelineConfig().detector_path)),
        'confidence_guarantee':None})
    print(json.dumps(chosen,indent=2))


def report():
    artifact=v2.read(OUT/'operating_point.json'); threshold=artifact['raw_confidence_threshold']
    sets={'fresh_test':examples('fresh_test')}
    old=v2.read(v2.OUT/'split_manifest.json')
    for role in ('evaluation','external'):
        sets['v2_'+role]=[(r,v2.read(v2.OUT/'detections'/f"{r['id']:04}.json")['detections']) for r in old['rows'] if r['role']==role]
    results={'operating_point':artifact,'datasets':{}}
    for name,rows in sets.items():
        y=[r['positive'] for r,d in rows]
        entry={}
        for label,t in [('before',.01),('after',threshold)]:
            pred=[any(d['raw_confidence']>=t for d in ds) for r,ds in rows]
            entry[label]=v2.bootstrap_metrics(y,pred)
            entry[label]['boxes_per_image']=sum(sum(d['raw_confidence']>=t for d in ds) for r,ds in rows)/len(rows)
            if name=='v2_evaluation':
                selected=[([d for d in ds if d['raw_confidence']>=t],r['boxes']) for r,ds in rows]
                tp=sum(sum(v2.region_truth(ds,boxes)) for ds,boxes in selected)
                n=sum(len(ds) for ds,b in selected); total=sum(len(b) for ds,b in selected)
                entry[label]['localization']={'tp':tp,'predictions':n,'gt':total,'precision':tp/n if n else 0,'recall':tp/total,'f1':2*tp/(n+total)}
        rng=np.random.default_rng(SEED)
        before=np.array([bool(ds) for r,ds in rows])==y
        after=np.array([any(d['raw_confidence']>=threshold for d in ds) for r,ds in rows])==y
        delta=after.astype(float)-before.astype(float)
        draws=[delta[rng.integers(0,len(rows),len(rows))].mean() for _ in range(5000)]
        entry['accuracy_delta']={'estimate':float(delta.mean()),'paired_bootstrap_ci95':np.quantile(draws,[.025,.975]).tolist()}
        results['datasets'][name]=entry
    v2.save(OUT/'results.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['prepare','infer','calibrate','report'])
    globals()[parser.parse_args().step]()
