#!/usr/bin/env python3
"""Measure a frozen final-pass RGB ablation; this is not a new progressive run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.protocol import failure_metrics
from amodal_benchmark.references import reference_paths


def derive_raw_lastpass(source, destination, record):
    """Keep final alpha/size and the frozen choice; remove only RGB correction."""
    if record['status'] != 'success':
        raise ValueError('A saved completion is required')
    trace = record['diagnostics']['surface_iterations'][-1]
    folder = source/'surface'/f'iteration-{trace["iteration"]:02d}'
    decision_path = folder/'decision.json'
    if sha256(decision_path) != trace['decision_sha256']:
        raise ValueError('Final decision does not match the completed target')
    decision = json.loads(decision_path.read_text())
    if decision['index'] != trace['selected']:
        raise ValueError('Selected candidate changed')
    candidate = folder/f'candidate-{decision["index"]}.png'
    if sha256(candidate) != decision['artifacts'][candidate.name]:
        raise ValueError('Raw candidate checksum mismatch')
    if sha256(source/'completion.png') != record['completion_sha256']:
        raise ValueError('Completed surface output checksum mismatch')
    with Image.open(source/'completion.png') as image:
        completed = image.convert('RGBA')
    with Image.open(source/'mask.png') as image:
        alpha = image.convert('L')
    if not np.array_equal(np.asarray(completed)[..., 3], np.asarray(alpha)):
        raise ValueError('Final alpha differs from its stored mask')
    r0, r1, c0, c1 = decision['crop_rc']
    if completed.size != (c1-c0, r1-r0):
        raise ValueError('Final candidate crop differs from the saved output frame')
    with Image.open(candidate) as image:
        raw = image.convert('RGB').resize(completed.size, Image.Resampling.BICUBIC)
    destination.mkdir(parents=True, exist_ok=True)
    jpeg = destination/'raw-final.jpg'
    raw.save(jpeg, quality=90)
    with Image.open(jpeg) as image:
        output = image.convert('RGBA')
    output.putalpha(alpha)
    path = destination/'raw-final.png'
    output.save(path)
    return path, dict(decision_sha256=sha256(decision_path),
        selected_index=decision['index'], iteration=trace['iteration'],
        raw_candidate_sha256=sha256(candidate), fixed_mask_sha256=sha256(source/'mask.png'),
        corrected_completion_sha256=record['completion_sha256'], derived_sha256=sha256(path),
        size=list(output.size), unchanged_alpha=True, unchanged_geometry=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'surface', 'dataset-root', 'output-dir', 'freeze'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=4)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for name, expected in freeze['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Frozen ablation source changed: '+name)
    for name, folder in (('baseline', args.baseline), ('surface', args.surface)):
        for filename, expected in freeze['runs'][name].items():
            if sha256(folder/filename) != expected:
                raise ValueError('Frozen source run changed: '+name+'/'+filename)
    rows, _ = read_official(args.dataset_root)
    if [row['id'] for row in rows] != freeze['target_ids']:
        raise ValueError('Ablation target inventory changed')
    import torch
    from amodal_benchmark.metrics import MetricModels
    from amodal_benchmark.provenance import environment_snapshot
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = dict(started_at=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(0),
        worker=args.worker_index, freeze_sha256=sha256(args.freeze),
        environment=environment_snapshot(), scope=freeze['scope'])
    runtime_path = args.output_dir/f'runtime-worker-{args.worker_index}.json'
    write_json(runtime_path, runtime)
    metric_config = json.loads((args.surface/'metric_config.json').read_text())
    if json.loads((args.baseline/'metric_config.json').read_text()) != metric_config:
        raise ValueError('Metric operations differ')
    models = MetricModels()
    for row in rows[args.worker_index::args.worker_count]:
        folders = {name: folder/'targets'/row['id'] for name, folder in
                   (('baseline', args.baseline), ('surface', args.surface))}
        records = {name:json.loads((folder/'result.json').read_text()) for name,folder in folders.items()}
        for key in ('seed', 'input_sha256', 'evaluation_reference_artifacts'):
            if records['baseline'][key] != records['surface'][key]:
                raise ValueError('Paired identity differs: '+row['id']+'/'+key)
        destination = args.output_dir/'targets'/row['id']
        result_path = destination/'ablation.json'
        expected = dict(freeze_sha256=sha256(args.freeze),
            baseline_result_sha256=sha256(folders['baseline']/'result.json'),
            surface_result_sha256=sha256(folders['surface']/'result.json'))
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing['inputs'] != expected:
                raise ValueError('Ablation inputs changed on resume')
            if existing.get('derivation') and sha256(destination/'raw-final.png') != existing['derivation']['derived_sha256']:
                raise ValueError('Derived image changed on resume')
            continue
        scores, cached = {}, {}
        for name in ('baseline', 'surface'):
            record = records[name]
            if record['status'] != 'success':
                scores[name] = failure_metrics()
                cached[name] = failure_metrics()
                continue
            folder = folders[name]
            if sha256(folder/'completion.png') != record['completion_sha256']:
                raise ValueError('Completed source output changed')
            run = args.baseline if name=='baseline' else args.surface
            paths = reference_paths(run/'visible_references', row)
            key = 'crop' if metric_config['alignment']=='released' else 'full'
            if sha256(paths[key]) != record['evaluation_reference_artifacts'][key]:
                raise ValueError('Reference changed')
            scores[name] = models.score(paths[key], folder/'completion.png', row['label'],
                                       record['geometry'], metric_config['alignment'])
            cached[name] = json.loads((folder/'metrics.json').read_text())['metrics']
        record = records['surface']
        derivation = None
        if record['status']=='success':
            raw, derivation = derive_raw_lastpass(folders['surface'], destination, record)
            paths = reference_paths(args.surface/'visible_references', row)
            key = 'crop' if metric_config['alignment']=='released' else 'full'
            scores['raw_final'] = models.score(paths[key], raw, row['label'],
                                              record['geometry'], metric_config['alignment'])
        else:
            scores['raw_final'] = failure_metrics()
        write_json(result_path, dict(id=row['id'], source=row['source'],
            image_identity=row['source']+'/'+row['original_id'], inputs=expected,
            baseline_status=records['baseline']['status'], surface_status=record['status'],
            scores=scores, prior_cached_scores=cached, derivation=derivation))
        print('MEASURED', row['id'], record['status'], flush=True)
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(runtime_path, runtime)


if __name__ == '__main__':
    main()
