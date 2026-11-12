#!/usr/bin/env python3
"""Review a consensus correction using out-of-fold choices on old development images."""
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
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_refinement import RefinementConfig, refine_surface
from amodal_benchmark.surface_ensemble import EnsembleConfig, refine_surface_pool
from measure_surface_followup import FollowupMetrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--extra-run-dir', type=Path)
    p.add_argument('--search', type=Path, required=True)
    p.add_argument('--configs', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dreamsim-cache', type=Path, required=True)
    p.add_argument('--candidate-seeds', type=int, nargs='+', default=[11,22,33,44])
    p.add_argument('--all-candidates', action='store_true',
                   help='Measure the complete common pool for joint development-only selection')
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=4)
    a = p.parse_args()
    experiment = a.experiment_dir.resolve()
    if experiment.name not in ('surface-consistency-20261108','surface-consistency-replication87-20261108'):
        raise ValueError('Correction development is restricted to the designated old cohorts')
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    choices = json.loads(a.search.read_text())['candidates'][0]['predictions']
    configs = json.loads(a.configs.read_text())
    cases = json.loads((experiment / 'cases.json').read_text())
    contract = dict(search_sha256=sha256(a.search),configs_sha256=sha256(a.configs),
        source_sha256=sha256(Path(__file__)),ensemble_sha256=sha256(ROOT / 'amodal_benchmark/surface_ensemble.py'),
        candidate_seeds=a.candidate_seeds,development_only=True,all_candidates=a.all_candidates)
    runtime = dict(**contract,job_id=os.environ.get('SLURM_JOB_ID'),worker=a.worker_index,
        started_at=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(0))
    write_json(a.output_dir / f'runtime-worker-{a.worker_index}.json',runtime)
    models = FollowupMetrics(a.dreamsim_cache)
    for case in cases[a.worker_index::a.worker_count]:
        destination = a.output_dir / (case['case_id'] + '.json')
        if destination.exists():
            if json.loads(destination.read_text())['contract'] != contract:
                raise ValueError('Changed development configuration on resume')
            continue
        inputs = experiment / 'inputs' / case['case_id']
        observed = np.array(Image.open(inputs / 'observed.png').convert('RGB'))
        visible = np.array(Image.open(inputs / 'visible.png').convert('L')) > 0
        candidates,masks = [],[]
        for seed in a.candidate_seeds:
            folder = a.run_dir / 'candidates' / case['case_id']
            if not (folder / f'seed-{seed}.json').exists() and a.extra_run_dir:
                folder = a.extra_run_dir / 'candidates' / case['case_id']
            record = json.loads((folder / f'seed-{seed}.json').read_text())
            if record['input_hashes'] != case['input_hashes']:
                raise ValueError('Candidate observation mismatch')
            for name,expected in record['artifacts'].items():
                if sha256(folder / name) != expected:
                    raise ValueError('Candidate checksum mismatch')
            candidates.append(np.array(Image.open(folder / f'seed-{seed}.png').convert('RGB')))
            masks.append(np.array(Image.open(folder / f'seed-{seed}-mask.png').convert('L')) > 0)
        index = choices[case['case_id']]
        calibrated = [refine_surface(observed,visible,image,mask,RefinementConfig(mode='affine',strength=1.))[0]
                      for image,mask in zip(candidates,masks)]
        pools = {'calibrated':calibrated}
        for name,config in configs.items():
            images,_ = refine_surface_pool(calibrated,masks,visible,EnsembleConfig(**config))
            pools[name] = images
        variants = {name:images[index] for name,images in pools.items()}
        evaluation = experiment / 'evaluation_only' / case['case_id']
        for name,expected in case['evaluation_hashes'].items():
            if sha256(evaluation / name) != expected:
                raise ValueError('Development truth checksum mismatch')
        truth = np.array(Image.open(evaluation / 'truth.png').convert('RGB'))
        hidden = np.array(Image.open(evaluation / 'hidden_core.png').convert('L')) > 0
        full = np.array(Image.open(evaluation / 'full.png').convert('L')) > 0
        reference = models.reference(truth,full,hidden,case['label'])
        if a.all_candidates:
            candidate_values = {name:[models.score(image,mask,reference) for image,mask in zip(images,masks)]
                                for name,images in pools.items()}
            values = {name:items[index] for name,items in candidate_values.items()}
        else:
            candidate_values = None
            values = {name:models.score(image,masks[index],reference) for name,image in variants.items()}
        artifacts = {}
        for name,image in variants.items():
            path = a.output_dir / 'images' / case['case_id'] / (name + '.png')
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(image).save(path)
            artifacts[name] = sha256(path)
        write_json(destination,dict(case_id=case['case_id'],source=case['source'],index=index,
            contract=contract,metrics=values,candidate_metrics=candidate_values,artifacts=artifacts))
        print('ENSEMBLE_METRICS',case['case_id'],flush=True)
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(a.output_dir / f'runtime-worker-{a.worker_index}.json',runtime)


if __name__ == '__main__':
    main()
