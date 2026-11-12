#!/usr/bin/env python3
"""Extract observable pretrained evidence; evaluation files are never opened."""
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
from amodal_benchmark.surface_features import ObservedSurfaceFeatures
from measure_surface_followup import FollowupMetrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dreamsim-cache', type=Path, required=True)
    p.add_argument('--extra-run-dir', type=Path)
    p.add_argument('--candidate-seeds', type=int, nargs='+', default=[11, 22, 33, 44])
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=4)
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cases = json.loads((a.experiment_dir / 'generation_manifest.json').read_text())
    contract = dict(source_sha256=sha256(Path(__file__)),
        features_sha256=sha256(ROOT / 'amodal_benchmark/surface_features.py'),
        observation_manifest_sha256=sha256(a.experiment_dir / 'generation_manifest.json'),
        dreamsim_manifest_sha256=sha256(a.dreamsim_cache / 'manifest.json'),
        candidate_seeds=a.candidate_seeds, truth_access=False)
    runtime = dict(**contract, started_at=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ.get('SLURM_JOB_ID'), worker=a.worker_index,
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    write_json(a.output_dir / f'runtime-worker-{a.worker_index}.json', runtime)
    extractor = ObservedSurfaceFeatures(FollowupMetrics(a.dreamsim_cache))
    for case in cases[a.worker_index::a.worker_count]:
        destination = a.output_dir / (case['case_id'] + '.json')
        if destination.exists():
            if json.loads(destination.read_text())['contract'] != contract:
                raise ValueError('Changed feature contract on resume')
            continue
        inputs = a.experiment_dir / 'inputs' / case['case_id']
        folder = a.run_dir / 'candidates' / case['case_id']
        for name, expected in case['input_hashes'].items():
            if sha256(inputs / name) != expected:
                raise ValueError('Observation checksum mismatch')
        observed = np.array(Image.open(inputs / 'observed.png').convert('RGB'))
        visible = np.array(Image.open(inputs / 'visible.png').convert('L')) > 0
        candidates, masks, hashes = [], [], {}
        for seed in a.candidate_seeds:
            candidate_folder = folder
            if not (folder / f'seed-{seed}.json').exists() and a.extra_run_dir:
                candidate_folder = a.extra_run_dir / 'candidates' / case['case_id']
            record_path = candidate_folder / f'seed-{seed}.json'
            record = json.loads(record_path.read_text())
            hashes[record_path.name] = sha256(record_path)
            for name, expected in record['artifacts'].items():
                if sha256(candidate_folder / name) != expected:
                    raise ValueError('Candidate checksum mismatch')
            rgb, mask = candidate_folder / f'seed-{seed}.png', candidate_folder / f'seed-{seed}-mask.png'
            candidates.append(np.array(Image.open(rgb).convert('RGB')) if rgb.exists() else np.zeros_like(observed))
            masks.append(np.array(Image.open(mask).convert('L')) > 0 if mask.exists() else np.zeros_like(visible))
        values = extractor(observed, visible, candidates, masks, case['label'])
        write_json(destination, dict(case_id=case['case_id'], contract=contract,
                   candidate_record_sha256=hashes, values=values))
        print('FEATURES', case['case_id'], flush=True)
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(a.output_dir / f'runtime-worker-{a.worker_index}.json', runtime)


if __name__ == '__main__':
    main()
