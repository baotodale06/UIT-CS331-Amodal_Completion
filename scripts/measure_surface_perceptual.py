#!/usr/bin/env python3
"""Evaluation-only masked spatial LPIPS; never used for candidate selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=4)
    args = p.parse_args()
    import torch
    import lpips
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = lpips.LPIPS(net='alex', spatial=True).eval().to('cuda')
    experiment, run = args.experiment_dir.resolve(), args.run_dir.resolve()
    cases = json.loads((experiment / 'cases.json').read_text())[args.worker_index::args.worker_count]
    seeds = json.loads((experiment / 'protocol.json').read_text())['candidate_seeds']
    output = experiment / 'perceptual'
    output.mkdir(exist_ok=True)
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(), job_id=os.environ.get('SLURM_JOB_ID'),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    source_sha256=sha256(Path(__file__)), cases_sha256=sha256(experiment / 'cases.json'),
                    metric='LPIPS AlexNet v0.1 spatial map averaged on hidden core; RGB outside core=127; native 512 frame',
                    checkpoint_hashes={str(path): sha256(path) for path in
                        (Path(os.environ['TORCH_HOME']) / 'hub/checkpoints').glob('alexnet*.pth')},
                    linear_weights_sha256=sha256(Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'))
    write_json(output / f'runtime-worker-{args.worker_index}.json', metadata)
    for case in cases:
        destination = output / (case['case_id'] + '.json')
        if destination.exists():
            raise ValueError('Refuse to overwrite existing perceptual metrics')
        evaluation = experiment / 'evaluation_only' / case['case_id']
        for filename in ('truth.png', 'hidden_core.png'):
            if sha256(evaluation / filename) != case['evaluation_hashes'][filename]:
                raise ValueError('Evaluation truth identity mismatch')
        truth = np.array(Image.open(evaluation / 'truth.png').convert('RGB'))
        hidden = np.array(Image.open(evaluation / 'hidden_core.png').convert('L')) > 0
        truth[~hidden] = 127
        first = torch.from_numpy(truth.copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
        mask = torch.from_numpy(hidden.copy()).cuda()
        values = []
        for seed in seeds:
            folder = run / 'candidates' / case['case_id']
            record = json.loads((folder / f'seed-{seed}.json').read_text())
            filename = f'seed-{seed}.png'
            value = dict(seed=seed, masked_spatial_lpips=None, missing_generation=True)
            if filename in record['artifacts']:
                if sha256(folder / filename) != record['artifacts'][filename]:
                    raise ValueError('Candidate RGB hash mismatch')
                candidate = np.array(Image.open(folder / filename).convert('RGB'))
                candidate[~hidden] = 127
                second = torch.from_numpy(candidate.copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
                started = time.perf_counter()
                with torch.inference_mode():
                    distance_map = model(first, second)[0, 0]
                if distance_map.shape != mask.shape:
                    raise ValueError('Spatial LPIPS frame mismatch')
                value.update(masked_spatial_lpips=float(distance_map[mask].mean().item()),
                             missing_generation=False, candidate_sha256=record['artifacts'][filename],
                             seconds=time.perf_counter() - started)
            values.append(value)
        write_json(destination, dict(case_id=case['case_id'], split=case['split'],
            truth_hashes=case['evaluation_hashes'], values=values))
        print('LPIPS', case['case_id'], [v['masked_spatial_lpips'] for v in values], flush=True)
    metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(output / f'runtime-worker-{args.worker_index}.json', metadata)


if __name__ == '__main__':
    main()
