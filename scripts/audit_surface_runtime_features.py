#!/usr/bin/env python3
"""Compare deployable observation features with the measured development provider."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_features import ObservedSurfaceFeatures
from amodal_benchmark.surface_runtime import SurfaceRuntimeFeatures
from measure_surface_followup import FollowupMetrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--dreamsim-cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=4)
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cases = json.loads((a.experiment_dir / 'generation_manifest.json').read_text())[:4]
    expected = ObservedSurfaceFeatures(FollowupMetrics(a.dreamsim_cache))
    runtime = SurfaceRuntimeFeatures(a.dreamsim_cache)
    results = []
    for case in cases[a.worker_index::a.worker_count]:
        inputs = a.experiment_dir / 'inputs' / case['case_id']
        observed = np.array(Image.open(inputs / 'observed.png').convert('RGB'))
        visible = np.array(Image.open(inputs / 'visible.png').convert('L')) > 0
        folder = a.experiment_dir / 'full/candidates' / case['case_id']
        images = [np.array(Image.open(folder / f'seed-{seed}.png').convert('RGB')) for seed in (11,22)]
        masks = [np.array(Image.open(folder / f'seed-{seed}-mask.png').convert('L')) > 0 for seed in (11,22)]
        first = expected(observed, visible, images, masks, case['label'])
        second = runtime.extract(observed, visible, images, masks, case['label'])
        maximum = max(abs(x[key] - y[key]) for x,y in zip(first,second) for key in x)
        if maximum > 1e-6:
            raise ValueError(f'Inference features differ from development: {maximum}')
        results.append(dict(case_id=case['case_id'], max_absolute_difference=maximum))
    write_json(a.output_dir / f'worker-{a.worker_index}.json', dict(passed=True,
        runtime_sha256=sha256(ROOT / 'amodal_benchmark/surface_runtime.py'), cases=results,
        observation_manifest_sha256=sha256(a.experiment_dir / 'generation_manifest.json'),
        truth_access=False, gpu=torch.cuda.get_device_name(0), torch=torch.__version__))


if __name__ == '__main__':
    main()
