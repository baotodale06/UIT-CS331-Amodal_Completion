#!/usr/bin/env python3
"""GPU checks that pretrained selection features cannot see hidden observation RGB."""
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
    import torch.nn.functional as functional
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    case = json.loads((a.experiment_dir / 'generation_manifest.json').read_text())[a.worker_index]
    folder = a.experiment_dir / 'inputs' / case['case_id']
    candidates = a.experiment_dir / 'full/candidates' / case['case_id']
    observed = np.array(Image.open(folder / 'observed.png').convert('RGB'))
    visible = np.array(Image.open(folder / 'visible.png').convert('L')) > 0
    image = np.array(Image.open(candidates / 'seed-11.png').convert('RGB'))
    mask = np.array(Image.open(candidates / 'seed-11-mask.png').convert('L')) > 0
    models = FollowupMetrics(a.dreamsim_cache)
    feature = ObservedSurfaceFeatures(models)
    first = feature(observed, visible, [image], [mask], case['label'])
    poison = observed.copy()
    poison[~visible] = np.random.default_rng(20261110).integers(0,256,(int((~visible).sum()),3),dtype=np.uint8)
    second = feature(poison, visible, [image], [mask], case['label'])
    if first != second:
        raise ValueError('Hidden observation values affected an inference feature')
    identity = feature(observed, visible, [observed], [visible], case['label'])[0]
    if abs(identity['visible_lpips']) > 1e-6 or abs(identity['visible_deep_distance']) > 1e-6:
        raise ValueError('Identical visible surfaces must have zero perceptual distance')
    with torch.inference_mode():
        x, y = models.dream_tensor(observed), models.dream_tensor(image)
        direct = float(models.dreamsim(x, y))
        embedded = float(1 - functional.cosine_similarity(models.dreamsim.embed(x), models.dreamsim.embed(y)))
    if abs(direct - embedded) > 1e-6:
        raise ValueError('Cached-embedding DreamSim differs from official forward')
    write_json(a.output_dir / f'worker-{a.worker_index}.json', dict(
        case_id=case['case_id'], hidden_observation_invariance=True, visible_identity=identity,
        dreamsim_forward=direct, dreamsim_embedding=embedded,
        source_sha256=sha256(Path(__file__)), features_sha256=sha256(ROOT / 'amodal_benchmark/surface_features.py')))
    print('PASS', case['case_id'], flush=True)


if __name__ == '__main__':
    main()
