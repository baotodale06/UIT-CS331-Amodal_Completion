#!/usr/bin/env python3
"""Evaluate a small surface-consensus correction grid on development images only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_refinement import refine_surface, RefinementConfig
from amodal_benchmark.surface_ensemble import refine_surface_pool, EnsembleConfig
from develop_surface_followup import load_development


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new development output file')
    configs = [EnsembleConfig(strength=0)]
    configs += [EnsembleConfig(strength=strength,sigma=sigma)
                for strength in (.5,1.) for sigma in (4.,12.,32.)]
    results = []
    for row in load_development():
        experiment, case_id = Path(row['experiment']), row['case_id']
        inputs, evaluation = experiment / 'inputs' / case_id, experiment / 'evaluation_only' / case_id
        observed = np.array(Image.open(inputs / 'observed.png').convert('RGB'))
        visible = np.array(Image.open(inputs / 'visible.png').convert('L')) > 0
        truth = np.array(Image.open(evaluation / 'truth.png').convert('RGB'))
        hidden = np.array(Image.open(evaluation / 'hidden_core.png').convert('L')) > 0
        folder = experiment / 'full/candidates' / case_id
        masks = [np.array(Image.open(folder / f'seed-{seed}-mask.png').convert('L')) > 0 for seed in (11,22,33,44)]
        candidates = [np.array(Image.open(folder / f'seed-{seed}.png').convert('RGB')) for seed in (11,22,33,44)]
        calibrated = [refine_surface(observed, visible, candidate, mask,
                      RefinementConfig(mode='affine',strength=1.))[0] for candidate, mask in zip(candidates,masks)]
        variants = []
        for config in configs:
            ensemble, _ = refine_surface_pool(calibrated,masks,visible,config)
            variants.append([float(np.abs(image[hidden].astype(float)-truth[hidden]).mean()/255) for image in ensemble])
        results.append(dict(case_id=case_id,source=row['source'],variants=variants))
        print('ENSEMBLE',case_id,flush=True)
    write_json(a.output,dict(configs=[asdict(c) for c in configs],cases=results,
                            source_sha256=sha256(ROOT / 'amodal_benchmark/surface_ensemble.py')))
    for i,config in enumerate(configs):
        print('MEAN',asdict(config),np.mean([r['variants'][i] for r in results]),flush=True)


if __name__ == '__main__':
    main()
