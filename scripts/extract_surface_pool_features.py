#!/usr/bin/env python3
"""Extract observation-only sampling-agreement features on old development pools."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_refinement import pool_surface_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = {}
    for name in ('surface-consistency-20261108', 'surface-consistency-replication87-20261108'):
        experiment = ROOT / 'output' / name
        cases = json.loads((experiment / 'generation_manifest.json').read_text())
        for case in cases:
            inputs = experiment / 'inputs' / case['case_id']
            folder = experiment / 'full/candidates' / case['case_id']
            observed = np.array(Image.open(inputs / 'observed.png').convert('RGB'))
            visible = np.array(Image.open(inputs / 'visible.png').convert('L')) > 0
            occlusion = np.array(Image.open(inputs / 'occlusion.png').convert('L')) > 0
            candidates = [np.array(Image.open(folder / f'seed-{seed}.png').convert('RGB')) for seed in (11, 22, 33, 44)]
            masks = [np.array(Image.open(folder / f'seed-{seed}-mask.png').convert('L')) > 0 for seed in (11, 22, 33, 44)]
            output[case['case_id']] = pool_surface_features(observed, visible, occlusion, candidates, masks)
            print('FEATURES', case['case_id'], flush=True)
    write_json(a.output, dict(source_sha256=sha256(ROOT / 'amodal_benchmark/surface_refinement.py'),
                             cases=output, truth_used=False))


if __name__ == '__main__':
    main()
