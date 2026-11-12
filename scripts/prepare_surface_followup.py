#!/usr/bin/env python3
"""Inventory fresh eligible images and prepare an immutable follow-up cohort."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/open_world_amodal'))
    p.add_argument('--per-source', type=int, default=64)
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args()
    previous = []
    for name in ('surface-consistency-20261108', 'surface-consistency-replication87-20261108'):
        previous += json.loads((ROOT / 'output' / name / 'cases.json').read_text())
    seen = {x['image_identity'] for x in previous}
    hashes = {x['source_sha256'] for x in previous}
    counts = Counter()
    rows, _ = read_official(a.dataset_root)
    for row in rows:
        identity = row['source'] + '/' + row['original_id']
        if identity in seen:
            continue
        path = a.reference_run / 'visible_references/masks' / row['reference']
        if not path.exists():
            continue
        mask = np.asarray(Image.open(path).convert('L')) > 0
        if mask.sum() < 4096:
            continue
        rr, cc = np.where(mask)
        if min(np.ptp(rr) + 1, np.ptp(cc) + 1) < 80:
            continue
        digest = sha256(a.dataset_root / row['image'])
        if digest in hashes:
            continue
        seen.add(identity)
        hashes.add(digest)
        counts[row['source']] += 1
    print(json.dumps(dict(eligible=counts, previous=len(previous))), flush=True)
    if not a.prepare:
        return
    control = a.output_dir.parent / (a.output_dir.name + '-design')
    control.mkdir(parents=True, exist_ok=False)
    write_json(control / 'excluded_cases.json', previous)
    config = json.loads((ROOT / 'configs/research/surface-consistency-v1.json').read_text())
    config.update(experiment=a.output_dir.name, selection_seed=20261110,
                  development_per_source=0, cases_per_source=a.per_source,
                  source_case_limits={s: min(a.per_source, counts[s]) for s in
                                      ('VG', 'COCOA', 'free_images', 'laion')},
                  normalization='Frozen using previous 183 cases, all now development; new test untouched')
    write_json(control / 'config.json', config)
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_surface_experiment.py'),
                    '--dataset-root', str(a.dataset_root), '--reference-run', str(a.reference_run),
                    '--output-dir', str(a.output_dir), '--config', str(control / 'config.json'),
                    '--exclude-cases', str(control / 'excluded_cases.json')], check=True)


if __name__ == '__main__':
    main()
