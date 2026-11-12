#!/usr/bin/env python3
"""Audit split image identity using decoded pixels, without evaluating completions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def fingerprint(row, dataset):
    path = dataset / row['source_image']
    if sha256(path) != row['source_sha256']:
        raise ValueError('Source image changed')
    with Image.open(path) as original:
        image = original.convert('RGB')
        decoded = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        thumbnail = np.array(image.resize((64, 64), Image.Resampling.BOX)).astype(np.float32) / 255
        return dict(case_id=row['case_id'], identity=row['image_identity'], decoded_sha256=decoded,
                    aspect=image.width / image.height, thumbnail=thumbnail)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--fresh-experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = [ROOT / 'output' / name / 'cases.json' for name in
             ('surface-consistency-20261108', 'surface-consistency-replication87-20261108')]
    old = [fingerprint(row, a.dataset_root) for path in paths for row in json.loads(path.read_text())]
    fresh_path = a.fresh_experiment / 'cases.json'
    fresh = [fingerprint(row, a.dataset_root) for row in json.loads(fresh_path.read_text())]
    thumbnails = np.stack([row['thumbnail'] for row in old])
    aspects = np.array([row['aspect'] for row in old])
    nearest, exact, near, within = [], [], [], []
    for row in fresh:
        errors = np.abs(thumbnails - row['thumbnail']).mean(axis=(1,2,3))
        comparable = np.abs(aspects / row['aspect'] - 1) <= .01
        errors[~comparable] = np.inf
        index = int(np.argmin(errors))
        item = dict(fresh=row['identity'], nearest_development=old[index]['identity'],
                    aligned_64px_rgb_mae=float(errors[index]) if np.isfinite(errors[index]) else None)
        nearest.append(item)
        exact += [dict(fresh=row['identity'], development=other['identity']) for other in old
                  if row['decoded_sha256'] == other['decoded_sha256']]
        near += [dict(fresh=row['identity'], development=old[i]['identity'], rgb_mae=float(errors[i]))
                 for i in np.flatnonzero(errors <= .005)]
    for i, row in enumerate(fresh):
        within += [dict(first=other['identity'], second=row['identity']) for other in fresh[:i]
                   if row['decoded_sha256'] == other['decoded_sha256']]
    write_json(a.output, dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='split identity audit only; no generated-image or hidden-error evaluation',
        exact_decoded_cross_split_duplicates=exact, near_aligned_cross_split_flags=near,
        exact_decoded_within_fresh_duplicates=within,
        near_rule='aspect within 1%; 64x64 BOX-resized RGB MAE <=0.005 on [0,1]',
        limitation='Does not certify absence of transformed, cropped, or semantically similar scenes',
        development_images=len(old), fresh_images=len(fresh), nearest=nearest,
        input_sha256={str(path):sha256(path) for path in [*paths, fresh_path]},
        source_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(development=len(old),fresh=len(fresh),exact_cross=len(exact),
        near_cross=len(near),exact_within_fresh=len(within),
        minimum_thumbnail_mae=min(item['aligned_64px_rgb_mae'] for item in nearest
                                  if item['aligned_64px_rgb_mae'] is not None))))


if __name__ == '__main__':
    main()
