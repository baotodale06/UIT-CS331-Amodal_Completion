"""Compare the admitted inference adapter pixelwise with every native validation mask.

Run on an allocated Quadro before a new benchmark smoke. This is a runtime
equivalence check, not another checkpoint selection or a benchmark score.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_constrained_runtime import load_prior, original_prior, shape_constrained_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--sam-validation', type=Path, required=True)
    parser.add_argument('--native-validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior adapter validation evidence')
    policy = shape_constrained_contract(args.checkpoint)
    training = policy['training_receipt']
    manifest_path = args.data / 'manifest.json'
    result_path = args.native_validation / 'result.json'
    if (sha256(manifest_path) != training['contract']['data_sha256']
            or sha256(result_path) != policy['evidence_files']['native-result.json']):
        raise ValueError('Runtime validation inputs changed')
    manifest, result = (json.loads(path.read_bytes()) for path in (manifest_path, result_path))
    selected = {row['id']: row for row in manifest['rows'] if row['split'] == 'validation'}
    if {row['id'] for row in result['rows']} != set(selected):
        raise ValueError('Runtime validation inventory changed')
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Use one allocated Quadro for native adapter validation')
    model = load_prior(args.checkpoint)
    groups = defaultdict(list)
    for row in result['rows']:
        groups[row['image_id']].append(row)
    records = []
    for image_id, objects in sorted(groups.items()):
        image_name = selected[objects[0]['id']]['image']
        image_path = (args.data / image_name).resolve()
        if not image_path.is_relative_to(args.data.resolve()) or sha256(image_path) != manifest['artifacts'][image_name]:
            raise ValueError('Runtime validation image changed')
        with Image.open(image_path) as image:
            rgb = np.array(image.convert('RGB'))
        for row in objects:
            if row['image_id'] != selected[row['id']]['image_id'] or row['image'] != image_name:
                raise ValueError('Runtime validation image identity changed')
            support_path = args.sam_validation / 'objects' / row['id'] / 'sam-visible.png'
            prediction_path = args.native_validation / 'objects' / row['id'] / 'prediction.png'
            if sha256(support_path) != row['sam_sha256'] or sha256(prediction_path) != row['prediction_sha256']:
                raise ValueError('Native prediction or original SAM support changed')
            with Image.open(support_path) as image:
                support = np.array(image) > 0
            with Image.open(prediction_path) as image:
                expected = np.array(image) > 0
            if support.shape != expected.shape or rgb.shape[:2] != support.shape:
                raise ValueError('Native frames differ')
            if support.any():
                probability, box, _, _ = original_prior(model, rgb, support)
                actual = probability >= .5
            else:
                box, actual = None, np.zeros_like(support)
            differences = int(np.count_nonzero(actual != expected))
            if differences or (box is not None and list(box) != row['box']):
                raise ValueError(f"Adapter/native mismatch for {row['id']}: {differences} pixels")
            records.append(dict(id=row['id'], image_id=image_id, different_pixels=differences,
                                expected_prediction_sha256=row['prediction_sha256']))
    write_json(args.output, dict(verified=True, at=datetime.now(timezone.utc).isoformat(),
        objects=len(records), images=len(groups), different_pixels=0, rows=records,
        shape_policy=policy, native_result_sha256=sha256(result_path),
        validation_source_sha256=sha256(Path(__file__)), gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__), cuda=torch.version.cuda, benchmark_success=False))
    print(f'Exact adapter/native replay: {len(records)} objects / {len(groups)} images')


if __name__ == '__main__':
    main()
