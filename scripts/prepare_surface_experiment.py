#!/usr/bin/env python3
"""Freeze image-disjoint held-out-pixel cases without inspecting completions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal_benchmark.dataset import read_official, sha256, write_json


def stable_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_case(image, mask, direction, fraction, resolution):
    """Resize clean source first, then remove *all* RGB inside the occluder."""
    rows, cols = np.where(mask)
    side = int(np.ceil(max(np.ptp(rows) + 1, np.ptp(cols) + 1) * 1.35))
    cx, cy = (int(cols.min()) + int(cols.max())) // 2, (int(rows.min()) + int(rows.max())) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    # Explicit neutral padding, including the mask, has no target evidence.
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    clipped = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
    canvas.paste(image.crop(clipped), (clipped[0] - box[0], clipped[1] - box[1]))
    truth = np.array(canvas.resize((resolution, resolution), Image.Resampling.LANCZOS))
    full = np.array(Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize(
        (resolution, resolution), Image.Resampling.NEAREST)) > 0
    yy, xx = np.indices(full.shape)
    coordinate = (xx, yy, resolution - 1 - xx, resolution - 1 - yy)[direction]
    threshold = int(np.quantile(coordinate[full], 1 - fraction))
    occlusion = coordinate >= threshold
    visible = full & ~occlusion
    hidden = full & occlusion
    core = np.array(Image.fromarray(full.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    hidden_core = core & occlusion
    observed = truth.copy()
    observed[occlusion] = 127
    return dict(truth=truth, observed=observed, visible=visible, occlusion=occlusion,
                hidden=hidden, hidden_core=hidden_core, full=full, crop_box=list(box),
                direction=direction, threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/research/surface-consistency-v1.json"))
    parser.add_argument("--exclude-cases", type=Path, help="Image-disjoint replication: exclude these existing cases")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use a new experiment directory; preparation is immutable")
    config = json.loads(args.config.read_text())
    source_limits = {source: config.get('source_case_limits', {}).get(source, config['cases_per_source'])
                     for source in ('VG', 'COCOA', 'free_images', 'laion')}
    all_rows, _ = read_official(args.dataset_root)
    ordered = sorted(all_rows, key=lambda r: stable_key(f"{config['selection_seed']}:{r['id']}"))
    counts, excluded, seen, cases = Counter(), Counter(), set(), []
    excluded_hashes = set()
    if args.exclude_cases:
        previous = json.loads(args.exclude_cases.read_text())
        seen.update(row['image_identity'] for row in previous)
        excluded_hashes.update(row['source_sha256'] for row in previous)
    for row in ordered:
        if counts[row['source']] >= source_limits[row['source']]:
            continue
        identity = row['source'] + '/' + row['original_id']
        if identity in seen:
            excluded['duplicate_image'] += 1
            continue
        path = args.reference_run / 'visible_references/masks' / row['reference']
        if not path.exists():
            excluded['missing_reference'] += 1
            continue
        mask = np.array(Image.open(path).convert('L')) > 0
        if mask.sum() < config['min_native_mask_pixels']:
            excluded['small_mask'] += 1
            continue
        rr, cc = np.where(mask)
        if min(np.ptp(rr) + 1, np.ptp(cc) + 1) < config['min_native_bbox_side']:
            excluded['small_bbox'] += 1
            continue
        image_path = args.dataset_root / row['image']
        image_hash = sha256(image_path)
        if image_hash in excluded_hashes:
            excluded['duplicate_image_bytes'] += 1
            continue
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            if mask.shape != (image.height, image.width):
                raise ValueError('Reference frame mismatch: ' + row['id'])
            direction = int(stable_key('direction:' + row['id'])[:8], 16) % 4
            arrays = make_case(image, mask, direction, config['hidden_fraction'], config['resolution'])
        if min(arrays['visible'].sum(), arrays['hidden_core'].sum()) < 1024:
            excluded['small_resized_support'] += 1
            continue
        seen.add(identity)
        excluded_hashes.add(image_hash)
        index = counts[row['source']]
        split = 'development' if index < config['development_per_source'] else 'test'
        case_id = f"{row['source']}-{index:03d}-{stable_key(row['id'])[:10]}"
        folder = args.output_dir / 'inputs' / case_id
        evaluation = args.output_dir / 'evaluation_only' / case_id
        folder.mkdir(parents=True)
        evaluation.mkdir(parents=True)
        for key in ('observed', 'visible', 'occlusion'):
            arr = arrays[key]
            Image.fromarray(arr if arr.ndim == 3 else arr.astype(np.uint8) * 255).save(folder / (key + '.png'))
        for key in ('truth', 'hidden', 'hidden_core', 'full'):
            arr = arrays[key]
            Image.fromarray(arr if arr.ndim == 3 else arr.astype(np.uint8) * 255).save(evaluation / (key + '.png'))
        case = dict(case_id=case_id, target_id=row['id'], source=row['source'], label=row['label'],
                    image_identity=identity, split=split, source_image=row['image'],
                    source_sha256=image_hash, reference_sha256=sha256(path),
                    crop_box=arrays['crop_box'], direction=direction, threshold=arrays['threshold'],
                    visible_pixels=int(arrays['visible'].sum()), hidden_pixels=int(arrays['hidden'].sum()),
                    hidden_core_pixels=int(arrays['hidden_core'].sum()),
                    actual_hidden_fraction=float(arrays['hidden'].sum() / arrays['full'].sum()),
                    input_hashes={p.name: sha256(p) for p in folder.iterdir()},
                    evaluation_hashes={p.name: sha256(p) for p in evaluation.iterdir()})
        cases.append(case)
        counts[row['source']] += 1
    if any(counts[s] != source_limits[s] for s in source_limits):
        write_json(args.output_dir / 'preparation_incomplete.json', dict(
            status='insufficient_eligible_sources', counts=counts, requested=source_limits,
            excluded=excluded, completed_cases=cases, no_generation_submitted=True))
        raise ValueError(f'Insufficient eligible sources: {counts}')
    write_json(args.output_dir / 'protocol.json', config)
    write_json(args.output_dir / 'cases.json', cases)
    # Generation manifest deliberately excludes GT support sizes and masks.
    write_json(args.output_dir / 'generation_manifest.json', [
        {k: row[k] for k in ('case_id', 'label', 'split', 'source', 'input_hashes')} for row in cases])
    write_json(args.output_dir / 'preparation.json', dict(counts=counts, excluded=excluded,
        config_sha256=sha256(args.config), cases_sha256=sha256(args.output_dir / 'cases.json'),
        preparation_source_sha256=sha256(Path(__file__)),
        excluded_cases_sha256=sha256(args.exclude_cases) if args.exclude_cases else None,
        reference_run=str(args.reference_run), total=len(cases)))
    print(json.dumps(dict(counts=counts, excluded=excluded, total=len(cases))))


if __name__ == '__main__':
    main()
