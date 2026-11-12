#!/usr/bin/env python3
"""Replay frozen fresh-test selections, corrections, MAE and artifact identities."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import score_surface
from amodal_benchmark.surface_policy import apply_frozen_surface
from amodal_benchmark.surface_refinement import RefinementConfig, refine_surface
from amodal_benchmark.surface_ensemble import EnsembleConfig, refine_surface_pool


def read_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def read_mask(path):
    return np.array(Image.open(path).convert('L')) > 0


def verify_case(arguments):
    case, paths, freeze = arguments
    experiment, run, extra, metrics, inference, features = map(Path, paths)
    case_id = case['case_id']
    inputs = experiment / 'inputs' / case_id
    for name, expected in case['input_hashes'].items():
        if sha256(inputs / name) != expected:
            raise ValueError('Observation checksum mismatch: ' + case_id)
    observed, visible, occlusion = read_rgb(inputs / 'observed.png'), read_mask(inputs / 'visible.png'), read_mask(inputs / 'occlusion.png')
    candidates, masks, scores, record_hashes = [], [], [], {}
    for seed in freeze['candidate_seeds']:
        folder = run / 'candidates' / case_id
        if not (folder / f'seed-{seed}.json').exists():
            folder = extra / 'candidates' / case_id
        record_path = folder / f'seed-{seed}.json'
        record = json.loads(record_path.read_text())
        record_hashes[record_path.name] = sha256(record_path)
        if record['input_hashes'] != case['input_hashes'] or record['seed'] != seed:
            raise ValueError('Candidate identity mismatch: ' + case_id)
        for name, expected in record['artifacts'].items():
            if sha256(folder / name) != expected:
                raise ValueError('Candidate checksum mismatch: ' + case_id)
        if f'seed-{seed}.png' not in record['artifacts']:
            raise ValueError('Missing generation requires explicit penalty replay')
        rgb, mask = read_rgb(folder / f'seed-{seed}.png'), read_mask(folder / f'seed-{seed}-mask.png')
        candidates.append(rgb)
        masks.append(mask)
        recomputed = score_surface(observed, visible, occlusion, rgb, mask)
        for key, value in record.get('surface', {}).items():
            actual = recomputed[key]
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not np.isclose(actual, value, atol=1e-10, rtol=0):
                    raise ValueError('Surface score changed: ' + case_id + '/' + key)
            elif actual != value:
                raise ValueError('Surface score changed: ' + case_id + '/' + key)
        scores.append(recomputed)
    cached = json.loads((features / (case_id + '.json')).read_text()) if freeze.get('pretrained_visible_features') else None
    if cached and (cached['candidate_record_sha256'] != record_hashes or cached['contract']['truth_access'] is not False):
        raise ValueError('Observable feature identity mismatch: ' + case_id)
    selected, mask, detail = apply_frozen_surface(observed, visible, occlusion, candidates, masks, freeze,
                                                 scores, cached['values'] if cached else None)
    independent = json.loads((inference / 'decisions' / (case_id + '.json')).read_text())
    measured = json.loads((metrics / (case_id + '.json')).read_text())
    if measured['selection']['trained_surface'] != detail['index'] or independent['index'] != detail['index']:
        raise ValueError('Independent selection disagrees: ' + case_id)
    for folder in (inference / 'selected' / case_id, metrics / 'selected' / case_id):
        if not np.array_equal(selected, read_rgb(folder / 'surface.png')) or not np.array_equal(mask, read_mask(folder / 'mask.png')):
            raise ValueError('Selected image differs from replay: ' + case_id)
    for name, expected in measured['selected_artifacts'].items():
        if sha256(metrics / 'selected' / case_id / name) != expected:
            raise ValueError('Metric selected artifact changed: ' + case_id)
    calibrated = [refine_surface(observed, visible, rgb, mask, RefinementConfig(**freeze['refinement']))[0]
                  for rgb, mask in zip(candidates, masks)]
    refined = refine_surface_pool(calibrated, masks, visible, EnsembleConfig(**freeze['ensemble']))[0] if freeze.get('ensemble') else calibrated
    # The observable selection above is complete before accessing evaluation RGB.
    evaluation = experiment / 'evaluation_only' / case_id
    for name, expected in case['evaluation_hashes'].items():
        if sha256(evaluation / name) != expected:
            raise ValueError('Evaluation checksum mismatch: ' + case_id)
    truth, hidden = read_rgb(evaluation / 'truth.png'), read_mask(evaluation / 'hidden_core.png')
    for kind, images in (('raw', candidates), ('calibrated', calibrated), ('refined', refined)):
        for rgb, recorded in zip(images, measured[kind]):
            mae = float(np.abs(rgb[hidden].astype(float) - truth[hidden]).mean() / 255)
            if abs(mae - recorded['hidden_core_mae']) > 1e-12:
                raise ValueError('Hidden MAE replay mismatch: ' + case_id + '/' + kind)
    return dict(case_id=case_id, candidates=len(candidates), selected=detail['index'],
                mae_checks=3 * len(candidates), exact_selected_rgb=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment-dir', 'run-dir', 'extra-run-dir', 'metrics-dir', 'inference-dir', 'features-dir', 'freeze', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if freeze['status'] != 'frozen_before_fresh_test':
        raise ValueError('Final freeze required')
    for name, expected in freeze['source_hashes'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    cases = json.loads((args.experiment_dir / 'cases.json').read_text())
    if (set(case['image_identity'] for case in cases) & set(freeze['development']['image_identities']) or
            set(case['source_sha256'] for case in cases) & set(freeze['development']['image_sha256'])):
        raise ValueError('Development and test overlap')
    paths = [str(getattr(args, name).resolve()) for name in ('experiment_dir', 'run_dir', 'extra_run_dir', 'metrics_dir', 'inference_dir', 'features_dir')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(verify_case, [(case, paths, freeze) for case in cases]))
    write_json(args.output, dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha256(args.freeze), cases=results, images=len(results),
        candidates=sum(row['candidates'] for row in results), mae_checks=sum(row['mae_checks'] for row in results),
        limitation='CPU replay checks saved neural-metric inputs and identities; it does not recompute learned metrics.',
        source_sha256=sha256(Path(__file__))))
    print('PASS', len(results), 'fresh images; exact observation-only choices, corrections and hidden MAE')


if __name__ == '__main__':
    main()
