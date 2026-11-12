#!/usr/bin/env python3
"""Freeze a learned policy using development files, never new test outcomes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-dir', type=Path, required=True)
    p.add_argument('--corrections', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--refinement-name', help='Explicit development-supported correction; default best CV-selected MAE')
    p.add_argument('--provisional', action='store_true', help='Development GPU review only; cannot authorize fresh test')
    p.add_argument('--multimetric', action='store_true', help='Use the balanced paper-metric development search')
    p.add_argument('--candidate-seeds', type=int, nargs='+', default=[11,22,33,44])
    p.add_argument('--scope', choices=('controlled', 'full_cached_postprocessing'), default='controlled')
    p.add_argument('--ensemble-config', type=Path, help='Development-selected consensus correction JSON')
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refuse to overwrite an immutable freeze')
    search_path = a.development_dir / ('model_search.json' if a.multimetric else 'ranker_cv.json')
    cv = json.loads(search_path.read_text())
    best = cv['candidates'][0] if a.multimetric else min(cv, key=lambda r: r['objective'])
    choices = best['predictions'] if a.multimetric else {r['case_id']: r['index'] for r in best['predictions']}
    correction = json.loads(a.corrections.read_text())
    means = ({name: float(np.mean([r['variants'][name][choices[r['case_id']]] for r in correction['cases']]))
              for name in correction['configs']} if max(choices.values()) < 4 else {})
    if not means and not a.refinement_name:
        raise ValueError('Expanded pools require an explicitly development-supported calibration')
    name = a.refinement_name or min(means, key=means.get)
    ranker = json.loads((a.development_dir / 'ranker.json').read_text())
    previous = []
    for experiment in ('surface-consistency-20261108', 'surface-consistency-replication87-20261108'):
        previous += json.loads((ROOT / 'output' / experiment / 'cases.json').read_text())
    freeze = dict(schema=1, created_at=datetime.now(timezone.utc).isoformat(),
        status='provisional_development_review' if a.provisional else 'frozen_before_fresh_test',
        method='Surface-PMC with observation-anchored ranker, sampling agreement and residual calibration',
        primary_policy='trained_surface_refined', primary_control='uniform_raw', secondary_control='seed_11_raw',
        endpoints=['hidden_core_mae', 'hidden_spatial_lpips', 'object_lpips', 'dreamsim', 'clip_text_cosine'],
        desired_relative_improvement_percent=[3, 5],
        inference_truth_access=False, candidate_seeds=a.candidate_seeds, evaluation_scope=a.scope,
        ranker=ranker, refinement=correction['configs'][name], refinement_name=name,
        pretrained_visible_features='visible_lpips' in ranker['feature_names'],
        scales=json.loads((ROOT / 'output/surface-consistency-20261108/analysis/development_freeze.json').read_text())['scales'],
        development=dict(images=len(previous), image_identities=[r['image_identity'] for r in previous],
            image_sha256=[r['source_sha256'] for r in previous], cross_validation='leave one source out',
            selected_cv={k:v for k,v in best.items() if k != 'predictions'},
            correction_cv_selected_mae=means,
            note='All earlier test cases are development here; CV is model selection, not new held-out evidence.'),
        source_hashes={path.relative_to(ROOT).as_posix(): sha256(path) for path in (
            ROOT / 'amodal_benchmark/surface_refinement.py', ROOT / 'scripts/develop_surface_followup.py',
            ROOT / 'scripts/extract_surface_pool_features.py', Path(__file__),
            ROOT / 'scripts/measure_surface_followup.py', ROOT / 'scripts/refine_open_world_surface.py',
            ROOT / 'scripts/develop_surface_multimetric.py', ROOT / 'amodal_benchmark/surface_features.py',
            ROOT / 'scripts/extract_surface_deep_features.py', ROOT / 'amodal_benchmark/surface.py',
            ROOT / 'amodal_benchmark/metrics.py', ROOT / 'amodal_benchmark/surface_ensemble.py',
            ROOT / 'amodal_benchmark/surface_policy.py', ROOT / 'scripts/select_surface_pool.py',
            ROOT / 'amodal_benchmark/surface_runtime.py', ROOT / 'evaluate_open_world.py',
            ROOT / 'scripts/run_surface_progressive_slurm.sh')},
        input_hashes={str(path): sha256(path) for path in (a.development_dir / 'ranker.json',
            search_path, a.corrections)})
    if a.ensemble_config:
        freeze['ensemble'] = json.loads(a.ensemble_config.read_text())
        freeze['input_hashes'][str(a.ensemble_config)] = sha256(a.ensemble_config)
    if a.scope == 'full_cached_postprocessing':
        freeze.update(primary_policy='surface_calibration', primary_control='accepted_cached_baseline',
            endpoints=['LPIPS', 'Feature Similarity', 'SSIM', 'CLIP'],
            note='Only the frozen refinement applies; no candidate ranker is used on this cached full run.')
    write_json(a.output, freeze)
    snapshot = a.output.parent / (a.output.stem + '-source')
    for relative in freeze['source_hashes']:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    print(json.dumps(dict(refinement=name, development_mae=means, status=freeze['status']), indent=2))


if __name__ == '__main__':
    main()
