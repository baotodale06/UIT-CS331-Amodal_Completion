#!/usr/bin/env python3
"""Develop a surface ranker exclusively on the previously published 183 cases."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import select_candidate
from amodal_benchmark.surface_refinement import (FEATURE_NAMES, POOL_FEATURE_NAMES, RefinementConfig,
    rank_features, predict_rank, refine_surface)


def load_development():
    rows = []
    for name in ('surface-consistency-20261108', 'surface-consistency-replication87-20261108'):
        experiment = ROOT / 'output' / name
        cases = {r['case_id']: r for r in json.loads((experiment / 'cases.json').read_text())}
        for filename in ('development_cases.json', 'test_cases.json'):
            path = experiment / 'analysis' / filename
            if not path.exists():
                continue
            for row in json.loads(path.read_text()):
                row['experiment'] = str(experiment)
                row['case'] = cases[row['case_id']]
                perceptual = json.loads((experiment / 'perceptual' / (row['case_id'] + '.json')).read_text())
                row['lpips'] = [v['masked_spatial_lpips'] for v in perceptual['values']]
                rows.append(row)
    return rows


def fit(rows, regularization, lpips_weight, valid_only=True, pool_features=False):
    names = FEATURE_NAMES + POOL_FEATURE_NAMES if pool_features else FEATURE_NAMES
    arrays = [np.stack([rank_features(s, names) for s in r['scores']]) for r in rows]
    all_features = np.concatenate(arrays)
    medians = np.nanmedian(all_features, axis=0)
    medians = np.nan_to_num(medians)
    all_features = np.where(np.isfinite(all_features), all_features, medians)
    center, scale = np.mean(all_features, axis=0), np.std(all_features, axis=0)
    scale = np.maximum(scale, .001)
    x, y = [], []
    for row, features in zip(rows, arrays):
        features = (np.where(np.isfinite(features), features, medians) - center) / scale
        mae = np.array([m['hidden_core_mae'] for m in row['metrics']])
        lpips = np.asarray(row['lpips'])
        target = (1 - lpips_weight) * mae / .17 + lpips_weight * lpips / .4
        # Within-image centering learns ordering rather than image difficulty.
        x.append(features - features.mean(0))
        y.append(target - target.mean())
    x, y = np.concatenate(x), np.concatenate(y)
    coefficients = np.linalg.solve(x.T @ x + regularization * np.eye(x.shape[1]), x.T @ y)
    return dict(feature_names=list(names), medians=medians.tolist(), center=center.tolist(),
                scale=scale.tolist(), coefficients=coefficients.tolist(),
                regularization=regularization, lpips_weight=lpips_weight, valid_only=valid_only)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--corrections', action='store_true')
    p.add_argument('--pool-features', type=Path)
    a = p.parse_args()
    rows = load_development()
    if a.pool_features:
        features = json.loads(a.pool_features.read_text())['cases']
        for row in rows:
            for score, extra in zip(row['scores'], features[row['case_id']]):
                score.update(extra)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    search = []
    for regularization in (1., 10., 100., 1000.):
        for weight in (0., .25, .5, .75, 1.):
            predictions = []
            for source in ('VG', 'COCOA', 'free_images', 'laion'):
                model = fit([r for r in rows if r['source'] != source], regularization, weight,
                            pool_features=bool(a.pool_features))
                for row in rows:
                    if row['source'] == source:
                        i, _ = predict_rank(row['scores'], model)
                        predictions.append(dict(case_id=row['case_id'], source=source, index=i,
                            mae=row['metrics'][i]['hidden_core_mae'], lpips=row['lpips'][i],
                            control_mae=np.mean([m['hidden_core_mae'] for m in row['metrics']]),
                            control_lpips=np.mean(row['lpips'])))
            mae_ratio = np.mean([v['mae'] for v in predictions]) / np.mean([v['control_mae'] for v in predictions])
            lpips_ratio = np.mean([v['lpips'] for v in predictions]) / np.mean([v['control_lpips'] for v in predictions])
            search.append(dict(regularization=regularization, lpips_weight=weight,
                               mae_ratio=mae_ratio, lpips_ratio=lpips_ratio,
                               objective=(mae_ratio + lpips_ratio) / 2,
                               predictions=predictions))
    best = min(search, key=lambda r: r['objective'])
    model = fit(rows, best['regularization'], best['lpips_weight'], pool_features=bool(a.pool_features))
    write_json(a.output_dir / 'ranker_cv.json', search)
    write_json(a.output_dir / 'ranker.json', model)
    print('RANKER', json.dumps({k: v for k, v in best.items() if k != 'predictions'}), flush=True)
    if not a.corrections:
        return
    configs = [RefinementConfig(mode='none')]
    configs += [RefinementConfig(mode=mode, strength=strength) for mode in ('shift', 'affine', 'field')
                for strength in (.25, .5, 1.)]
    variants = {f'{c.mode}-{c.strength}': c for c in configs}
    corrections = []
    for row in rows:
        experiment, case = Path(row['experiment']), row['case']
        input_dir = experiment / 'inputs' / row['case_id']
        eval_dir = experiment / 'evaluation_only' / row['case_id']
        observed = np.array(Image.open(input_dir / 'observed.png').convert('RGB'))
        visible = np.array(Image.open(input_dir / 'visible.png')) > 0
        truth = np.array(Image.open(eval_dir / 'truth.png').convert('RGB'))
        hidden = np.array(Image.open(eval_dir / 'hidden_core.png')) > 0
        item = dict(case_id=row['case_id'], source=row['source'], variants={})
        folder = experiment / 'full/candidates' / row['case_id']
        candidates = [(np.array(Image.open(folder / f'seed-{seed}.png').convert('RGB')),
                       np.array(Image.open(folder / f'seed-{seed}-mask.png')) > 0) for seed in (11, 22, 33, 44)]
        for name, config in variants.items():
            values = []
            for candidate, mask in candidates:
                refined, details = refine_surface(observed, visible, candidate, mask, config)
                values.append(float(np.abs(refined[hidden].astype(float) - truth[hidden]).mean() / 255))
            item['variants'][name] = values
        corrections.append(item)
        print('CORRECTION', row['case_id'], flush=True)
    write_json(a.output_dir / 'correction_development.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        configs={k: asdict(v) for k, v in variants.items()}, cases=corrections,
        source_sha256=sha256(ROOT / 'amodal_benchmark/surface_refinement.py')))
    for name in variants:
        before = np.mean([r['variants']['none-0.5'] for r in corrections])
        after = np.mean([r['variants'][name] for r in corrections])
        print('SUMMARY', name, before, after, 1 - after / before, flush=True)


if __name__ == '__main__':
    main()
