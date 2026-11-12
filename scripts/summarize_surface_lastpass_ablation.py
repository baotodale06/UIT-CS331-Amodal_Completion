#!/usr/bin/env python3
"""Summarize every frozen last-pass ablation target without selecting a subset."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from audit_surface_progressive_result import estimate, METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    freeze_sha = sha256(args.freeze)
    for name, expected in freeze['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Frozen measurement source changed: '+name)
    paths = list(args.run_dir.glob('targets/*/*/ablation.json'))
    rows = {row['id']:row for row in (json.loads(path.read_text()) for path in paths)}
    if len(rows) != len(paths) or set(rows) != set(freeze['target_ids']):
        raise ValueError('Require exactly the complete frozen target inventory')
    rows = [rows[identity] for identity in freeze['target_ids']]
    runtimes = [json.loads((args.run_dir/f'runtime-worker-{i}.json').read_text()) for i in range(4)]
    for runtime in runtimes:
        if not runtime.get('finished_at') or runtime['freeze_sha256'] != freeze_sha:
            raise ValueError('All workers must finish with the same freeze')
        if runtime['environment']['packages'] != runtimes[0]['environment']['packages']:
            raise ValueError('Metric worker package inventories differ')
    drift = {name:{metric:0. for metric in METRICS} for name in ('baseline','surface')}
    images = 0
    for row in rows:
        if row['inputs']['freeze_sha256'] != freeze_sha:
            raise ValueError('A target was scored with another freeze')
        if row['surface_status']=='success':
            path=args.run_dir/'targets'/row['id']/'raw-final.png'
            if sha256(path) != row['derivation']['derived_sha256']:
                raise ValueError('Derived RGB changed: '+row['id'])
            images += 1
        elif row['derivation'] is not None or row['scores']['raw_final'] != row['scores']['surface']:
            raise ValueError('Ablation changed failure treatment')
        for name in drift:
            for metric in METRICS:
                value=row['scores'][name][metric]
                prior=row['prior_cached_scores'][name][metric]
                if (value is None) != (prior is None):
                    raise ValueError('Original metric availability changed')
                if value is not None:
                    drift[name][metric]=max(drift[name][metric],abs(value-prior))
    contrasts = {}
    for baseline, method in (('baseline','surface'), ('baseline','raw_final'), ('surface','raw_final')):
        items = [dict(id=row['id'],source=row['source'],image_identity=row['image_identity'],
            baseline=row['scores'][baseline],surface=row['scores'][method]) for row in rows]
        contrasts[method+'_vs_'+baseline]={metric:estimate(items,metric) for metric in METRICS}
    analysis = args.run_dir/'analysis'
    analysis.mkdir(exist_ok=True)
    for source in (Path(__file__), ROOT/'scripts/audit_surface_progressive_result.py',
                   ROOT/'scripts/measure_surface_lastpass_ablation.py'):
        shutil.copyfile(source, analysis/source.name)
    shutil.copyfile(args.freeze, analysis/'freeze.json')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),targets=len(rows),
        derived_images=images,freeze_sha256=freeze_sha,contrasts=contrasts,
        original_score_max_absolute_drift=drift,
        metric_replay_within_1e_5=all(value<=1e-5 for side in drift.values() for value in side.values()),
        raw_final_all_four_full_points_at_least3=all(row['relative_gain_percent']>=3
            for row in contrasts['raw_final_vs_baseline'].values()),
        scope=freeze['scope'], unchanged_masks_geometry_status=True,
        source_sha256={source.name:sha256(source) for source in analysis.glob('*.py')})
    write_json(analysis/'summary.json',report)
    lines=['# Frozen last-pass raw RGB ablation','',freeze['scope'],'',
        f'All {len(rows)} targets retained; {images} raw RGB derivations. Masks, geometry, failures and preceding trajectories are fixed.','',
        'No diffusion rerun or parameter search was performed. Baseline and corrected outputs were scored again alongside the raw RGB ablation.','']
    for name,metrics in contrasts.items():
        lines += ['## '+name,'','| Metric | Control | Method | Gain | 95% interval | n |',
                  '| --- | ---: | ---: | ---: | --- | ---: |']
        for metric,value in metrics.items():
            low,high=value['relative_ci_percent']
            lines.append(f'| {metric} | {value["baseline"]:.6f} | {value["method"]:.6f} | '
                f'{value["relative_gain_percent"]:+.3f}% | [{low:+.3f}%, {high:+.3f}%] | {value["n"]} |')
        lines.append('')
    lines += ['CLIP contrasts retain only shared scored targets; other metrics retain all failure penalties. '
              'Intervals use 5,000 source-stratified image-cluster bootstrap samples. '
              'There is no hidden-RGB MAE in this naturally occluded benchmark.','',
              'This exploratory diagnostic cannot establish the behavior of an end-to-end raw progressive run. '
              'Earlier corrected selections affected its trajectories and final masks.','',
              'Original-score maximum absolute replay drift:','',json.dumps(drift,indent=2),'']
    (analysis/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
