#!/usr/bin/env python3
"""Audit recovered last-pass evidence against the previously verified full runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json

METRICS = ('CLIP', 'LPIPS', 'Feature Similarity', 'SSIM')
PENALTIES = {'CLIP': None, 'LPIPS': 1.0, 'Feature Similarity': 0.0, 'SSIM': 0.0}
FREEZE_SHA = '4866d3f2a1a26e207b45a101ef8e3c0316cc18af828188b1edf017229a7210c8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_target(row, official, records, cached, result_hashes, freeze_sha):
    """Check status, candidate lineage, original caches, and failure accounting."""
    identity = official['id']
    require(row['id'] == identity and row['source'] == official['source']
            and row['image_identity'] == official['source']+'/'+official['original_id'],
            'Target identity differs: '+identity)
    require(row['inputs']['freeze_sha256'] == freeze_sha, 'Target freeze differs')
    for side in ('baseline', 'surface'):
        require(row['inputs'][side+'_result_sha256'] == result_hashes[side],
                'Original result hash differs: '+identity)
        require(row[side+'_status'] == records[side]['status'], 'Original status differs')
        require(row['prior_cached_scores'][side] == cached[side], 'Original cache differs')
        values = row['scores'][side]
        require(set(values) == set(METRICS), 'Metric inventory differs')
        if records[side]['status'] != 'success':
            require(values == PENALTIES, 'Failure penalties changed')
        else:
            require(all(isinstance(value, (int, float)) and np.isfinite(value)
                        for value in values.values()), 'Successful score is missing or nonfinite')
    for key in ('input_sha256', 'seed', 'evaluation_reference_artifacts'):
        require(records['baseline'][key] == records['surface'][key], 'Paired inputs differ')
    if row['surface_status'] != 'success':
        require(row['derivation'] is None and row['scores']['raw_final'] == PENALTIES,
                'Raw derivation changed a failed target')
        return
    require(set(row['scores']['raw_final']) == set(METRICS)
            and all(isinstance(value, (int, float)) and np.isfinite(value)
                    for value in row['scores']['raw_final'].values()), 'Invalid raw scores')
    derived = row['derivation']
    trace = records['surface']['diagnostics']['surface_iterations'][-1]
    require(derived['decision_sha256'] == trace['decision_sha256']
            and derived['selected_index'] == trace['selected']
            and derived['iteration'] == trace['iteration'], 'Final candidate lineage differs')
    require(derived['corrected_completion_sha256'] == records['surface']['completion_sha256'],
            'Corrected completion linkage differs')
    require(derived['unchanged_alpha'] is True and derived['unchanged_geometry'] is True,
            'Derived mask or geometry was changed')
    for key in ('raw_candidate_sha256', 'fixed_mask_sha256', 'derived_sha256'):
        value = derived[key]
        require(len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Invalid derived artifact hash')
    require(len(derived['size']) == 2 and min(derived['size']) > 0, 'Invalid output size')


def paired_estimate(rows, control, method, metric):
    """Separate arithmetic/bootstrap implementation; same declared seed and strata."""
    paired = [row for row in rows if row['scores'][control][metric] is not None
              and row['scores'][method][metric] is not None]
    values = np.array([[row['scores'][control][metric], row['scores'][method][metric]]
                       for row in paired], dtype=np.float64)
    require(len(values) > 0, 'No paired values')
    sign = -1 if metric == 'LPIPS' else 1
    means = values.mean(axis=0)
    rng = np.random.default_rng(20261110)
    samples = np.zeros((5000, 3), dtype=np.float64)
    for source in sorted({row['source'] for row in paired}):
        selected = [i for i, row in enumerate(paired) if row['source'] == source]
        names = [paired[i]['image_identity'] for i in selected]
        unique, group_index = np.unique(names, return_inverse=True)
        totals = np.zeros((len(unique), 3), dtype=np.float64)
        np.add.at(totals[:, :2], group_index, values[selected])
        np.add.at(totals[:, 2], group_index, 1)
        draws = rng.integers(0, len(unique), (5000, len(unique)))
        samples += totals[draws].sum(axis=1)
    absolute = sign*(samples[:, 1]-samples[:, 0])/samples[:, 2]
    denominators = np.abs(samples[:, 0]/samples[:, 2])
    undefined = int(np.count_nonzero(denominators == 0))
    delta = sign*(values[:, 1]-values[:, 0]).mean()
    return dict(n=len(paired), images=len({row['image_identity'] for row in paired}),
        baseline=float(means[0]), method=float(means[1]), absolute_gain=float(delta),
        relative_gain_percent=float(100*delta/abs(means[0])) if means[0] else None,
        absolute_ci=np.percentile(absolute, [2.5, 97.5]).tolist(),
        relative_ci_percent=[None, None] if undefined else
            np.percentile(100*absolute/denominators, [2.5, 97.5]).tolist(),
        undefined_relative_bootstrap_samples=undefined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-dir', type=Path,
                        default=ROOT/'output/surface-lastpass-ablation-20261111-recovery')
    parser.add_argument('--full-recovery-dir', type=Path,
                        default=ROOT/'output/surface-job-recovery-20261110')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'output/surface-lastpass-review-20261111')
    args = parser.parse_args()
    receipt = json.loads((args.recovery_dir/'receipt.json').read_text())
    require(sha256(args.recovery_dir/'evidence.tar.gz') == receipt['sha256'],
            'Diagnostic archive differs')
    recovered = args.recovery_dir/'evidence'
    for name, expected in receipt['member_sha256'].items():
        require(sha256(recovered/name) == expected, 'Recovered diagnostic member differs: '+name)
    full_receipt = json.loads((args.full_recovery_dir/'progressive-receipt.json').read_text())
    full_root = args.full_recovery_dir/'progressive'
    # Verify only files read here against the independently recovered full-run manifest.
    checked_full = set()
    def read_full(path):
        relative = path.relative_to(full_root).as_posix()
        require(sha256(path) == full_receipt['member_sha256'][relative],
                'Original full-run member differs: '+relative)
        checked_full.add(relative)
        return json.loads(path.read_text())
    run = recovered/'output/surface-lastpass-ablation-20261111'
    freeze_path = run/'analysis/freeze.json'
    require(sha256(freeze_path) == FREEZE_SHA, 'Frozen diagnostic changed')
    freeze = json.loads(freeze_path.read_text())
    summary = json.loads((run/'analysis/summary.json').read_text())
    official, _ = read_official(ROOT/'datasets/open_world_amodal')
    require([row['id'] for row in official] == freeze['target_ids'], 'Official target order changed')
    paths = list(run.glob('targets/*/*/ablation.json'))
    rows = {row['id']: row for row in (json.loads(path.read_text()) for path in paths)}
    require(len(rows) == len(paths) == len(official) == 2565
            and set(rows) == set(freeze['target_ids']), 'Incomplete or duplicate target inventory')
    runs = {side: full_root/'output'/name for side, name in
            (('baseline', 'pdmc-matched-control-20261110'), ('surface', 'pdmc-surface-progressive-20261110'))}
    for side, folder in runs.items():
        for filename, expected in freeze['runs'][side].items():
            read_full(folder/filename)
            require(sha256(folder/filename) == expected, 'Frozen original run differs')
    inventories = {side: Counter() for side in runs}
    drift = {side: {metric: 0.0 for metric in METRICS} for side in runs}
    for item in official:
        row = rows[item['id']]
        records, cached, hashes = {}, {}, {}
        for side, folder in runs.items():
            target = folder/'targets'/item['id']
            records[side] = read_full(target/'result.json')
            hashes[side] = sha256(target/'result.json')
            cached[side] = read_full(target/'metrics.json')['metrics'] if records[side]['status']=='success' else PENALTIES.copy()
            inventories[side][records[side]['status']] += 1
        check_target(row, item, records, cached, hashes, FREEZE_SHA)
        for side in runs:
            for metric, prior in cached[side].items():
                if prior is not None:
                    drift[side][metric] = max(drift[side][metric], abs(row['scores'][side][metric]-prior))
    runtimes = [json.loads((run/f'runtime-worker-{i}.json').read_text()) for i in range(4)]
    frozen_at = datetime.fromisoformat(freeze['created_at'])
    for index, runtime in enumerate(runtimes):
        require(runtime['worker'] == index and runtime['freeze_sha256'] == FREEZE_SHA,
                'Runtime worker identity differs')
        require(frozen_at < datetime.fromisoformat(runtime['started_at'])
                < datetime.fromisoformat(runtime['finished_at']), 'Runtime timing differs')
        require(runtime['environment']['packages'] == runtimes[0]['environment']['packages'],
                'Worker packages differ')
    require(len({runtime['job_id'] for runtime in runtimes}) == 1, 'Worker allocations differ')
    ordered = [rows[item['id']] for item in official]
    contrasts = {}
    for control, method in (('baseline', 'surface'), ('baseline', 'raw_final'), ('surface', 'raw_final')):
        key = method+'_vs_'+control
        contrasts[key] = {}
        for metric in METRICS:
            result = paired_estimate(ordered, control, method, metric)
            reported = summary['contrasts'][key][metric]
            for field, value in result.items():
                np.testing.assert_allclose(value, reported[field], atol=1e-10, rtol=0)
            contrasts[key][metric] = result
    require(drift == summary['original_score_max_absolute_drift'], 'Replay drift summary differs')
    require(summary['targets'] == 2565 and summary['derived_images'] == inventories['surface']['success'] == 958,
            'Summary target counts differ')
    audit = dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(),
        targets=len(rows), derived_images=958, status_counts=inventories,
        recovered_member_checks=len(receipt['member_sha256']), original_member_checks=len(checked_full),
        contrasts=contrasts, original_score_max_absolute_drift=drift,
        native_pngs_locally_verified=False, neural_metrics_recomputed=False,
        scope=freeze['scope'], source_sha256=sha256(Path(__file__)))
    write_json(args.output_dir/'audit.json', audit)
    write_json(args.output_dir/'paired_records.json', ordered)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
