#!/usr/bin/env python3
"""Independently audit recovered full-run metadata and quantify coverage effects."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json

METRICS = ('CLIP', 'LPIPS', 'Feature Similarity', 'SSIM')
PENALTY = {'CLIP': None, 'LPIPS': 1., 'Feature Similarity': 0., 'SSIM': 0.}


def estimate(rows, metric):
    selected = [row for row in rows if row['baseline'][metric] is not None
                and row['surface'][metric] is not None]
    if not selected:
        return None
    first = np.array([row['baseline'][metric] for row in selected])
    second = np.array([row['surface'][metric] for row in selected])
    direction = -1 if metric == 'LPIPS' else 1
    rng = np.random.default_rng(20261110)
    sampled = np.zeros((5000, 3))
    for source in sorted({row['source'] for row in selected}):
        grouped = defaultdict(list)
        for row in selected:
            if row['source'] == source:
                grouped[row['image_identity']].append(row)
        totals = np.array([[sum(row['baseline'][metric] for row in group),
                            sum(row['surface'][metric] for row in group), len(group)]
                           for _, group in sorted(grouped.items())])
        indices = rng.integers(len(totals), size=(5000, len(totals)))
        sampled += totals[indices].sum(axis=1)
    difference = direction * (sampled[:, 1] - sampled[:, 0]) / sampled[:, 2]
    denominator = np.abs(sampled[:, 0] / sampled[:, 2])
    undefined = int((denominator == 0).sum())
    return dict(n=len(selected), images=len({row['image_identity'] for row in selected}),
        baseline=float(first.mean()), method=float(second.mean()),
        absolute_gain=float(direction * (second-first).mean()),
        relative_gain_percent=float(100 * direction * (second-first).mean() / abs(first.mean()))
            if first.mean() else None,
        absolute_ci=np.percentile(difference, [2.5, 97.5]).tolist(),
        relative_ci_percent=[None, None] if undefined else
            np.percentile(100 * difference / denominator, [2.5, 97.5]).tolist(),
        undefined_relative_bootstrap_samples=undefined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-dir', type=Path, default=ROOT/'output/surface-job-recovery-20261110')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output/surface-progressive-review-20261111')
    args = parser.parse_args()
    receipt = json.loads((args.recovery_dir/'progressive-receipt.json').read_text())
    assert sha256(args.recovery_dir/'progressive-evidence.tar.gz') == receipt['archive_sha256']
    recovered = args.recovery_dir/'progressive'
    for relative, expected in receipt['member_sha256'].items():
        assert sha256(recovered/relative) == expected, relative
    runs = [recovered/'output'/name for name in
            ('pdmc-matched-control-20261110', 'pdmc-surface-progressive-20261110')]
    comparison_dir = recovered/'output/surface-followup-20261110/progressive-comparison'
    comparison = json.loads((comparison_dir/'comparison.json').read_text())
    configs = [json.loads((run/'run_config.json').read_text()) for run in runs]
    policy = configs[1].pop('surface_policy')
    assert configs[0] == configs[1]
    assert policy['freeze_sha256'] == comparison['freeze_sha256'] == \
        'a0a626fcfddec095a1d7b677b496309fccb79ff6b4e446f5127b69368f3358e1'
    assert (runs[0]/'metric_config.json').read_bytes() == (runs[1]/'metric_config.json').read_bytes()
    rows, _ = read_official(ROOT/'datasets/open_world_amodal')
    assert len(rows) == comparison['targets'] == 2565
    pairs, transitions, inventories = [], Counter(), [Counter(), Counter()]
    preserved_records = 0
    unfinished_snapshots = []
    for official in rows:
        records, metrics = [], []
        for index, run in enumerate(runs):
            folder = run/'targets'/official['id']
            record = json.loads((folder/'result.json').read_text())
            assert record['id'] == official['id']
            records.append(record)
            inventories[index][record['status']] += 1
            if record['status'] == 'success':
                cache = json.loads((folder/'metrics.json').read_text())
                assert cache['inputs']['completion_sha256'] == record['completion_sha256']
                assert cache['inputs']['reference_sha256'] in record['evaluation_reference_artifacts'].values()
                metrics.append(cache['metrics'])
            else:
                metrics.append(PENALTY.copy())
        for key in ('input_sha256', 'evaluation_reference_artifacts', 'seed'):
            assert records[0][key] == records[1][key], (official['id'], key)
        first, second = [record['status'] == 'success' for record in records]
        transition = 'shared' if first and second else 'lost' if first else 'gained' if second else 'both_failed'
        transitions[transition] += 1
        pairs.append(dict(id=official['id'], source=official['source'],
            image_identity=official['source']+'/'+official['original_id'], transition=transition,
            baseline=metrics[0], surface=metrics[1],
            baseline_iterations=records[0].get('iterations'), surface_iterations=records[1].get('iterations'),
            baseline_capped=records[0].get('iteration_limit_reached'),
            surface_capped=records[1].get('iteration_limit_reached')))
    for run in runs:
        execution = json.loads((run/'execution.json').read_text())
        assert execution['status'] == 'complete'
        for identity, expected in execution['result_artifacts'].items():
            assert sha256(run/'targets'/identity/'result.json') == expected
        for path in (run/'provenance').glob('*/execution.json'):
            historical = json.loads(path.read_text())
            if historical['id'] == execution['id']:
                continue
            if 'result_artifacts' not in historical:
                unfinished_snapshots.append(dict(run=run.name, id=historical['id'], status=historical['status']))
            for identity, expected in historical.get('result_artifacts', {}).items():
                assert sha256(run/'targets'/identity/'result.json') == expected
                preserved_records += 1
    before = json.loads((ROOT/'output/surface-followup-20261110-design/resume-20261111-before.json').read_text())
    resumed_checks = 0
    for run in runs:
        for relative, expected in before[run.name]['result_sha256'].items():
            assert sha256(run/relative) == expected
            resumed_checks += 1
    metrics, contributions, shared_metrics = {}, {}, {}
    for metric in METRICS:
        measured = estimate(pairs, metric)
        reported = comparison['metrics'][metric]
        for key in ('n', 'images', 'baseline', 'method', 'absolute_gain',
                    'relative_gain_percent', 'absolute_ci', 'relative_ci_percent'):
            np.testing.assert_allclose(measured[key], reported[key], atol=1e-10, rtol=0)
        metrics[metric] = measured
        shared_metrics[metric] = estimate([row for row in pairs if row['transition']=='shared'], metric)
        sign = -1 if metric == 'LPIPS' else 1
        contributions[metric] = {kind: sum(sign*(row['surface'][metric]-row['baseline'][metric])
            for row in pairs if row['transition']==kind and row['baseline'][metric] is not None
            and row['surface'][metric] is not None)/measured['n'] for kind in transitions}
        for side in ('baseline', 'surface'):
            values = [row[side][metric] for row in pairs if row[side][metric] is not None]
            reported_overall = comparison['overall_metrics'][metric]
            assert len(values) == reported_overall[side+'_n']
            np.testing.assert_allclose(np.mean(values), reported_overall[side], atol=1e-10, rtol=0)
    assert transitions['shared'] == comparison['shared_successes'] == 946
    assert transitions['gained'] == comparison['gained_successes'] == 12
    assert transitions['lost'] == comparison['lost_successes'] == 5
    audit = dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(),
        recovered_member_checks=len(receipt['member_sha256']), targets=len(rows),
        historical_result_hash_checks=preserved_records, pre_resume_result_hash_checks=resumed_checks,
        unfinished_historical_snapshots=unfinished_snapshots, status_counts=inventories,
        transitions=transitions, metrics=metrics, shared_success_metrics=shared_metrics,
        absolute_gain_contributions=contributions,
        full_metric_point_target_met=all(row['relative_gain_percent'] >= 3 for row in metrics.values()),
        cap_counts={side:sum(row['transition'] in ('shared', 'gained' if side=='surface' else 'lost')
            and bool(row[side+'_capped']) for row in pairs) for side in ('baseline', 'surface')},
        native_pngs_locally_verified=False, neural_metrics_recomputed=False,
        scope='Independent metadata, checksum, arithmetic, and bootstrap audit. Remote full/provenance verification is separate.',
        source_sha256=sha256(Path(__file__)))
    write_json(args.output_dir/'audit.json', audit)
    write_json(args.output_dir/'paired_records.json', pairs)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
