#!/usr/bin/env python3
"""Audit annotated cohort artifacts, selection replay, and paired estimates locally."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.cohort import select_manifest_rows
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.protocol import METRICS, failure_metrics
from amodal_benchmark.surface_refinement import predict_rank, refine_surface, RefinementConfig
from amodal_benchmark.surface_ensemble import refine_surface_pool, EnsembleConfig
from amodal_benchmark.surface_runtime import progressive_seed
from amodal_benchmark.verify import verify_run
from audit_surface_progressive_result import estimate


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    design = ROOT / 'output/surface-annotated-20261112-design'
    study = ROOT / 'output/surface-annotated-20261112'
    output = study / 'report'
    freeze = json.loads((design / 'method-freeze.json').read_text())
    parent = ROOT / 'output/surface-followup-20261110-design/final-method-freeze.json'
    require(sha256(parent) == freeze['parent_freeze_sha256'], 'Parent method identity changed')
    prior = json.loads(parent.read_text())
    for key in ('ranker', 'candidate_seeds', 'refinement', 'ensemble', 'pretrained_visible_features'):
        require(freeze.get(key) == prior.get(key), 'Transferred method was refitted: ' + key)
    cohort = json.loads((design / 'cohort.json').read_text())
    previous = json.loads((design / 'previous-images.json').read_text())
    for case in cohort['cases']:
        require(case['image_identity'] not in previous['image_identities'], 'Training/test image overlap')
        require(case['source_sha256'] not in previous['image_sha256'], 'Training/test source bytes overlap')
    require(cohort['image_audit']['passed'], 'Image split audit failed')
    rows, _ = read_official(ROOT / 'datasets/open_world_amodal')
    rows = select_manifest_rows(rows, design / 'cohort.json')
    folders = {side: study / (side + '-cohort') for side in ('baseline', 'surface')}
    verification = {side: verify_run(path, ROOT / 'datasets/open_world_amodal', require_provenance=True)
                    for side, path in folders.items()}
    configs = {side: json.loads((path / 'run_config.json').read_text()) for side, path in folders.items()}
    policy = configs['surface'].pop('surface_policy')
    require(configs['baseline'] == configs['surface'], 'Inference configurations differ')
    require(configs['baseline']['detector_vocabulary']['mode'] == 'annotated', 'Wrong baseline vocabulary')
    require(policy['freeze_sha256'] == sha256(design / 'method-freeze.json'), 'Wrong method freeze')
    for key in ('targets_sha256', 'inputs_sha256', 'references_sha256'):
        require(verification['baseline']['data_identity'][key] == verification['surface']['data_identity'][key],
                'Paired identity differs: ' + key)
    pairs, decisions, reasons, passes, preservation = [], [], Counter(), Counter(), []
    checked_images = 0
    for row in rows:
        records, metrics = {}, {}
        for side, run in folders.items():
            target = run / 'targets' / row['id']
            record = records[side] = json.loads((target / 'result.json').read_text())
            metrics[side] = (json.loads((target / 'metrics.json').read_text())['metrics']
                             if record['status'] == 'success' else failure_metrics())
        for key in ('seed', 'input_sha256', 'evaluation_reference_artifacts'):
            require(records['baseline'].get(key) == records['surface'].get(key), 'Target pair mismatch')
        surface = records['surface']
        traces = surface.get('diagnostics', {}).get('surface_iterations', [])
        passes[len(traces)] += 1
        if not traces:
            reasons[surface.get('failure_reason', surface['status'])] += 1
        for trace in traces:
            folder = folders['surface'] / 'targets' / row['id'] / 'surface' / f'iteration-{trace["iteration"]:02d}'
            require(sha256(folder / 'decision.json') == trace['decision_sha256'], 'Decision hash differs')
            decision = json.loads((folder / 'decision.json').read_text())
            require(decision['inference_truth_access'] is False, 'Unexpected inference truth access')
            require(decision['target_seed'] == surface['seed'], 'Global candidate seed drift')
            seeds = [progressive_seed(surface['seed'], trace['iteration'], value) for value in freeze['candidate_seeds']]
            require(decision['candidate_seeds'] == seeds, 'Candidate seed inventory differs')
            for name, expected in decision['artifacts'].items():
                require(sha256(folder / name) == expected, 'Pool artifact differs: ' + name)
                checked_images += 1
            selected, losses = predict_rank(decision['observable_scores'], freeze['ranker'])
            require(selected == trace['selected'] == decision['index'], 'Learned choice does not replay')
            for first, second in zip(losses, decision['losses']):
                require((first is None and second is None) or
                        (first is not None and second is not None and abs(first - second) < 1e-10),
                        'Logged rank values do not replay')
            observed = np.array(Image.open(folder / 'observed.png').convert('RGB'))
            visible = np.array(Image.open(folder / 'visible.png').convert('L')) > 0
            candidates = [np.array(Image.open(folder / f'candidate-{i}.png').convert('RGB')) for i in range(8)]
            masks = [np.array(Image.open(folder / f'mask-{i}.png').convert('L')) > 0 for i in range(8)]
            calibrated = [refine_surface(observed, visible, image, mask, RefinementConfig(**freeze['refinement']))[0]
                          for image, mask in zip(candidates, masks)]
            refined, _ = refine_surface_pool(calibrated, masks, visible, EnsembleConfig(**freeze['ensemble']))
            recorded = np.array(Image.open(folder / 'selected.png').convert('RGB'))
            error = np.abs(refined[selected].astype(int) - recorded.astype(int))
            require(not error.any(), 'Corrected selected RGB does not replay exactly')
            require(np.array_equal(recorded[visible], observed[visible]), 'Selection-scale visible pixels changed')
            decisions.append(dict(id=row['id'], iteration=trace['iteration'], selected=selected,
                                  fallback=decision['fallback'], exact_rgb_replay=True))
        pairs.append(dict(id=row['id'], source=row['source'], label=row['label'],
            image_identity=row['source'] + '/' + row['original_id'], baseline=metrics['baseline'], surface=metrics['surface'],
            baseline_status=records['baseline']['status'], surface_status=surface['status'],
            baseline_reason=records['baseline'].get('failure_reason'), surface_reason=surface.get('failure_reason'),
            baseline_iterations=records['baseline'].get('iterations'), surface_iterations=surface.get('iterations'),
            surface_passes=len(traces)))
    remote = json.loads((study / 'cohort-comparison/comparison.json').read_text())
    require(len(decisions) == sum(1 for _ in folders['surface'].glob('targets/*/*/surface/*/decision.json')),
            'An unaccounted generated pool is missing from target diagnostics')
    comparisons = {metric: estimate(pairs, metric) for metric in METRICS}
    for metric, values in comparisons.items():
        expected = remote['metrics'][metric]
        if values is None:
            require(expected['n'] == 0, 'Paired metric unexpectedly missing')
            continue
        for key in ('n', 'images', 'baseline', 'method', 'absolute_gain', 'relative_gain_percent', 'absolute_ci', 'relative_ci_percent'):
            np.testing.assert_allclose(values[key], expected[key], atol=1e-12, rtol=1e-12, err_msg=metric + '/' + key)
    transitions = Counter(row['baseline_status'] + ' -> ' + row['surface_status'] for row in pairs)
    decomposition = Counter()
    for row in pairs:
        first, second = row['baseline_status'] == 'success', row['surface_status'] == 'success'
        group = 'shared_success' if first and second else 'gained' if second else 'lost' if first else 'both_failed'
        decomposition[group] += (row['baseline']['LPIPS'] - row['surface']['LPIPS']) / len(pairs)
    per_source = {source: {metric: estimate([row for row in pairs if row['source'] == source], metric)
                           for metric in METRICS} for source in sorted({row['source'] for row in pairs})}
    shared = [row for row in pairs if row['baseline_status'] == row['surface_status'] == 'success']
    result = dict(passed=True, audited_at=datetime.now(timezone.utc).isoformat(),
        targets=len(pairs), images=len(cohort['cases']), metrics=comparisons, per_source=per_source,
        shared_success_metrics={metric: estimate(shared, metric) for metric in METRICS},
        baseline_statistics=verification['baseline']['statistics'], surface_statistics=verification['surface']['statistics'],
        transitions=dict(transitions), lpips_gain_decomposition=dict(decomposition),
        surface_pass_histogram=dict(passes), no_surface_sampling_reasons=dict(reasons),
        decisions_replayed=len(decisions), generated_candidates=8 * len(decisions), pool_image_hashes_checked=checked_images,
        fallback_decisions=sum(row['fallback'] for row in decisions),
        method_parameters_unchanged=True, selected_rgb_exact_replay=True, local_neural_metrics_rerun=False,
        limitation='Selection replay uses logged observable features; pretrained feature extraction is not rerun locally. '
                   'RGB correction replay is exact at the saved candidate resolution, before native JPEG serialization.',
        sources={str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                 'scripts/audit_surface_progressive_result.py': sha256(ROOT / 'scripts/audit_surface_progressive_result.py')})
    write_json(output / 'audit.json', result)
    write_json(output / 'paired_records.json', pairs)
    write_json(output / 'decision_replay.json', decisions)
    print(json.dumps(dict(passed=True, targets=len(pairs), decisions=len(decisions), metrics=comparisons), indent=2))


if __name__ == '__main__':
    main()
