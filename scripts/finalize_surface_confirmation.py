#!/usr/bin/env python3
"""Verify and report the single predeclared selection-only confirmation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_policy import verify_frozen_surface_sources


def check_cohort(cases, freeze):
    contract = freeze['confirmation']
    if len(cases) != contract['expected_images'] or dict(Counter(c['source'] for c in cases)) != contract['sources']:
        raise ValueError('Incomplete or changed confirmation cohort')
    for key, excluded in [('image_identity', 'excluded_image_identities'), ('source_sha256', 'excluded_image_sha256')]:
        values = [c[key] for c in cases]
        if len(set(values)) != len(values) or set(values) & set(contract[excluded]):
            raise ValueError('Confirmation image overlap: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    verify_frozen_surface_sources(freeze)
    for relative, expected in freeze['confirmation']['input_sha256'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError('Confirmation input changed: ' + relative)
    parent_path = ROOT / 'output/surface-followup-20261110-design/final-method-freeze.json'
    if sha256(parent_path) != freeze['parent_freeze_sha256']:
        raise ValueError('Parent freeze changed')
    parent = json.loads(parent_path.read_text())
    if freeze['ranker'] != parent['ranker'] or freeze['candidate_seeds'] != parent['candidate_seeds']:
        raise ValueError('Ranker or candidate budget changed')
    if (freeze.get('ensemble') or freeze['refinement']['mode'] != 'none'
            or freeze['refinement']['preserve_visible'] or freeze['refinement']['strength'] != 0):
        raise ValueError('Selection-only must leave all RGB unchanged')
    experiment = args.experiment_dir
    cases = json.loads((experiment / 'cases.json').read_text())
    check_cohort(cases, freeze)
    if args.preflight:
        print('PASS immutable128-case confirmation, disjoint from349 prior images, unchanged ranker')
        return
    inference = json.loads((experiment / 'inference/contract.json').read_text())
    if inference['freeze_sha256'] != sha256(args.freeze) or inference['evaluation_data_access']:
        raise ValueError('Observation-only inference contract mismatch')
    for index in range(4):
        runtime = json.loads((experiment / f'metrics/runtime-worker-{index}.json').read_text())
        if datetime.fromisoformat(inference['finished_at']) > datetime.fromisoformat(runtime['started_at']):
            raise ValueError('Metrics began before all independent choices were published')
    replay = json.loads((experiment / 'analysis/replay-verification.json').read_text())
    if not replay['passed'] or replay['images'] != len(cases) or replay['freeze_sha256'] != sha256(args.freeze):
        raise ValueError('Complete native replay required')
    for case in cases:
        row = json.loads((experiment / 'metrics' / (case['case_id'] + '.json')).read_text())
        decision = json.loads((experiment / 'inference/decisions' / (case['case_id'] + '.json')).read_text())
        if row['candidate_seeds'] != freeze['candidate_seeds'] or row['selection']['trained_surface'] != decision['index']:
            raise ValueError('Candidate/selection mismatch')
        if row['raw'] != row['refined'] or row['raw'] != row['calibrated']:
            raise ValueError('A correction unexpectedly changed raw metrics')
        if decision['status'] == 'generated':
            name = f"seed-{decision['seed']}.png"
            candidate = experiment / 'full/candidates' / case['case_id'] / name
            # PNG serialization is verified by decoded native replay; copy the
            # exact selected generator bytes for the deployable artifact.
            destination = experiment / 'selection-only' / case['case_id']
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, destination / 'selected.png')
            shutil.copyfile(candidate.with_name(name.replace('.png', '-mask.png')), destination / 'mask.png')
    summary = json.loads((experiment / 'analysis/summary.json').read_text())
    values = {metric: summary['metrics'][metric]['trained_surface_raw']['uniform_raw'] for metric in freeze['endpoints']}
    point_success = all(v['relative_gain_percent'] >= 3 and v['n'] == len(cases) for v in values.values())
    interval_success = all(v['relative_ci_percent'][0] >= 3 and v['n'] == len(cases) for v in values.values())
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), images=len(cases),
        primary_policy=freeze['primary_policy'], primary_control=freeze['primary_control'],
        freeze_sha256=sha256(args.freeze), metrics=values, point_target_met=point_success,
        lower_95_percent_bounds_at_least3=interval_success, generation_failures=summary['generation_failures'],
        original_five_hour_deadline_missed=True, replay_passed=True,
        source_sha256=sha256(Path(__file__)),
        scope='Independent controlled confirmation on64 VG and64 COCOA images; not full progressive evaluation.')
    write_json(experiment / 'analysis/confirmation.json', result)
    lines = ['# Independent selection-only confirmation', '',
        '**All four3% point targets met.**' if point_success else '**The all-metric3% point target was not met.**', '',
        'The unchanged eight-candidate Surface-PMC ranker selects an unmodified generated image.',
        'All128 images were frozen before generation and excluded all349 previously used images.',
        'The candidate-average baseline is expected uniform selection from the identical eight-candidate pool.', '',
        '| Metric | Uniform baseline | Selection only | Relative gain | Paired95% interval | n |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for name, value in values.items():
        lo, hi = value['relative_ci_percent']
        lines.append(f"| {name} | {value['baseline']:.6f} | {value['method']:.6f} | {value['relative_gain_percent']:+.3f}% | [{lo:+.3f}, {hi:+.3f}]% | {value['n']} |")
    lines += ['', f"Generation failures: {summary['generation_failures']}; candidate errors: {summary['candidate_errors']}.",
        'Intervals use5000 paired source-stratified image resamples. Lower bounds all>=3%: ' + str(interval_success) + '.',
        'These are artificial occlusions of originally observed pixels, with paper-style perceptual metrics.',
        'This test covers VG and COCOA only. It does not establish full progressive benchmark improvement.',
        'Selection-only was preferred after the preceding166-case secondary results; the present128 are a new independent test.',
        'The ranker was not refit. CLIP is also an explicit selection feature. Semantic-only and fixed-seed controls are in comparisons.csv.',
        'The original five-hour deadline was missed. Historical primary and secondary results remain unchanged.', '',
        '[Full comparisons](comparisons.csv), [numerical summary](summary.json), [exact native replay](replay-verification.json).']
    (experiment / 'analysis/CONFIRMATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
