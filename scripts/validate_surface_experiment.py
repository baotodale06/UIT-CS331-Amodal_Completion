#!/usr/bin/env python3
"""Independently check experiment inputs, artifacts, score replay and selections."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.surface import score_surface, SurfaceConfig, METHODS, select_candidate
from amodal_benchmark.surface_evaluation import load_pool, evaluate_pool, selections, summarize_cases
from scripts.prepare_surface_experiment import make_case


def assert_equivalent(actual, expected, location='value', tolerance=1e-10):
    """Reject missing fields, nonfinite values and changed evaluation results."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Evaluation fields changed: ' + location)
        for key in expected:
            assert_equivalent(actual[key], expected[key], location + '/' + key, tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Evaluation length changed: ' + location)
        for index, (first, second) in enumerate(zip(actual, expected)):
            assert_equivalent(first, second, location + '/' + str(index), tolerance)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(actual, (float, int)) or not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tolerance:
            raise ValueError('Evaluation number changed: ' + location)
    elif actual != expected:
        raise ValueError('Evaluation value changed: ' + location)


def validate_evaluation_rows(experiment, run, cases, protocol, lock):
    """Recompute hidden errors from PNGs and every policy from candidate records."""
    by_id = {case['case_id']: case for case in cases}
    metric_checks = selection_checks = summary_checks = 0
    for path in (experiment / 'analysis').glob('*_cases.json'):
        split = path.name.removesuffix('_cases.json')
        rows = json.loads(path.read_text())
        expected_ids = {case['case_id'] for case in cases if case['split'] == split}
        if len(rows) != len(expected_ids) or {row['case_id'] for row in rows} != expected_ids:
            raise ValueError('Analysis coverage changed: ' + split)
        for row in rows:
            case = by_id[row['case_id']]
            records = load_pool(experiment, run, case, protocol['candidate_seeds'])
            metrics = evaluate_pool(experiment, run, case, records)
            assert_equivalent(metrics, row['metrics'], row['case_id'] + '/metrics')
            assert_equivalent([r.get('surface', {'valid': False}) for r in records], row['scores'], row['case_id'] + '/scores')
            method = lock['development_method'] if split == 'test' else None
            chosen = selections(records, metrics, lock['scales'], method)
            assert_equivalent(chosen, row['selected'], row['case_id'] + '/selections')
            metric_checks += len(metrics)
            selection_checks += len(chosen)
        summary = json.loads((experiment / 'analysis' / (split + '_summary.json')).read_text())
        assert_equivalent(summarize_cases(rows, split), summary['methods'], split + '/summary')
        summary_checks += len(summary['methods'])
    return dict(candidate_metric_replays=metric_checks, all_policy_replays=selection_checks,
                method_summary_replays=summary_checks)


def validate_case(arguments):
    experiment, run, dataset, reference_run, case, official, protocol = arguments
    experiment, run, dataset, reference_run = map(Path, (experiment, run, dataset, reference_run))
    count = 0
    source = dataset / case['source_image']
    reference = reference_run / 'visible_references/masks' / official['reference']
    if sha256(source) != case['source_sha256'] or sha256(reference) != case['reference_sha256']:
        raise ValueError('Source/reference hash mismatch: ' + case['case_id'])
    mask = np.array(Image.open(reference).convert('L')) > 0
    regenerated = make_case(Image.open(source).convert('RGB'), mask, case['direction'],
                            protocol['hidden_fraction'], protocol['resolution'])
    for section, hashes in [('inputs', case['input_hashes']), ('evaluation_only', case['evaluation_hashes'])]:
        for name, expected in hashes.items():
            path = experiment / section / case['case_id'] / name
            if sha256(path) != expected:
                raise ValueError('Prepared artifact hash mismatch: ' + str(path))
            count += 1
            actual = np.array(Image.open(path))
            desired = regenerated[Path(name).stem]
            if desired.ndim == 2:
                desired = desired.astype(np.uint8) * 255
            if not np.array_equal(actual, desired):
                raise ValueError('Input preparation replay mismatch: ' + str(path))
    visible, occlusion, observed = regenerated['visible'], regenerated['occlusion'], regenerated['observed']
    if not np.all(observed[occlusion] == 127) or (visible & occlusion).any():
        raise ValueError('Hidden RGB leakage in prepared observations')
    diffs, statuses, rgb_hashes = [], [], []
    for seed in protocol['candidate_seeds']:
        folder = run / 'candidates' / case['case_id']
        record = json.loads((folder / f'seed-{seed}.json').read_text())
        if record['seed'] != seed or record['case_id'] != case['case_id'] or record['input_hashes'] != case['input_hashes']:
            raise ValueError('Generation identity mismatch')
        statuses.append(record['status'])
        for name, expected in record['artifacts'].items():
            path = folder / name
            if sha256(path) != expected:
                raise ValueError('Candidate changed: ' + str(path))
            with Image.open(path) as image:
                image.load()
                if image.size != (protocol['resolution'],) * 2:
                    raise ValueError('Candidate frame changed')
            count += 1
        if f'seed-{seed}.png' in record['artifacts']:
            rgb_hashes.append(record['artifacts'][f'seed-{seed}.png'])
        if record.get('surface'):
            candidate = np.array(Image.open(folder / f'seed-{seed}.png').convert('RGB'))
            predicted = np.array(Image.open(folder / f'seed-{seed}-mask.png').convert('L')) > 0
            replay = score_surface(observed, visible, occlusion, candidate, predicted,
                SurfaceConfig(radius=protocol['boundary_radius'], min_region_pixels=protocol['min_region_pixels'],
                              min_visible_recall=protocol['min_visible_recall']))
            for key, expected in record['surface'].items():
                actual = replay[key]
                if isinstance(expected, float):
                    difference = abs(expected - actual)
                    if not np.isfinite(difference) or difference > 1e-4:
                        raise ValueError(f'Score replay differs: {case["case_id"]}/{seed}/{key}: {difference}')
                    diffs.append(difference)
                elif actual != expected:
                    raise ValueError(f'Score support/validity differs: {key}')
    return dict(case_id=case['case_id'], artifact_checks=count, max_score_difference=max(diffs, default=0),
                statuses=statuses, distinct_rgb_candidates=len(set(rgb_hashes)), generated_rgb_candidates=len(rgb_hashes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--freeze-from', type=Path)
    p.add_argument('--smoke-dir', type=Path)
    args = p.parse_args()
    cases = json.loads((args.experiment_dir / 'cases.json').read_text())
    protocol = json.loads((args.experiment_dir / 'protocol.json').read_text())
    official = {r['id']: r for r in read_official(args.dataset_root)[0]}
    if len({c['image_identity'] for c in cases}) != len(cases) or len({c['source_sha256'] for c in cases}) != len(cases):
        raise ValueError('Case/source leakage')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(validate_case, [(str(args.experiment_dir), str(args.run_dir),
            str(args.dataset_root), str(args.reference_run), case, official[case['target_id']], protocol) for case in cases]))
    lock_path = args.freeze_from or args.experiment_dir / 'analysis/development_freeze.json'
    lock = json.loads(lock_path.read_text())
    for relative, expected in lock['analysis_source_hashes'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError('Frozen scientific source changed: ' + relative)
    if not args.freeze_from:
        for relative, expected in lock['record_hashes'].items():
            if sha256(args.run_dir / relative) != expected:
                raise ValueError('Frozen development candidate changed: ' + relative)
    evaluation_replays = validate_evaluation_rows(args.experiment_dir, args.run_dir, cases, protocol, lock)
    selection_checks = 0
    for path in (args.experiment_dir / 'analysis').glob('*_cases.json'):
        for row in json.loads(path.read_text()):
            for name, weights in METHODS.items():
                replay = select_candidate(row['scores'], weights, lock['scales'])
                if replay != row['selected'][name]:
                    raise ValueError('Selection replay mismatch')
                selection_checks += 1
    repeated = []
    if args.smoke_dir:
        for path in (args.smoke_dir / 'candidates').rglob('seed-*.json'):
            smoke = json.loads(path.read_text())
            full = json.loads((args.run_dir / 'candidates' / smoke['case_id'] / path.name).read_text())
            # Numeric precision of the scorer was repaired after the smoke; RGB
            # and segmentation were generated with identical inference settings.
            for name, expected in smoke['artifacts'].items():
                repeated.append(dict(case_id=smoke['case_id'], seed=smoke['seed'], artifact=name,
                                     identical=full['artifacts'].get(name) == expected))
    result = dict(status='passed', cases=len(cases), source_identity_checks=2 * len(cases),
        **evaluation_replays,
        artifact_checks=sum(r['artifact_checks'] for r in records), selection_checks=selection_checks,
        max_score_replay_difference=max(r['max_score_difference'] for r in records),
        candidate_statuses=Counter(s for r in records for s in r['statuses']),
        pools_with_duplicate_rgbs=sum(r['distinct_rgb_candidates'] != r['generated_rgb_candidates'] for r in records),
        seed_repeat_checks=repeated, scorer_sha256=sha256(ROOT / 'amodal_benchmark/surface.py'),
        validator_sha256=sha256(Path(__file__)), freeze_sha256=sha256(lock_path),
        preparation='Every prepared RGB/mask array exactly replayed from original source image and SAM mask')
    write_json(args.experiment_dir / 'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
