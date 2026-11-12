#!/usr/bin/env python3
"""Compare complete progressive runs with identical inputs and frozen references."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.cohort import select_manifest_rows, cohort_identity
from amodal_benchmark.protocol import METRICS, failure_metrics
from amodal_benchmark.registry import register_run
from amodal_benchmark.verify import verify_run
from finalize_surface_open_world import image_cluster_interval


def format_number(value, specification='.6f'):
    return 'unavailable' if value is None else format(value, specification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'surface', 'dataset-root', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--target-manifest', type=Path,
                        help='Compare a frozen research cohort; never register it as a full run')
    args = parser.parse_args()
    if args.target_manifest and args.registry:
        parser.error('A research cohort cannot be registered as a full experiment')
    reports = [verify_run(folder, args.dataset_root, require_full=not bool(args.target_manifest), require_provenance=True)
               for folder in (args.baseline, args.surface)]
    for key in ('targets_sha256', 'inputs_sha256', 'references_sha256'):
        if reports[0]['data_identity'][key] != reports[1]['data_identity'][key]:
            raise ValueError('Paired progressive comparison identity differs: ' + key)
    baseline_config = json.loads((args.baseline / 'run_config.json').read_text())
    surface_config = json.loads((args.surface / 'run_config.json').read_text())
    method = surface_config.pop('surface_policy')
    if baseline_config != surface_config:
        raise ValueError('Progressive configurations differ beyond the frozen surface policy')
    metric_configs = [json.loads((folder / 'metric_config.json').read_text())
                      for folder in (args.baseline, args.surface)]
    if metric_configs[0] != metric_configs[1]:
        raise ValueError('The two runs use different metric operations')
    environments = []
    for folder in (args.baseline, args.surface):
        execution = json.loads((folder / 'execution.json').read_text())
        environments.append(json.loads((folder / 'provenance' / execution['id'] / 'environment.json').read_text()))
    if environments[0]['packages'] != environments[1]['packages']:
        raise ValueError('Matched runs have different installed packages')
    rows, _ = read_official(args.dataset_root)
    if args.target_manifest:
        rows = select_manifest_rows(rows, args.target_manifest)
        expected_ids = [row['id'] for row in rows]
        for folder in (args.baseline, args.surface):
            execution = json.loads((folder / 'execution.json').read_text())
            if execution['target_ids'] != expected_ids:
                raise ValueError('Research run inventory differs from the frozen cohort')
    paired, transitions, shared, lost, gained = [], Counter(), 0, 0, 0
    for row in rows:
        records, metrics = [], []
        for folder in (args.baseline, args.surface):
            target = folder / 'targets' / row['id']
            record = json.loads((target / 'result.json').read_text())
            records.append(record)
            metrics.append(json.loads((target / 'metrics.json').read_text())['metrics']
                           if record['status'] == 'success' else failure_metrics())
        for key in ('input_sha256', 'evaluation_reference_artifacts', 'seed'):
            if records[0].get(key) != records[1].get(key):
                raise ValueError('Target comparison differs: ' + row['id'] + '/' + key)
        first, second = [record['status'] == 'success' for record in records]
        shared += int(first and second)
        lost += int(first and not second)
        gained += int(second and not first)
        transitions[records[0]['status'] + ' -> ' + records[1]['status']] += 1
        paired.append(dict(id=row['id'], source=row['source'],
                           image_identity=row['source'] + '/' + row['original_id'],
                           baseline=metrics[0], surface=metrics[1]))
    comparisons = {}
    for metric in METRICS:
        keep = [item for item in paired if item['baseline'][metric] is not None and item['surface'][metric] is not None]
        comparisons[metric] = image_cluster_interval(keep, metric) if keep else dict(
            n=0, images=0, baseline=None, method=None, absolute_gain=None,
            absolute_ci=[None, None], relative_gain_percent=None,
            relative_ci_percent=[None, None], unavailable_reason='No shared metric observations')
        comparisons[metric]['excluded_pairs'] = len(rows) - len(keep)
    overall_metrics = {
        metric: dict(baseline=reports[0]['statistics']['Overall'][metric],
                     surface=reports[1]['statistics']['Overall'][metric],
                     baseline_n=reports[0]['statistics']['Overall'][metric + '_n'],
                     surface_n=reports[1]['statistics']['Overall'][metric + '_n'])
        for metric in METRICS
    }
    analysis_sources = [ROOT / 'scripts' / name for name in (
        'compare_surface_progressive.py', 'finalize_surface_open_world.py',
        'analyze_surface_followup.py')]
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), metrics=comparisons,
        overall_metrics=overall_metrics,
        targets=len(rows), shared_successes=shared, lost_successes=lost, gained_successes=gained,
        status_transitions=dict(transitions), same_inputs=True, same_references=True,
        baseline_statistics=reports[0]['statistics'], surface_statistics=reports[1]['statistics'],
        candidate_budget=len(method['frozen_method']['candidate_seeds']),
        freeze_sha256=method['freeze_sha256'], source_sha256=sha256(Path(__file__)),
        analysis_source_hashes={path.relative_to(ROOT).as_posix(): sha256(path)
                                for path in analysis_sources},
        baseline_execution_sha256=sha256(args.baseline / 'execution.json'),
        surface_execution_sha256=sha256(args.surface / 'execution.json'),
        evaluation_scope='frozen research cohort' if args.target_manifest else 'full benchmark',
        research_cohort=cohort_identity(args.target_manifest) if args.target_manifest else None,
        limitation='CLIP paired contrast uses shared successes. Overall CLIP means can cover different targets '
                   'and are descriptive, not a paired treatment effect. '
                   'Surface uses additional sampling. This natural-occlusion benchmark provides no hidden RGB MAE.')
    for path in analysis_sources:
        destination = args.output_dir / 'analysis_source' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for name, verification in zip(('baseline', 'surface'), reports):
        write_json(args.output_dir / (name + '-verification.json'), verification)
    write_json(args.output_dir / 'comparison.json', report)
    lines = ['# Matched progressive Surface-PMC comparison', '',
             f'{len(rows)} targets; shared successes {shared}, gained {gained}, lost {lost}.', '',
             '| Metric | Paired control | Surface | Relative gain | 95% interval | n |',
             '| --- | ---: | ---: | ---: | --- | ---: |']
    for metric, value in comparisons.items():
        low, high = value['relative_ci_percent']
        interval = 'unavailable' if low is None or high is None else f'[{low:+.3f}%, {high:+.3f}%]'
        gain = 'unavailable' if value['relative_gain_percent'] is None else f'{value["relative_gain_percent"]:+.3f}%'
        lines.append(f'| {metric} | {format_number(value["baseline"])} | {format_number(value["method"])} | '
                     f'{gain} | {interval} | {value["n"]} |')
    lines += ['', report['limitation'], '', '## Overall run means and denominators', '',
              '| Metric | Control mean | Control n | Surface mean | Surface n |',
              '| --- | ---: | ---: | ---: | ---: |']
    for metric, value in overall_metrics.items():
        first, second = format_number(value['baseline']), format_number(value['surface'])
        lines.append(f'| {metric} | {first} | {value["baseline_n"]} | {second} | {value["surface_n"]} |')
    lines += ['', 'LPIPS, VGG feature similarity and SSIM retain the configured failure penalties. '
              'CLIP has no invented failure score.', '', '## Saved-output coverage', '',
              '| Source | Targets | Control saved | Control failures | Surface saved | Surface failures |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for source, first in reports[0]['statistics'].items():
        second = reports[1]['statistics'][source]
        lines.append(f'| {source} | {first["targets"]} | {first["successes"]} | {first["failures"]} | '
                     f'{second["successes"]} | {second["failures"]} |')
    lines += ['',
              'Relative intervals are unavailable when bootstrap baseline means include zero; '
              'these draws are counted in comparison.json and are never silently dropped. '
              'Absolute intervals remain available. A metric with no shared observations has no paired estimate.', '',
              'Inputs, references, model checkpoints, metric operations and installed packages match. '
              'The sole run-contract extension is the frozen progressive surface policy. '
              'Intervals resample source images while retaining all their targets.']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if args.registry:
        for folder, hypothesis in ((args.baseline, 'Matched runtime control for frozen Surface-PMC'),
                                   (args.surface, 'Eight-candidate observation-anchored Surface-PMC improves progressive output')):
            register_run(folder, args.dataset_root, args.registry, hypothesis=hypothesis)
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
