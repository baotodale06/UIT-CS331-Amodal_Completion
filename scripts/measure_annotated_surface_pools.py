#!/usr/bin/env python3
"""Frozen final-pool diagnostic for the annotated progressive cohort."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.cohort import select_manifest_rows
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.protocol import METRICS, failure_metrics
from amodal_benchmark.references import reference_paths
from amodal_benchmark.verify import verify_run
from measure_surface_candidate_headroom import derive_candidate, pool_policies
from finalize_surface_open_world import image_cluster_interval

POLICIES = ('frozen_raw', 'uniform_raw', 'uniform_valid', 'first_raw', 'semantic_only', 'corrected')
ORACLES = tuple('oracle_' + metric for metric in METRICS)


def policies(scores, observables, selected, corrected):
    outputs, choices = pool_policies(scores, observables, selected, corrected)
    valid = [i for i, row in enumerate(observables) if row['valid']] or [0]
    outputs['uniform_valid'] = {metric: float(np.mean([scores[i][metric] for i in valid])) for metric in METRICS}
    outputs['first_raw'] = scores[0]
    choices['first_raw'] = 0
    return outputs, choices


def measure(args, freeze, rows):
    import torch
    from amodal_benchmark.metrics import MetricModels
    from amodal_benchmark.provenance import environment_snapshot
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    roots = {name: ROOT / freeze[name + '_run'] for name in ('baseline', 'surface')}
    configs = [json.loads((path / 'metric_config.json').read_text()) for path in roots.values()]
    if configs[0] != configs[1]:
        raise ValueError('Metric configurations differ')
    config = configs[0]
    runtime = dict(worker=args.worker_index, started_at=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha256(args.freeze), job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(0), environment=environment_snapshot())
    write_json(args.output_dir / f'runtime-worker-{args.worker_index}.json', runtime)
    models = MetricModels()
    for row in rows[args.worker_index::args.worker_count]:
        folders = {name: path / 'targets' / row['id'] for name, path in roots.items()}
        records = {name: json.loads((path / 'result.json').read_text()) for name, path in folders.items()}
        for key in ('seed', 'input_sha256', 'evaluation_reference_artifacts'):
            if records['baseline'][key] != records['surface'][key]:
                raise ValueError('Paired identity mismatch: ' + row['id'] + '/' + key)
        destination = args.output_dir / 'targets' / row['id']
        result_path = destination / 'pool.json'
        inputs = dict(freeze_sha256=sha256(args.freeze), **{
            name + '_result_sha256': sha256(path / 'result.json') for name, path in folders.items()})
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if old['inputs'] != inputs:
                raise ValueError('Pool inputs changed on resume')
            for index, artifact in enumerate(old['candidate_artifacts']):
                if sha256(destination / f'candidate-{index}.png') != artifact['derived_sha256']:
                    raise ValueError('Derived candidate changed')
            continue
        rescored, cached, references = {}, {}, {}
        for name, folder in folders.items():
            record = records[name]
            if record['status'] != 'success':
                rescored[name] = cached[name] = failure_metrics()
                continue
            paths = reference_paths(roots[name] / 'visible_references', row)
            key = 'crop' if config['alignment'] == 'released' else 'full'
            reference = references[name] = paths[key]
            if sha256(reference) != record['evaluation_reference_artifacts'][key]:
                raise ValueError('Reference changed')
            if sha256(folder / 'completion.png') != record['completion_sha256']:
                raise ValueError('Completion changed')
            cached[name] = json.loads((folder / 'metrics.json').read_text())['metrics']
            rescored[name] = models.score(reference, folder / 'completion.png', row['label'],
                                          record['geometry'], config['alignment'])
            if max(abs(rescored[name][metric] - cached[name][metric]) for metric in METRICS) > freeze['replay_tolerance']:
                raise ValueError('Original metric cache replay exceeds frozen tolerance')
        scores, artifacts, observables, decision = [], [], [], None
        if records['surface']['status'] == 'success':
            record = records['surface']
            trace = record['diagnostics']['surface_iterations'][-1]
            decision = json.loads((folders['surface'] / 'surface' /
                                  f'iteration-{trace["iteration"]:02d}' / 'decision.json').read_text())
            if decision['index'] != trace['selected'] or len(decision['observable_scores']) != 8:
                raise ValueError('Final decision inventory differs')
            observables = decision['observable_scores']
            for index in range(8):
                path, artifact = derive_candidate(folders['surface'], destination, record, decision, index)
                artifacts.append(artifact)
                scores.append(models.score(references['surface'], path, row['label'], record['geometry'], config['alignment']))
        write_json(result_path, dict(id=row['id'], source=row['source'],
            image_identity=row['source'] + '/' + row['original_id'], inputs=inputs,
            baseline_status=records['baseline']['status'], surface_status=records['surface']['status'],
            controls=rescored, cached_controls=cached, candidate_scores=scores,
            candidate_artifacts=artifacts, observable_scores=observables,
            selected_index=decision['index'] if decision else None,
            decision_sha256=records['surface']['diagnostics']['surface_iterations'][-1]['decision_sha256'] if decision else None))
        print('MEASURED', row['id'], len(scores), flush=True)
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / f'runtime-worker-{args.worker_index}.json', runtime)


def summarize(args, freeze, official):
    expected = [row['id'] for row in official]
    paths = sorted(args.output_dir.glob('targets/*/*/pool.json'))
    records = {row['id']: row for row in (json.loads(path.read_text()) for path in paths)}
    if set(records) != set(expected) or len(records) != len(paths):
        raise ValueError('Incomplete or duplicate diagnostic target inventory')
    runtimes = [json.loads((args.output_dir / f'runtime-worker-{i}.json').read_text()) for i in range(4)]
    for runtime in runtimes:
        if not runtime.get('finished_at') or runtime['freeze_sha256'] != sha256(args.freeze):
            raise ValueError('Incomplete worker or freeze drift')
        if runtime['environment']['packages'] != runtimes[0]['environment']['packages']:
            raise ValueError('Metric worker environments differ')
    paired, choices, successes, drift = [], {}, 0, {metric: 0. for metric in METRICS}
    for identity in expected:
        record = records[identity]
        if record['inputs']['freeze_sha256'] != sha256(args.freeze):
            raise ValueError('Diagnostic freeze changed')
        for side in ('baseline', 'surface'):
            for metric in METRICS:
                if record['controls'][side][metric] is not None:
                    drift[metric] = max(drift[metric], abs(record['controls'][side][metric] - record['cached_controls'][side][metric]))
        if record['surface_status'] == 'success':
            scores, choice = policies(record['candidate_scores'], record['observable_scores'],
                                     record['selected_index'], record['controls']['surface'])
            choices[identity] = choice
            successes += 1
            for index, artifact in enumerate(record['candidate_artifacts']):
                if sha256(args.output_dir / 'targets' / identity / f'candidate-{index}.png') != artifact['derived_sha256']:
                    raise ValueError('Derived candidate bytes changed')
        else:
            if record['candidate_scores'] or record['candidate_artifacts']:
                raise ValueError('Failure gained candidates')
            scores = {name: failure_metrics() for name in (*POLICIES, *ORACLES)}
        paired.append(dict(id=identity, source=record['source'], image_identity=record['image_identity'],
                           scores=dict(baseline=record['controls']['baseline'], **scores)))
    if max(drift.values()) > freeze['replay_tolerance']:
        raise ValueError('Replay drift exceeds tolerance')
    contrasts = [('baseline', policy) for policy in (*POLICIES, *ORACLES)]
    contrasts += [(control, 'frozen_raw') for control in ('uniform_raw', 'uniform_valid', 'first_raw', 'semantic_only')]
    contrasts += [('frozen_raw', 'corrected')]
    comparisons = {}
    for control, method in contrasts:
        metrics = {}
        for metric in METRICS:
            keep = [dict(id=row['id'], source=row['source'], image_identity=row['image_identity'],
                         baseline=row['scores'][control], surface=row['scores'][method]) for row in paired
                    if row['scores'][control][metric] is not None and row['scores'][method][metric] is not None]
            metrics[metric] = image_cluster_interval(keep, metric) if keep else dict(n=0, relative_gain_percent=None)
            metrics[metric]['excluded_pairs'] = len(paired) - len(keep)
        comparisons[method + '_vs_' + control] = metrics
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), targets=len(paired),
        saved_pools=successes, scored_candidates=8 * successes, freeze_sha256=sha256(args.freeze),
        cache_replay_max_absolute_drift=drift, contrasts=comparisons, scope=freeze['scope'],
        oracle_warning='Different reference-informed choices per metric; not deployable or jointly attainable.')
    write_json(args.output_dir / 'analysis/summary.json', result)
    write_json(args.output_dir / 'analysis/paired_records.json', paired)
    write_json(args.output_dir / 'analysis/choices.json', choices)
    print(json.dumps(dict(targets=len(paired), saved_pools=successes, replay_drift=drift), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=4)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    a.freeze, a.output_dir = a.freeze.resolve(), a.output_dir.resolve()
    freeze = json.loads(a.freeze.read_text())
    for name, expected in freeze['files'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Diagnostic source/input drift: ' + name)
    rows, _ = read_official(ROOT / freeze['dataset_root'])
    rows = select_manifest_rows(rows, ROOT / freeze['target_manifest'])
    if a.worker_count != 4 or not 0 <= a.worker_index < 4:
        raise ValueError('Exactly four allocated workers required')
    if a.preflight:
        for side in ('baseline', 'surface'):
            run = ROOT / freeze[side + '_run']
            verify_run(run, ROOT / freeze['dataset_root'], require_provenance=True)
            execution = json.loads((run / 'execution.json').read_text())
            if execution['target_ids'] != [row['id'] for row in rows]:
                raise ValueError('Snapshot inventory does not match frozen cohort')
        print('PASS: annotated cohort diagnostic preflight')
    else:
        (summarize if a.summarize else measure)(a, freeze, rows)


if __name__ == '__main__':
    main()
