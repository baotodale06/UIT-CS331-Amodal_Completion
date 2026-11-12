#!/usr/bin/env python3
"""Evaluate saved final-pool RGB headroom. Oracle selections are diagnostics only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.references import reference_paths
from amodal_benchmark.protocol import failure_metrics
from audit_surface_lastpass_result import paired_estimate, METRICS


def derive_candidate(source, destination, record, decision, index):
    """Use upstream serialization and the existing final alpha for every candidate."""
    trace = record['diagnostics']['surface_iterations'][-1]
    folder = source/'surface'/f'iteration-{trace["iteration"]:02d}'
    if sha256(folder/'decision.json') != trace['decision_sha256']:
        raise ValueError('Final pool decision changed')
    if json.loads((folder/'decision.json').read_text()) != decision:
        raise ValueError('Provided decision differs')
    candidate = folder/f'candidate-{index}.png'
    if sha256(candidate) != decision['artifacts'][candidate.name]:
        raise ValueError('Candidate bytes changed')
    if sha256(source/'completion.png') != record['completion_sha256']:
        raise ValueError('Final completion changed')
    with Image.open(source/'completion.png') as image:
        completed = image.convert('RGBA')
    with Image.open(source/'mask.png') as image:
        mask = image.convert('L')
    if not np.array_equal(np.asarray(completed)[..., 3], np.asarray(mask)):
        raise ValueError('Final alpha changed')
    r0, r1, c0, c1 = decision['crop_rc']
    if completed.size != (c1-c0, r1-r0):
        raise ValueError('Pool crop differs from completion frame')
    with Image.open(candidate) as image:
        rgb = image.convert('RGB').resize(completed.size, Image.Resampling.BICUBIC)
    destination.mkdir(parents=True, exist_ok=True)
    jpeg = destination/f'candidate-{index}.jpg'
    rgb.save(jpeg, quality=90)
    with Image.open(jpeg) as image:
        output = image.convert('RGBA')
    output.putalpha(mask)
    path = destination/f'candidate-{index}.png'
    output.save(path)
    return path, dict(candidate_sha256=sha256(candidate), derived_sha256=sha256(path),
                      fixed_mask_sha256=sha256(source/'mask.png'))


def pool_policies(candidate_scores, observable_scores, selected, corrected):
    """Observable controls and separate per-metric optimistic RGB ceilings."""
    if len(candidate_scores) != 8 or len(observable_scores) != 8:
        raise ValueError('Exactly eight candidates required')
    if any(set(row) != set(METRICS) or
           not all(isinstance(v, (int, float)) and np.isfinite(v) for v in row.values())
           for row in candidate_scores):
        raise ValueError('Missing or nonfinite candidate scores')
    valid = [i for i, row in enumerate(observable_scores) if row['valid']]
    semantic = max(valid, key=lambda i: observable_scores[i]['candidate_text_cosine']) if valid else 0
    outputs = dict(frozen_raw=candidate_scores[selected], semantic_only=candidate_scores[semantic],
        uniform_raw={metric:float(np.mean([row[metric] for row in candidate_scores])) for metric in METRICS},
        corrected=corrected)
    choices = dict(frozen_raw=selected, semantic_only=semantic)
    # Each oracle chooses a different candidate using evaluation-reference scores.
    # No hybrid score vector is presented as one achievable output.
    for metric in METRICS:
        sign = -1 if metric == 'LPIPS' else 1
        index = max(range(8), key=lambda i: sign*candidate_scores[i][metric])
        key = 'oracle_'+metric
        outputs[key] = candidate_scores[index]
        choices[key] = index
    return outputs, choices


def measure(args, freeze):
    import torch
    from amodal_benchmark.metrics import MetricModels
    from amodal_benchmark.provenance import environment_snapshot
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows, _ = read_official(ROOT/freeze['dataset_root'])
    if [row['id'] for row in rows] != freeze['target_ids']:
        raise ValueError('Target inventory differs')
    surface, ablation = ROOT/freeze['surface_run'], ROOT/freeze['ablation_run']
    config = json.loads((surface/'metric_config.json').read_text())
    if sha256(surface/'metric_config.json') != freeze['metric_config_sha256']:
        raise ValueError('Metric operations differ')
    runtime = dict(worker=args.worker_index, started_at=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha256(args.freeze), job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(0), environment=environment_snapshot())
    write_json(args.output_dir/f'runtime-worker-{args.worker_index}.json', runtime)
    models = MetricModels()
    for item in rows[args.worker_index::args.worker_count]:
        identity = item['id']
        prior_path = ablation/'targets'/identity/'ablation.json'
        if sha256(prior_path) != freeze['ablation_record_sha256'][identity]:
            raise ValueError('Parent ablation changed: '+identity)
        prior = json.loads(prior_path.read_text())
        source = surface/'targets'/identity
        if sha256(source/'result.json') != prior['inputs']['surface_result_sha256']:
            raise ValueError('Completed target changed')
        record = json.loads((source/'result.json').read_text())
        destination = args.output_dir/'targets'/identity
        path = destination/'headroom.json'
        inputs = dict(ablation_sha256=sha256(prior_path), freeze_sha256=sha256(args.freeze))
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['inputs'] != inputs:
                raise ValueError('Headroom resume inputs changed')
            for index, artifact in enumerate(existing['artifacts']):
                if sha256(destination/f'candidate-{index}.png') != artifact['derived_sha256']:
                    raise ValueError('Derived candidate changed on resume')
            continue
        scores, artifacts, observables, decision = [], [], [], None
        if record['status'] == 'success':
            trace = record['diagnostics']['surface_iterations'][-1]
            decision = json.loads((source/'surface'/f'iteration-{trace["iteration"]:02d}'/'decision.json').read_text())
            if decision['index'] != trace['selected'] or len(decision['observable_scores']) != 8:
                raise ValueError('Final pool inventory changed')
            observables = decision['observable_scores']
            reference = reference_paths(surface/'visible_references', item)
            key = 'crop' if config['alignment']=='released' else 'full'
            if sha256(reference[key]) != record['evaluation_reference_artifacts'][key]:
                raise ValueError('Reference bytes changed')
            for index in range(8):
                image, artifact = derive_candidate(source, destination, record, decision, index)
                artifacts.append(artifact)
                scores.append(models.score(reference[key], image, item['label'], record['geometry'], config['alignment']))
            selected = decision['index']
            if artifacts[selected]['derived_sha256'] != prior['derivation']['derived_sha256']:
                raise ValueError('Selected raw RGB does not reproduce the audited ablation')
            if max(abs(scores[selected][m]-prior['scores']['raw_final'][m]) for m in METRICS) > 1e-5:
                raise ValueError('Selected metric replay drift exceeds the frozen tolerance')
        write_json(path, dict(id=identity, source=item['source'], image_identity=prior['image_identity'],
            inputs=inputs, baseline_status=prior['baseline_status'], surface_status=prior['surface_status'],
            controls=prior['scores'], candidate_scores=scores, artifacts=artifacts,
            observable_scores=observables, selected_index=decision['index'] if decision else None))
        print('MEASURED', identity, len(scores), flush=True)
    runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir/f'runtime-worker-{args.worker_index}.json', runtime)


def summarize(args, freeze):
    paths = list(args.output_dir.glob('targets/*/*/headroom.json'))
    rows = {r['id']: r for r in (json.loads(p.read_text()) for p in paths)}
    if len(rows) != len(paths) or set(rows) != set(freeze['target_ids']):
        raise ValueError('Incomplete final-pool inventory')
    runtimes = [json.loads((args.output_dir/f'runtime-worker-{i}.json').read_text()) for i in range(4)]
    for runtime in runtimes:
        if not runtime.get('finished_at') or runtime['freeze_sha256'] != sha256(args.freeze):
            raise ValueError('Unfinished or changed runtime')
        if runtime['environment']['packages'] != runtimes[0]['environment']['packages']:
            raise ValueError('Worker packages differ')
    policies = ['frozen_raw', 'semantic_only', 'uniform_raw', 'corrected']+['oracle_'+m for m in METRICS]
    pairs, choices, successes, drift = [], {}, 0, {m:0.0 for m in METRICS}
    for identity in freeze['target_ids']:
        row = rows[identity]
        if row['inputs'] != dict(freeze_sha256=sha256(args.freeze), ablation_sha256=freeze['ablation_record_sha256'][identity]):
            raise ValueError('Headroom record inputs differ')
        if row['surface_status']=='success':
            scores, decision = pool_policies(row['candidate_scores'], row['observable_scores'], row['selected_index'], row['controls']['surface'])
            choices[identity] = decision
            successes += 1
            for index, artifact in enumerate(row['artifacts']):
                if sha256(args.output_dir/'targets'/identity/f'candidate-{index}.png') != artifact['derived_sha256']:
                    raise ValueError('Saved RGB derivation differs')
            for metric in METRICS:
                drift[metric] = max(drift[metric], abs(scores['frozen_raw'][metric]-row['controls']['raw_final'][metric]))
        else:
            if row['candidate_scores'] or row['artifacts']:
                raise ValueError('Failed target gained candidates')
            scores = {name:failure_metrics() for name in policies}
        pairs.append(dict(id=identity, source=row['source'], image_identity=row['image_identity'],
                          scores=dict(baseline=row['controls']['baseline'], **scores)))
    if successes != 958 or max(drift.values()) > 1e-5:
        raise ValueError('Inventory or metric replay differs')
    comparisons = {policy:{metric:paired_estimate(pairs, 'baseline', policy, metric) for metric in METRICS}
                   for policy in policies}
    ceilings = {metric:comparisons['oracle_'+metric][metric] for metric in METRICS}
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), scope=freeze['scope'],
        targets=len(rows), saved_pools=successes, scored_candidates=successes*8,
        contrasts=comparisons, independent_single_metric_oracle_ceilings=ceilings,
        any_single_metric_raw_ceiling_below3=any(value['relative_gain_percent']<3 for value in ceilings.values()),
        selected_score_max_absolute_drift=drift, freeze_sha256=sha256(args.freeze),
        warning='Oracle choices use evaluation references, are different for each metric, and are not a deployable method or joint attainable result.')
    analysis = args.output_dir/'analysis'
    write_json(analysis/'summary.json', result)
    write_json(analysis/'oracle_and_control_choices.json', choices)
    lines=['# Saved final-pool RGB headroom', '', freeze['scope'], '', result['warning'], '',
        f'All {len(rows)} targets retained; {successes} pools and {successes*8} scored RGB candidates.', '',
        '| Policy | CLIP gain | LPIPS gain | VGG gain | SSIM gain |', '| --- | ---: | ---: | ---: | ---: |']
    for policy, metrics in comparisons.items():
        lines.append('| '+policy+' | '+' | '.join(f'{metrics[m]["relative_gain_percent"]:+.3f}%' for m in METRICS)+' |')
    lines += ['', 'CLIP uses 946 shared successful targets; other metrics retain all 2,565 targets and original failure penalties. '
        'Fixed masks, geometry and prior trajectories mean these bounds apply only to saved final RGB selection. '
        'Uniform control samples all eight raw candidates, including candidates the original validity rule would exclude. '
        'Semantic-only control uses the recorded valid-candidate rule and observation-only text cosine. '
        'No hidden-RGB MAE ground truth exists here. No ranker was fitted or diffusion rerun.', '',
        'Full paired intervals and all cross-metric tradeoffs are in summary.json.', '']
    (analysis/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    design = args.freeze.parent
    files = [*paths, *args.output_dir.glob('runtime-worker-*.json'), *analysis.glob('*'), args.freeze]
    files += [ROOT/name for name in freeze['source_sha256']]
    files = sorted(set(files))
    manifest = {p.relative_to(ROOT).as_posix():sha256(p) for p in files}
    archive = design/'evidence.tar.gz'
    if archive.exists():
        raise ValueError('Preserve existing headroom archive')
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in files:
            bundle.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    write_json(design/'evidence-manifest.json', manifest)
    write_json(design/'evidence-receipt.json', dict(sha256=sha256(archive), bytes=archive.stat().st_size,
                                                   members=len(manifest), freeze_sha256=sha256(args.freeze)))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=4)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    args.freeze, args.output_dir = args.freeze.resolve(), args.output_dir.resolve()
    freeze = json.loads(args.freeze.read_text())
    for name, expected in freeze['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Frozen headroom source changed: '+name)
    if args.worker_count != 4 or not 0 <= args.worker_index < 4:
        raise ValueError('Four workers required')
    (summarize if args.summarize else measure)(args, freeze)


if __name__ == '__main__':
    main()
