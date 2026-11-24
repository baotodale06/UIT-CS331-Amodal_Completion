#!/usr/bin/env python3
"""Independently review completed annotated metadata without changing the study."""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.protocol import METRICS, failure_metrics
from amodal_benchmark.provenance import digest, verify_provenance
from amodal_benchmark.surface_refinement import predict_rank
from amodal_benchmark.surface_runtime import progressive_seed
from audit_surface_progressive_result import estimate

DESIGN = ROOT/'output/surface-annotated-20261112-design'
STUDY = ROOT/'output/surface-annotated-20261112-freshrefs'
OUTPUT = ROOT/'output/surface-annotated-full-review-20261114'
RECOVERY = ROOT/'output/surface-annotated-20261112-recovery'/f'{STUDY.name}-full-metadata'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare_estimates(actual, expected):
    for metric in METRICS:
        for key in ('n', 'images', 'baseline', 'method', 'absolute_gain',
                    'relative_gain_percent', 'absolute_ci', 'relative_ci_percent'):
            np.testing.assert_allclose(actual[metric][key], expected[metric][key],
                                       atol=1e-10, rtol=1e-10, err_msg=metric+'/'+key)


def main():
    receipt = read(RECOVERY/'recovery-receipt.json')
    remote_receipt = read(RECOVERY/'remote-receipt.json')
    manifest = read(RECOVERY/'remote-manifest.json')
    require(receipt['verified'], 'Recovery is incomplete')
    require(receipt['member_sha256'] == manifest, 'Recovery manifest differs')
    require(sha256(RECOVERY/'evidence.tar.gz') == receipt['archive_sha256'] ==
            remote_receipt['archive_sha256'], 'Archive checksum mismatch')
    require(sha256(RECOVERY/'remote-manifest.json') == remote_receipt['manifest_sha256'],
            'Manifest checksum mismatch')
    for relative, expected in manifest.items():
        require(sha256(ROOT/relative) == expected, 'Recovered artifact changed: '+relative)
    print('PASS: all recovered metadata hashes', flush=True)

    comparison = read(STUDY/'full-comparison/comparison.json')
    remote_audit = read(STUDY/'full-comparison/independent-audit.json')
    require(remote_audit['passed'], 'Remote independent audit failed')
    require(sha256(STUDY/'full-comparison/comparison.json') ==
            remote_receipt['completion_gate_sha256'], 'Export gate changed')
    freeze = read(DESIGN/'method-freeze.json')
    parent_path = ROOT/'output/surface-followup-20261110-design/final-method-freeze.json'
    require(sha256(parent_path) == freeze['parent_freeze_sha256'], 'Parent method changed')
    parent = read(parent_path)
    for key in ('ranker', 'candidate_seeds', 'refinement', 'ensemble', 'pretrained_visible_features'):
        require(freeze[key] == parent[key], 'Transferred method parameters changed: '+key)
    runs = {side: STUDY/side for side in ('baseline', 'surface')}
    executions = {side: verify_provenance(run) for side, run in runs.items()}
    configs = {side: read(run/'run_config.json') for side, run in runs.items()}
    policy = configs['surface'].pop('surface_policy')
    require(configs['baseline'] == configs['surface'], 'Unmatched inference contracts')
    require(configs['baseline']['detector_vocabulary']['mode'] == 'annotated', 'Wrong baseline')
    require(policy['freeze_sha256'] == comparison['freeze_sha256'] == sha256(DESIGN/'method-freeze.json'),
            'Wrong frozen method')
    require(read(runs['baseline']/'metric_config.json') == read(runs['surface']/'metric_config.json'),
            'Unmatched metric operations')
    environments = {side: read(run/'provenance'/executions[side]['id']/'environment.json')
                    for side, run in runs.items()}
    require(environments['baseline']['packages'] == environments['surface']['packages'],
            'Installed packages differ')
    verification = {side: read(STUDY/'full-comparison'/f'{side}-verification.json') for side in runs}
    for side, report in verification.items():
        require(report['verified'] and report['provenance_verified'] and report['full_evaluation'],
                'Missing remote full verification')
        require(sha256(runs[side]/'execution.json') == comparison[side+'_execution_sha256'],
                'Comparison execution changed')
    require(verification['baseline']['data_identity'] == verification['surface']['data_identity'],
            'Paired data identities differ')
    print('PASS: matched contracts and execution provenance', flush=True)

    official, _ = read_official(ROOT/'datasets/open_world_amodal')
    expected_ids = [row['id'] for row in official]
    for execution in executions.values():
        require(execution['target_ids'] == expected_ids, 'Full target inventory changed')
    pairs, decisions, preserved_cohort = [], [], 0
    status_counts = {side: Counter() for side in runs}
    reason_counts = {side: Counter() for side in runs}
    historical_checks = 0
    for side, run in runs.items():
        for snapshot in (run/'provenance').glob('*/execution.json'):
            for identity, expected in read(snapshot).get('result_artifacts', {}).items():
                require(sha256(run/'targets'/identity/'result.json') == expected,
                        'A completed historical result was changed')
                historical_checks += 1
    for row in official:
        records, metrics = {}, {}
        for side, run in runs.items():
            target = run/'targets'/row['id']
            record = records[side] = read(target/'result.json')
            require(record['id'] == row['id'], 'Wrong target identity')
            status_counts[side][record['status']] += 1
            if record['status'] == 'success':
                cache = read(target/'metrics.json')
                require(cache['inputs']['completion_sha256'] == record['completion_sha256'],
                        'Metric completion identity mismatch')
                require(cache['inputs']['reference_sha256'] in record['evaluation_reference_artifacts'].values(),
                        'Metric reference identity mismatch')
                metrics[side] = cache['metrics']
            else:
                reason_counts[side][record.get('failure_reason', record['status'])] += 1
                metrics[side] = failure_metrics()
            cohort = STUDY/(side+'-cohort')/'targets'/row['id']/'result.json'
            if cohort.exists():
                require(sha256(cohort) == sha256(target/'result.json'), 'Frozen cohort result changed')
                preserved_cohort += 1
        for key in ('input_sha256', 'evaluation_reference_artifacts', 'seed'):
            require(records['baseline'][key] == records['surface'][key], 'Unmatched target '+row['id'])
        first, second = (records[side]['status'] == 'success' for side in runs)
        transition = 'shared_success' if first and second else 'gained' if second else 'lost' if first else 'both_failed'
        pair = dict(id=row['id'], source=row['source'], label=row['label'],
                    image_identity=row['source']+'/'+row['original_id'], transition=transition, **metrics)
        for side, record in records.items():
            pair.update({side+'_status': record['status'], side+'_reason': record.get('failure_reason'),
                         side+'_iterations': record.get('iterations'),
                         side+'_capped': record.get('iteration_limit_reached', False)})
        traces = records['surface'].get('diagnostics', {}).get('surface_iterations', [])
        pair['surface_passes'] = len(traces)
        for trace in traces:
            path = runs['surface']/'targets'/row['id']/'surface'/f'iteration-{trace["iteration"]:02d}'/'decision.json'
            require(sha256(path) == trace['decision_sha256'], 'Decision hash differs')
            decision = read(path)
            require(decision['inference_truth_access'] is False, 'Truth accessed during inference')
            require(decision['target_seed'] == records['surface']['seed'], 'Target seed differs')
            expected_seeds = [progressive_seed(decision['target_seed'], trace['iteration'], seed)
                              for seed in freeze['candidate_seeds']]
            require(decision['candidate_seeds'] == expected_seeds, 'Candidate seeds differ')
            selected, losses = predict_rank(decision['observable_scores'], freeze['ranker'])
            require(selected == trace['selected'] == decision['index'], 'Frozen selector does not replay')
            require(len(losses) == len(decision['losses']) == comparison['candidate_budget'], 'Wrong candidate count')
            for actual, expected in zip(losses, decision['losses']):
                require((actual is None and expected is None) or
                        (actual is not None and expected is not None and abs(actual-expected) < 1e-10),
                        'Selector loss does not replay')
            decisions.append(dict(id=row['id'], iteration=trace['iteration'], selected=selected,
                                  fallback=decision['fallback'], sha256=trace['decision_sha256']))
        pairs.append(pair)
    require(len(decisions) == len(list(runs['surface'].glob('targets/*/*/surface/*/decision.json'))),
            'Unaccounted candidate pools remain')
    require(preserved_cohort == 408, 'Frozen cohort preservation is incomplete')
    print('PASS: target pairs, preserved cohort and all selector decisions', flush=True)

    metrics = {metric: estimate(pairs, metric) for metric in METRICS}
    compare_estimates(metrics, comparison['metrics'])
    compare_estimates(metrics, remote_audit['metrics'])
    transitions = Counter(row['baseline_status']+' -> '+row['surface_status'] for row in pairs)
    require(dict(transitions) == comparison['status_transitions'] == remote_audit['transitions'],
            'Failure transitions differ')
    require({side:dict(counts) for side, counts in status_counts.items()} == remote_audit['status_counts'],
            'Status accounting differs')
    coverage = Counter(row['transition'] for row in pairs)
    for actual, field in (('shared_success', 'shared_successes'), ('gained', 'gained_successes'), ('lost', 'lost_successes')):
        require(coverage[actual] == comparison[field], 'Coverage differs')
    unseen_inventory = read(DESIGN/'unseen-inventory.json')
    unseen_ids = set(unseen_inventory['target_ids'])
    require(len(unseen_ids) == len(unseen_inventory['target_ids']) and unseen_ids <= set(expected_ids),
            'Invalid unseen inventory')
    require(sha256(DESIGN/'unseen-inventory.json') == remote_audit['unseen_inventory_sha256'],
            'Unseen inventory changed')
    unseen = [row for row in pairs if row['id'] in unseen_ids]
    unseen_metrics = {metric: estimate(unseen, metric) for metric in METRICS}
    compare_estimates(unseen_metrics, remote_audit['unseen_metrics'])
    per_source = {source:{metric:estimate([row for row in pairs if row['source'] == source], metric)
                          for metric in METRICS} for source in sorted({row['source'] for row in pairs})}
    shared = [row for row in pairs if row['transition'] == 'shared_success']
    contributions = {group:sum(row['baseline']['LPIPS']-row['surface']['LPIPS']
                              for row in pairs if row['transition'] == group)/len(pairs) for group in coverage}
    np.testing.assert_allclose(sum(contributions.values()), metrics['LPIPS']['absolute_gain'], atol=1e-12)
    for source in (*per_source, 'Overall'):
        selected = pairs if source == 'Overall' else [row for row in pairs if row['source'] == source]
        for side in runs:
            reported = verification[side]['statistics'][source]
            require(reported['targets'] == len(selected), 'Source target denominator differs')
            require(reported['successes'] == sum(row[side+'_status'] == 'success' for row in selected),
                    'Source saved-output denominator differs')
            for metric in METRICS:
                values = [row[side][metric] for row in selected if row[side][metric] is not None]
                require(reported[metric+'_n'] == len(values), 'Source metric denominator differs')
                np.testing.assert_allclose(np.mean(values), reported[metric], atol=1e-12, rtol=1e-12)
    print('PASS: full, unseen and source statistics; all confidence intervals', flush=True)

    extra = OUTPUT/'remote-evidence'
    extra_receipt = read(extra/'receipt.json')
    for relative, record in extra_receipt['members'].items():
        require(sha256(extra/relative) == record['sha256'], 'Supplementary evidence changed')
    registry_path = extra/'output/surface-annotated-20261112-design/freshrefs-experiments.csv'
    with registry_path.open(newline='', encoding='utf-8') as stream:
        registry = list(csv.DictReader(stream))
    require(len(registry) == 2, 'Expected exactly two completed registry entries')
    for side, run in runs.items():
        entry, = [item for item in registry if item['run_dir'].endswith('/'+STUDY.name+'/'+side)]
        require(entry['status'] == 'complete' and int(entry['targets']) == len(pairs), 'Incomplete registry')
        require(entry['summary_sha256'] == sha256(run/'summary.json'), 'Registry summary mismatch')
        require(entry['entry_id'] == digest({'run':entry['run_dir'], 'execution':executions[side]['id']}),
                'Registry execution mismatch')
        identity = verification[side]['data_identity']
        require(entry['protocol_sha256'] == digest(dict(inference=read(run/'run_config.json'),
            metrics=read(run/'metric_config.json'), data=identity)), 'Registry protocol mismatch')
        for key in ('inputs_sha256', 'references_sha256'):
            require(entry[key] == identity[key], 'Registry identity mismatch')
        for key in ('targets', 'successes', 'failures', 'failure_rate', *METRICS):
            np.testing.assert_allclose(float(entry[key]), verification[side]['statistics']['Overall'][key], atol=1e-12)
    jobs = {}
    fields = ('id', 'name', 'partition', 'state', 'seconds', 'start', 'end', 'exit_code', 'allocated', 'nodes')
    for line in extra_receipt['accounting'].splitlines():
        values = line.strip().split('|')
        record = dict(zip(fields, values))
        record['seconds'] = int(record['seconds'])
        jobs[record['id']] = record
    require(jobs['39490']['state'] == 'TIMEOUT', 'Unexpected original continuation state')
    for identifier in ('39486','39487','39488','39491','39492','39495','39497'):
        require(jobs[identifier]['state'] == 'COMPLETED' and jobs[identifier]['exit_code'] == '0:0',
                'Unsuccessful required job '+identifier)
    compute = {}
    for side, ids in (('baseline', ['39486','39488']), ('surface', ['39487','39490','39491'])):
        require(all('gres/gpu=4' in jobs[job]['allocated'] for job in ids), 'Expected four allocated GPUs')
        seconds = sum(jobs[job]['seconds'] for job in ids)
        compute[side] = dict(jobs=ids, allocation_seconds=seconds, reserved_gpu_hours=seconds*4/3600)
    compute['allocation_ratio'] = compute['surface']['allocation_seconds']/compute['baseline']['allocation_seconds']
    compute['scope'] = 'Cohort plus full continuation, including timeout; excludes smoke, failed reference attempts, pool diagnostic, CPU audits and idle queue time.'
    result = dict(passed=True, at=datetime.now(timezone.utc).isoformat(), targets=len(pairs),
        images=len({row['image_identity'] for row in pairs}), recovered_metadata_members=len(manifest),
        archive_sha256=receipt['archive_sha256'], source_sha256=sha256(Path(__file__)),
        estimate_source_sha256=sha256(ROOT/'scripts/audit_surface_progressive_result.py'),
        metrics=metrics, unseen_metrics=unseen_metrics, unseen_targets=len(unseen),
        unseen_images=len({row['image_identity'] for row in unseen}), per_source=per_source,
        shared_success_metrics={metric:estimate(shared, metric) for metric in METRICS},
        status_counts={side:dict(counts) for side,counts in status_counts.items()},
        failure_reasons={side:dict(counts) for side,counts in reason_counts.items()},
        transitions=dict(transitions), coverage=dict(coverage), lpips_gain_contributions=contributions,
        capped_successes={side:sum(row[side+'_capped'] and row[side+'_status']=='success' for row in pairs)
                          for side in runs}, preserved_cohort_result_hashes=preserved_cohort,
        historical_result_hash_checks=historical_checks, decisions_replayed=len(decisions),
        recorded_candidates=len(decisions)*comparison['candidate_budget'],
        fallback_decisions=sum(row['fallback'] for row in decisions),
        surface_pass_histogram=dict(Counter(row['surface_passes'] for row in pairs)),
        no_sampling_reasons=dict(Counter(row['surface_reason'] or row['surface_status']
                                        for row in pairs if not row['surface_passes'])),
        compute=compute, registry_entries=registry, jobs=jobs,
        full_metric_point_target_met=all(value['relative_gain_percent'] >= 3 for value in metrics.values()),
        method_parameters_unchanged=True, local_full_native_verification=False,
        local_neural_metrics_recomputed=False, local_feature_extraction_recomputed=False,
        local_full_rgb_correction_replay=False, numpy_version=np.__version__,
        scope='Independent metadata, provenance, registry, stored-feature selector, arithmetic and bootstrap audit. '
              'Native full-run artifacts were verified on the cluster; full PNGs remain there. '
              'Primary cohort and its native replay remain separate. New decomposition/source analyses are post-hoc descriptive.')
    write_json(OUTPUT/'audit.json', result)
    write_json(OUTPUT/'paired-records.json', pairs)
    write_json(OUTPUT/'decision-replay.json', decisions)
    print(json.dumps(dict(passed=True, targets=len(pairs), decisions=len(decisions),
                          metrics=metrics, coverage=coverage, compute=compute), indent=2))


if __name__ == '__main__':
    main()
