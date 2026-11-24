"""Read-only quality diagnostic on the pre-existing 193-target engineering snapshot.

This runtime-selected VG prefix is not representative full-benchmark evidence.
It cannot satisfy a benchmark gate or change the frozen inference policy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    sys.path.insert(0, str(study/'postprocessing-source'))
    from amodal_benchmark.dataset import sha256, write_json
    from amodal_benchmark.metrics import MetricModels
    from amodal_benchmark.protocol import failure_metrics
    from amodal_benchmark.references import reference_paths
    from scripts.assess_shape_v2 import assess, table
    from scripts.audit_shape_v2 import paired_estimate

    frozen = json.loads((study/'postprocessing-source-manifest.json').read_bytes())
    for name, expected in frozen['files'].items():
        if sha256(study/'postprocessing-source'/name) != expected:
            raise ValueError('Frozen scoring/audit source changed: '+name)
    audit_path = study/'partial-audit-45665.json'
    audit = json.loads(audit_path.read_bytes())
    if not audit['completed_target_snapshot_verified'] or audit['targets'] != 193:
        raise ValueError('Require the existing engineering snapshot, with every terminal status')
    if sha256(study/'policy-freeze.json') != audit['freeze_sha256']:
        raise ValueError('Frozen policy changed')
    control = study.parent/'surface-annotated-20261112-freshrefs/baseline'
    if sha256(control/'per_target.csv') != 'ceb0da09942d105e034efe1948054161120959f3be76046111083fbe5a460a1b':
        raise ValueError('Registered original control changed')
    registered = table(control/'per_target.csv')
    metric_config = json.loads((control/'metric_config.json').read_bytes())
    if metric_config['alignment'] != 'released':
        raise ValueError('Unexpected scoring alignment')
    output = study/'snapshot-quality-45665'
    if output.exists():
        raise FileExistsError('Preserve previous snapshot scores')
    output.mkdir()
    # Inventory is fixed before looking at any quality scores. Completed failure
    # records are included; no success-intersection filtering is allowed.
    selected = []
    for tid, expected in audit['result_hashes'].items():
        path = study/'benchmark/targets'/tid/'result.json'
        if sha256(path) != expected:
            raise ValueError('Previously audited target changed: '+tid)
        record = json.loads(path.read_bytes())
        old = json.loads((control/'targets'/tid/'result.json').read_bytes())
        for key in ('id', 'seed', 'input_sha256', 'evaluation_reference_artifacts'):
            if record[key] != old[key]:
                raise ValueError('Control and method input mismatch: '+tid+'/'+key)
        paths = reference_paths(study/'benchmark/visible_references', record)
        for name in ('full', 'crop', 'mask'):
            if sha256(paths[name]) != record['evaluation_reference_artifacts'][name]:
                raise ValueError('Original reference changed: '+tid)
        if record['status'] not in ('success', 'not_detected', 'ambiguous_target', 'no_completion'):
            raise ValueError('Review unexpected terminal status: '+tid)
        selected.append((record, old, paths['crop']))
    import torch
    if torch.cuda.device_count() != 1 or 'QUADRO' not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError('Use one allocated Quadro for the diagnostic')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), job_id=os.environ.get('SLURM_JOB_ID'),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        metric_config=metric_config, audit_sha256=sha256(audit_path), freeze_sha256=audit['freeze_sha256'],
        source_sha256=sha256(Path(__file__)), metric_source_sha256=sha256(study/'postprocessing-source/amodal_benchmark/metrics.py'),
        control_csv_sha256=sha256(control/'per_target.csv'), target_ids=[row[0]['id'] for row in selected],
        full_benchmark_verified=False, benchmark_success=False, policy_changed=False)
    write_json(output/'contract.json', runtime)
    model = MetricModels()
    rows = []
    for record, old, reference in selected:
        tid = record['id']
        values = {}
        for side, root, saved in (('method', study/'benchmark', record), ('rescored_control', control, old)):
            if saved['status'] == 'success':
                completion = root/'targets'/tid/'completion.png'
                if sha256(completion) != saved['completion_sha256']:
                    raise ValueError('Completion changed: '+side+'/'+tid)
                values[side] = model.score(reference, completion, saved['label'], saved['geometry'], 'released')
            else:
                values[side] = failure_metrics()
        row = dict(id=tid, source=record['source'], image_identity=record['image'],
            method_status=record['status'], control_status=old['status'],
            method_record_sha256=audit['result_hashes'][tid],
            control_record_sha256=sha256(control/'targets'/tid/'result.json'),
            reference_sha256=sha256(reference), control={key:registered[tid][key] for key in values['method']},
            method_completion_sha256=record.get('completion_sha256'),
            control_completion_sha256=old.get('completion_sha256'), **values)
        write_json(output/'targets'/tid/'metrics.json', row)
        rows.append(row)
        print(f"Scored {len(rows)}/{len(selected)} {tid}", flush=True)
    control_values = {row['id']:row['control'] for row in rows}
    method_values = {row['id']:row['method'] for row in rows}
    assessed = assess(control_values, method_values, list(control_values))
    # Do not expose the generic subset arithmetic as a benchmark success gate.
    scores = {key:value for key,value in assessed.items() if key not in
        ('eligible_for_success_claim', 'all_four_point_thresholds_pass', 'caveat')}
    differences = {metric:max(abs(float(row['control'][metric])-row['rescored_control'][metric])
        for row in rows if row['control'][metric] not in ('', None)) for metric in values['method']}
    result = dict(complete=True, targets=len(rows), source_counts=dict(Counter(row['source'] for row in rows)),
        method_statuses=dict(Counter(row['method_status'] for row in rows)),
        control_statuses=dict(Counter(row['control_status'] for row in rows)),
        registered_control_max_rescore_difference=differences,
        metrics=scores['metrics'], clip_coverage_not_reduced=scores['clip_coverage_not_reduced'],
        paired_intervals={metric:paired_estimate(rows, metric) for metric in values['method']},
        rows=rows, contract_sha256=sha256(output/'contract.json'),
        benchmark_success=False, full_benchmark_verified=False, policy_changed=False,
        limitation='Fixed runtime-completed 193-target VG prefix, selected before scoring; not random or a full benchmark. All terminal failures retained. No policy tuning or stopping rule follows this diagnostic.')
    for row in rows:
        if sha256(study/'benchmark/targets'/row['id']/'result.json') != row['method_record_sha256']:
            raise ValueError('Concurrent result mutation')
    write_json(output/'result.json', result)
    print(json.dumps({key:value for key,value in result.items() if key != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
