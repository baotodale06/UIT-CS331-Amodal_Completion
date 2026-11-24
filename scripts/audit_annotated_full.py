#!/usr/bin/env python3
"""Audit a completed annotated full comparison and its frozen unseen subset."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.protocol import METRICS, failure_metrics
from audit_surface_progressive_result import estimate


def main():
    design = ROOT/'output/surface-annotated-20261112-design'
    active = json.loads((design/'active-study.json').read_text())
    study = ROOT/active['study']
    folders = {side:study/side for side in ('baseline','surface')}
    output = study/'full-comparison'
    comparison = json.loads((output/'comparison.json').read_text())
    executions = {side:json.loads((folder/'execution.json').read_text()) for side,folder in folders.items()}
    rows, _ = read_official(ROOT/'datasets/open_world_amodal')
    expected = [row['id'] for row in rows]
    for side, execution in executions.items():
        if execution['status'] != 'complete' or execution['target_ids'] != expected:
            raise ValueError('Full run is unfinished or incomplete')
        if sha256(folders[side]/'execution.json') != comparison[side+'_execution_sha256']:
            raise ValueError('Full comparison execution changed')
    configs = {side:json.loads((folder/'run_config.json').read_text()) for side,folder in folders.items()}
    policy = configs['surface'].pop('surface_policy')
    if configs['baseline'] != configs['surface'] or configs['baseline']['detector_vocabulary']['mode'] != 'annotated':
        raise ValueError('Wrong comparison configuration')
    if policy['freeze_sha256'] != sha256(design/'method-freeze.json'):
        raise ValueError('Trained method changed')
    pairs, counts, transitions = [], {side:Counter() for side in folders}, Counter()
    for official in rows:
        records, metrics = {}, {}
        for side, folder in folders.items():
            target = folder/'targets'/official['id']
            if sha256(target/'result.json') != executions[side]['result_artifacts'][official['id']]:
                raise ValueError('Full-run target identity changed')
            record = records[side] = json.loads((target/'result.json').read_text())
            counts[side][record['status']] += 1
            if record['id'] != official['id']:
                raise ValueError('Wrong target record')
            if record['status'] == 'success':
                cache = json.loads((target/'metrics.json').read_text())
                if cache['inputs']['completion_sha256'] != record['completion_sha256']:
                    raise ValueError('Metric input differs from completion')
                if cache['inputs']['reference_sha256'] not in record['evaluation_reference_artifacts'].values():
                    raise ValueError('Metric input differs from reference')
                metrics[side] = cache['metrics']
            else:
                metrics[side] = failure_metrics()
        for key in ('input_sha256','evaluation_reference_artifacts','seed'):
            if records['baseline'][key] != records['surface'][key]:
                raise ValueError('Paired inputs differ')
        transitions[records['baseline']['status']+' -> '+records['surface']['status']] += 1
        pairs.append(dict(id=official['id'],source=official['source'],
            image_identity=official['source']+'/'+official['original_id'],**metrics,
            baseline_status=records['baseline']['status'],surface_status=records['surface']['status']))
    measured = {metric:estimate(pairs,metric) for metric in METRICS}
    for metric, result in measured.items():
        for key in ('n','images','baseline','method','absolute_gain','relative_gain_percent','absolute_ci','relative_ci_percent'):
            np.testing.assert_allclose(result[key],comparison['metrics'][metric][key],atol=1e-10,rtol=1e-10)
    if dict(transitions) != comparison['status_transitions']:
        raise ValueError('Full failure transition accounting differs')
    unseen = json.loads((design/'unseen-inventory.json').read_text())
    identifiers = set(unseen['target_ids'])
    if len(identifiers) != len(unseen['target_ids']) or not identifiers <= set(expected):
        raise ValueError('Invalid frozen unseen inventory')
    selected = [row for row in pairs if row['id'] in identifiers]
    sensitivity = {metric:estimate(selected,metric) for metric in METRICS}
    result = dict(passed=True,at=datetime.now(timezone.utc).isoformat(),targets=len(pairs),
        statistics_scope='Full operational benchmark, including training/development overlap',
        metrics=measured,status_counts={side:dict(values) for side,values in counts.items()},
        transitions=dict(transitions),unseen_targets=len(selected),
        unseen_images=len({row['image_identity'] for row in selected}),unseen_metrics=sensitivity,
        unseen_inventory_sha256=sha256(design/'unseen-inventory.json'),
        unseen_limitation='Identity/source-byte exclusions only; near-copy screen was exhaustive only for the primary cohort.',
        local_neural_metrics_rerun=False,source_sha256=sha256(Path(__file__)),
        estimate_source_sha256=sha256(ROOT/'scripts/audit_surface_progressive_result.py'))
    write_json(output/'independent-audit.json',result)
    write_json(output/'independent-paired-records.json',pairs)
    lines=['# Frozen unseen-image sensitivity analysis','',
           f'{len(selected)} targets from {result["unseen_images"]} images. Nominal exploratory intervals; full results remain separate.','',
           result['unseen_limitation'],'',
           '| Metric | Control | Surface | Relative gain | 95% interval | n |',
           '| --- | ---: | ---: | ---: | --- | ---: |']
    for metric, row in sensitivity.items():
        lines.append(f'| {metric} | {row["baseline"]:.6f} | {row["method"]:.6f} | '
            f'{row["relative_gain_percent"]:+.3f}% | {row["relative_ci_percent"]} | {row["n"]} |')
    lines += ['', 'All targets retain configured failure penalties. CLIP uses shared successes only. '
              'No new fit, inference or learned-metric scoring was performed in this analysis.', '']
    (output/'UNSEEN_SENSITIVITY.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ('audit_annotated_full.py','audit_surface_progressive_result.py'):
        destination=output/'analysis_source/scripts'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'scripts'/name,destination)
    print(json.dumps(dict(passed=True,targets=len(pairs),unseen_targets=len(selected),metrics=measured),indent=2))


if __name__ == '__main__':
    main()
