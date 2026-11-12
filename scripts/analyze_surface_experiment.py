#!/usr/bin/env python3
"""Freeze development choices, then evaluate a disjoint held-out test set."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import METHODS
from amodal_benchmark.surface_evaluation import (load_pool, evaluate_pool, normalization,
                                                selections, summarize_cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--stage', choices=('freeze', 'test'), required=True)
    p.add_argument('--freeze-from', type=Path, help='Image-disjoint replication: reuse the pilot development freeze')
    args = p.parse_args()
    experiment, run = args.experiment_dir, args.run_dir
    output = experiment / 'analysis'
    output.mkdir(exist_ok=True)
    cases = json.loads((experiment / 'cases.json').read_text())
    protocol = json.loads((experiment / 'protocol.json').read_text())
    if len({c['image_identity'] for c in cases}) != len(cases):
        raise ValueError('Source image leakage between cases')
    if len({c['source_sha256'] for c in cases}) != len(cases):
        raise ValueError('Byte-identical source image leakage between cases')
    split = 'development' if args.stage == 'freeze' else 'test'
    active = [case for case in cases if case['split'] == split]
    pools = [load_pool(experiment, run, case, protocol['candidate_seeds']) for case in active]
    lock_path = args.freeze_from or output / 'development_freeze.json'
    source_root = Path(__file__).resolve().parents[1]
    analysis_hashes = {relative: sha256(source_root / relative) for relative in (
        'scripts/analyze_surface_experiment.py', 'amodal_benchmark/surface_evaluation.py',
        'amodal_benchmark/surface.py', 'amodal_benchmark/appearance.py')}
    if args.stage == 'freeze':
        if args.freeze_from:
            raise ValueError('Replication cannot refit development choices')
        if lock_path.exists():
            raise ValueError('Development decisions are already frozen; do not overwrite')
        scales = normalization(pools)
        development_method = None
    else:
        lock = json.loads(lock_path.read_text())
        if args.freeze_from:
            parent = args.freeze_from.parent.parent
            if lock['protocol_sha256'] != sha256(parent / 'protocol.json') or lock['cases_sha256'] != sha256(parent / 'cases.json'):
                raise ValueError('Pilot identity changed after development freeze')
            parent_protocol = json.loads((parent / 'protocol.json').read_text())
            cohort_keys = {'experiment', 'cases_per_source', 'development_per_source', 'source_case_limits',
                           'replication_parent', 'normalization'}
            if ({k: v for k, v in protocol.items() if k not in cohort_keys} !=
                    {k: v for k, v in parent_protocol.items() if k not in cohort_keys}):
                raise ValueError('Replication changed scientific settings beyond the declared cohort')
            parent_cases = json.loads((parent / 'cases.json').read_text())
            if ({c['image_identity'] for c in cases} & {c['image_identity'] for c in parent_cases} or
                    {c['source_sha256'] for c in cases} & {c['source_sha256'] for c in parent_cases}):
                raise ValueError('Replication overlaps pilot source images')
            if any(c['split'] != 'test' for c in cases):
                raise ValueError('Replication must be entirely held out')
        elif lock['protocol_sha256'] != sha256(experiment / 'protocol.json') or lock['cases_sha256'] != sha256(experiment / 'cases.json'):
            raise ValueError('Experiment identity changed after development freeze')
        if lock['scorer_sha256'] != sha256(Path(__file__).resolve().parents[1] / 'amodal_benchmark/surface.py'):
            raise ValueError('Scoring source changed after development freeze')
        if lock['methods'] != METHODS:
            raise ValueError('Selection methods changed after development freeze')
        if lock['analysis_source_hashes'] != analysis_hashes:
            raise ValueError('Scientific analysis source changed after development freeze')
        scales, development_method = lock['scales'], lock['development_method']
    rows = []
    for case, records in zip(active, pools):
        metrics = evaluate_pool(experiment, run, case, records)
        chosen = selections(records, metrics, scales, development_method)
        rows.append(dict(case_id=case['case_id'], source=case['source'], split=case['split'],
                         label=case['label'], target_id=case['target_id'], metrics=metrics,
                         scores=[r.get('surface', {'valid': False}) for r in records], selected=chosen,
                         statuses=[r['status'] for r in records]))
    summary = summarize_cases(rows, split)
    if args.stage == 'freeze':
        eligible = ['fixed_seed_11', *METHODS]
        development_method = min(eligible, key=lambda method: summary[method]['hidden_core_mae']['mean'])
        write_json(lock_path, dict(created_at=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha256(experiment / 'protocol.json'), cases_sha256=sha256(experiment / 'cases.json'),
            scorer_sha256=sha256(Path(__file__).resolve().parents[1] / 'amodal_benchmark/surface.py'),
            analysis_source_hashes=analysis_hashes,
            analysis_runtime={'python': sys.version, 'numpy': np.__version__},
            methods=METHODS, scales=scales, development_method=development_method,
            development_case_ids=[c['case_id'] for c in active],
            record_hashes={str(path.relative_to(run)): sha256(path) for case in active for path in
                           (run / 'candidates' / case['case_id']).glob('seed-*.json')},
            development_summary=summary))
    write_json(output / f'{split}_cases.json', rows)
    write_json(output / f'{split}_summary.json', dict(methods=summary,
        candidate_statuses=Counter(r['status'] for records in pools for r in records),
        invalid_reasons=Counter(reason for records in pools for r in records
                               for reason in r.get('surface', {}).get('invalid_reasons', [])),
        development_method=development_method, cases=len(rows), candidates=sum(map(len, pools)),
        freeze_sha256=sha256(lock_path), analyzed_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(split=split, development_method=development_method,
        results={name: {key: val[key] for key in ('hidden_core_mae', 'gain_vs_uniform', 'fallbacks')}
                 for name, val in summary.items()}), indent=2))


if __name__ == '__main__':
    main()
