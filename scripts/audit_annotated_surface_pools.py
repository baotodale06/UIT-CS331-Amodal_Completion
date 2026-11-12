#!/usr/bin/env python3
"""Independently replay annotated final-pool policies and paired intervals."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.protocol import METRICS, failure_metrics
from audit_surface_progressive_result import estimate
from measure_surface_candidate_headroom import derive_candidate


def independent_policies(record):
    """Use only recorded candidates/features, with explicit invalid-pool fallback."""
    names = ('frozen_raw', 'uniform_raw', 'uniform_valid', 'first_raw', 'semantic_only', 'corrected')
    names += tuple('oracle_' + metric for metric in METRICS)
    if record['surface_status'] != 'success':
        if record['candidate_scores'] or record['candidate_artifacts']:
            raise ValueError('Failed target has scored candidates')
        return {name: failure_metrics() for name in names}, {}
    scores, observable = record['candidate_scores'], record['observable_scores']
    if len(scores) != 8 or len(observable) != 8:
        raise ValueError('Incomplete candidate pool')
    if any(set(row) != set(METRICS) or not all(np.isfinite(v) for v in row.values()) for row in scores):
        raise ValueError('Invalid candidate metrics')
    valid = [i for i, row in enumerate(observable) if row['valid']] or [0]
    semantic = sorted(valid, key=lambda i: (-observable[i]['candidate_text_cosine'], i))[0]
    choices = dict(frozen_raw=record['selected_index'], first_raw=0, semantic_only=semantic)
    for metric in METRICS:
        sign = 1 if metric == 'LPIPS' else -1
        choices['oracle_' + metric] = min(range(8), key=lambda i: (sign * scores[i][metric], i))
    policies = {name: scores[index] for name, index in choices.items()}
    policies['corrected'] = record['controls']['surface']
    policies['uniform_raw'] = {metric: sum(row[metric] for row in scores) / 8 for metric in METRICS}
    policies['uniform_valid'] = {metric: sum(scores[i][metric] for i in valid) / len(valid) for metric in METRICS}
    return policies, choices


def main():
    design = ROOT / 'output/surface-annotated-20261112-design'
    study = ROOT / 'output/surface-annotated-20261112'
    diagnostic = study / 'pool-diagnostic'
    freeze = json.loads((design / 'pool-freeze.json').read_text())
    cohort = json.loads((design / 'cohort.json').read_text())
    summary = json.loads((diagnostic / 'analysis/summary.json').read_text())
    expected_choices = json.loads((diagnostic / 'analysis/choices.json').read_text())
    paths = list(diagnostic.glob('targets/*/*/pool.json'))
    records = {row['id']: row for row in (json.loads(path.read_text()) for path in paths)}
    if set(records) != set(cohort['target_ids']) or len(records) != len(paths):
        raise ValueError('Diagnostic inventory does not equal the frozen cohort')
    pairs, choices, derivations = [], {}, 0
    drift = {metric: 0. for metric in METRICS}
    with tempfile.TemporaryDirectory(prefix='annotated-pool-audit-') as temporary:
        for identity in cohort['target_ids']:
            row = records[identity]
            if row['inputs']['freeze_sha256'] != sha256(design / 'pool-freeze.json'):
                raise ValueError('Diagnostic freeze changed')
            for side in ('baseline', 'surface'):
                parent = study / (side + '-cohort') / 'targets' / identity
                if sha256(parent / 'result.json') != row['inputs'][side + '_result_sha256']:
                    raise ValueError('Parent target identity differs')
                result = json.loads((parent / 'result.json').read_text())
                if result['status'] != row[side + '_status']:
                    raise ValueError('Target status differs')
                metric_path = parent / 'metrics.json'
                expected = json.loads(metric_path.read_text())['metrics'] if result['status'] == 'success' else failure_metrics()
                if row['cached_controls'][side] != expected:
                    raise ValueError('Original metric cache differs')
                for metric in METRICS:
                    actual = row['controls'][side][metric]
                    if expected[metric] is None:
                        if actual is not None:
                            raise ValueError('Invented failure metric')
                    else:
                        drift[metric] = max(drift[metric], abs(actual - expected[metric]))
                if side == 'surface' and result['status'] == 'success':
                    trace = result['diagnostics']['surface_iterations'][-1]
                    folder = parent / 'surface' / f'iteration-{trace["iteration"]:02d}'
                    if row['decision_sha256'] != sha256(folder / 'decision.json'):
                        raise ValueError('Final choice identity differs')
                    decision = json.loads((folder / 'decision.json').read_text())
                    if row['observable_scores'] != decision['observable_scores'] or row['selected_index'] != decision['index']:
                        raise ValueError('Candidate selection features differ')
                    for index, expected_artifact in enumerate(row['candidate_artifacts']):
                        _, actual_artifact = derive_candidate(parent, Path(temporary), result, decision, index)
                        if actual_artifact != expected_artifact:
                            raise ValueError('Native candidate derivation differs')
                        derivations += 1
            values, chosen = independent_policies(row)
            if chosen:
                choices[identity] = chosen
            pairs.append(dict(id=identity, source=row['source'], image_identity=row['image_identity'],
                              scores=dict(baseline=row['controls']['baseline'], **values)))
    if choices != expected_choices:
        raise ValueError('Recorded policy choices do not replay independently')
    comparisons = {}
    for name, expected in summary['contrasts'].items():
        method, control = name.split('_vs_')
        rows = [dict(source=row['source'], image_identity=row['image_identity'],
                     baseline=row['scores'][control], surface=row['scores'][method]) for row in pairs]
        comparisons[name] = {}
        for metric in METRICS:
            actual = estimate(rows, metric)
            if actual is None:
                if expected[metric]['n'] != 0:
                    raise ValueError('Missing paired observations')
            else:
                for key in ('n', 'images', 'baseline', 'method', 'absolute_gain', 'relative_gain_percent', 'absolute_ci', 'relative_ci_percent'):
                    np.testing.assert_allclose(actual[key], expected[metric][key], atol=1e-10, rtol=1e-10)
            comparisons[name][metric] = actual
    if max(drift.values()) > freeze['replay_tolerance']:
        raise ValueError('Neural metric replay exceeds the pre-specified tolerance')
    write_json(study / 'report/pool-audit.json', dict(passed=True,
        audited_at=datetime.now(timezone.utc).isoformat(), targets=len(records),
        candidate_derivations=derivations, choices_replayed=len(choices),
        cache_replay_drift=drift, contrasts=comparisons, local_neural_metrics_rerun=False,
        source_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(passed=True, targets=len(records), candidate_derivations=derivations)))


if __name__ == '__main__':
    main()
