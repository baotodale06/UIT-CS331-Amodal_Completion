#!/usr/bin/env python3
"""Compare separate diagnostic executions without altering frozen experiments."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import write_json
from scripts.validate_surface_experiment import assert_equivalent


def differences(first, second, path=''):
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError('Changed fields at ' + path)
        return [item for key in first for item in differences(first[key], second[key], path + '/' + str(key))]
    if isinstance(first, list):
        if len(first) != len(second):
            raise ValueError('Changed length at ' + path)
        return [item for index, (a, b) in enumerate(zip(first, second))
                for item in differences(a, b, path + '/' + str(index))]
    if first == second:
        return []
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        raise ValueError('Changed nonnumeric value at ' + path)
    return [dict(path=path, local=first, cluster=second, absolute_difference=abs(first - second))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    local, cluster = args.pilot / 'diagnostics', args.pilot / 'cluster-diagnostics'
    a, b = [json.loads((folder / 'cases.json').read_text()) for folder in (local, cluster)]
    changed = differences(a, b)
    assert_equivalent(json.loads((local / 'summary.json').read_text()),
                      json.loads((cluster / 'summary.json').read_text()))
    preference_changes = sum((x['fixed_support_truth']['truth_minus_chosen'] > 1e-10) !=
                             (y['fixed_support_truth']['truth_minus_chosen'] > 1e-10)
                             for x, y in zip(a, b))
    if preference_changes:
        raise ValueError('Truth-preference decisions changed')
    result = dict(status='passed_aggregate_and_preference_replay',
        verified_at=datetime.now(timezone.utc).isoformat(),
        summary_file_identical=(local / 'summary.json').read_bytes() == (cluster / 'summary.json').read_bytes(),
        truth_preference_decision_changes=preference_changes, per_case_differing_numbers=len(changed),
        largest_per_case_difference=max(changed, key=lambda d: d['absolute_difference']) if changed else None,
        local_runtime=dict(python='3.10', numpy='1.23.5', pillow='9.5.0'),
        scope='Post-hoc diagnostic summaries and truth-preference decisions agree. Per-case terms are numerically sensitive and not byte-identical. This is separate from the original experiments, which passed exact raw-score replay. Cluster diagnostic runtime was not recorded in its manifest.')
    write_json(args.pilot / 'diagnostic_replay.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
