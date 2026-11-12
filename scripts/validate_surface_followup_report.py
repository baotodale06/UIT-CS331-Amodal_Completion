#!/usr/bin/env python3
"""Check fresh report counts, primary means, chronology and saved artifact hashes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir
    freeze = json.loads(args.freeze.read_text())
    freeze_hash = sha256(args.freeze)
    cases = json.loads((experiment / 'cases.json').read_text())
    summary = json.loads((experiment / 'analysis-eight/summary.json').read_text())
    replay = json.loads((experiment / 'analysis-eight/replay-verification.json').read_text())
    inference = json.loads((experiment / 'inference-eight/contract.json').read_text())
    report = experiment / 'report'
    gallery = json.loads((report / 'gallery_cases.json').read_text())
    status = json.loads((report / 'result-status.json').read_text())
    assert summary['images'] == replay['images'] == len(cases) == len(gallery)
    assert replay['passed'] and replay['candidates'] == len(cases) * len(freeze['candidate_seeds'])
    assert summary['sources'] == dict(Counter(case['source'] for case in cases))
    assert {case['case_id'] for case in gallery} == {case['case_id'] for case in cases}
    assert len({case['image_identity'] for case in cases}) == len(cases)
    for record in (summary, replay, inference, status):
        assert record['freeze_sha256'] == freeze_hash
    assert inference['evaluation_data_access'] is False
    assert datetime.fromisoformat(freeze['created_at']) < datetime.fromisoformat(inference['started_at'])
    selected_at = datetime.fromisoformat(inference['finished_at'])
    for index in range(4):
        runtime = json.loads((experiment / f'metrics-eight/runtime-worker-{index}.json').read_text())
        assert runtime['freeze_sha256'] == freeze_hash
        assert datetime.fromisoformat(runtime['started_at']) > selected_at
        assert datetime.fromisoformat(runtime['finished_at']) > datetime.fromisoformat(runtime['started_at'])
    rows = [json.loads((experiment / 'metrics-eight' / (case['case_id'] + '.json')).read_text())
            for case in cases]
    primary = ('hidden_core_mae', 'object_lpips', 'dreamsim', 'clip_text_cosine')
    primary_gains = {}
    selection_only_gains = {}
    for metric in primary:
        control = np.array([np.mean([value[metric] for value in row['raw']]) for row in rows])
        selected = np.array([row['refined'][row['selection']['trained_surface']][metric] for row in rows])
        sign = 1 if metric == 'clip_text_cosine' else -1
        expected = summary['metrics'][metric]['trained_surface_refined']['uniform_raw']
        assert expected['n'] == len(cases) and expected['excluded_pairs'] == 0
        values = (control.mean(), selected.mean(), sign * (selected - control).mean(),
                  100 * sign * (selected - control).mean() / abs(control.mean()))
        for key, actual in zip(('baseline', 'method', 'absolute_gain', 'relative_gain_percent'), values):
            assert np.isclose(actual, expected[key], atol=1e-12, rtol=0), (metric, key)
        primary_gains[metric] = float(values[-1])
        raw_selected = np.array([row['raw'][row['selection']['trained_surface']][metric] for row in rows])
        raw_gain = 100 * sign * (raw_selected - control).mean() / abs(control.mean())
        raw_expected = summary['metrics'][metric]['trained_surface_raw']['uniform_raw']
        assert np.isclose(raw_selected.mean(), raw_expected['method'], atol=1e-12, rtol=0)
        assert np.isclose(raw_gain, raw_expected['relative_gain_percent'], atol=1e-12, rtol=0)
        selection_only_gains[metric] = float(raw_gain)
    assert status['meets_all_primary_point_targets'] == all(value >= 3 for value in primary_gains.values())
    assert status['selection_only_is_secondary'] is True
    assert status['selection_only_component_meets_all_point_targets'] == all(value >= 3 for value in selection_only_gains.values())
    checks = 0
    for case in gallery:
        assert sha256(report / case['panel']) == case['panel_sha256']
        row = json.loads((experiment / 'metrics-eight' / (case['case_id'] + '.json')).read_text())
        decision = json.loads((experiment / 'inference-eight/decisions' / (case['case_id'] + '.json')).read_text())
        assert decision['index'] == row['selection']['trained_surface']
        assert decision['artifacts'] == row['selected_artifacts'] == case['selected_scientific_sha256']
        for name, expected in row['selected_artifacts'].items():
            assert sha256(experiment / 'metrics-eight/selected' / case['case_id'] / name) == expected
            checks += 1
    assert (report / 'REPORT.pdf').read_bytes().startswith(b'%PDF-')
    result = dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(),
        images=len(cases), selected_artifact_checks=checks, panels=len(gallery),
        primary_gains=primary_gains, selection_only_secondary_gains=selection_only_gains,
        all_selections_preceded_fresh_scoring=True,
        freeze_sha256=freeze_hash, summary_sha256=sha256(experiment / 'analysis-eight/summary.json'),
        report_sha256=sha256(report / 'REPORT.md'), pdf_sha256=sha256(report / 'REPORT.pdf'),
        validator_sha256=sha256(Path(__file__)),
        limitation='Neural scores are checked through input identities and arithmetic; CPU replay does not recompute networks.')
    write_json(report / 'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
