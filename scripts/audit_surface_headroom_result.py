#!/usr/bin/env python3
"""Independently reconstruct observable controls and per-metric oracle diagnostics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from audit_surface_progressive_result import estimate, METRICS, PENALTY


def main():
    recovery = ROOT/'output/surface-candidate-headroom-20261111-recovery'
    receipt = json.loads((recovery/'receipt.json').read_text())
    assert sha256(recovery/'evidence.tar.gz') == receipt['sha256']
    recovered = recovery/'evidence'
    for name, expected in receipt['member_sha256'].items():
        assert sha256(recovered/name) == expected, name
    freeze_path = ROOT/'output/surface-candidate-headroom-20261111-design/freeze.json'
    freeze = json.loads(freeze_path.read_text())
    assert sha256(freeze_path) == receipt['result']['freeze_sha256']
    run = recovered/'output/surface-candidate-headroom-20261111'
    parent = ROOT/'output/surface-lastpass-ablation-20261111-recovery/evidence/output/surface-lastpass-ablation-20261111'
    paths = list(run.glob('targets/*/*/headroom.json'))
    rows = {r['id']:r for r in (json.loads(p.read_text()) for p in paths)}
    assert len(rows) == len(paths) == len(freeze['target_ids']) == 2565
    assert set(rows) == set(freeze['target_ids'])
    policies = ['frozen_raw', 'semantic_only', 'uniform_raw', 'corrected']+['oracle_'+m for m in METRICS]
    pairs = {name:[] for name in policies}
    choices = {}
    drift = {m:0.0 for m in METRICS}
    successes = 0
    for identity in freeze['target_ids']:
        row = rows[identity]
        parent_path = parent/'targets'/identity/'ablation.json'
        assert sha256(parent_path) == freeze['ablation_record_sha256'][identity]
        prior = json.loads(parent_path.read_text())
        assert row['inputs'] == dict(freeze_sha256=sha256(freeze_path), ablation_sha256=sha256(parent_path))
        for key in ('id', 'source', 'image_identity', 'baseline_status', 'surface_status'):
            assert row[key] == prior[key], (identity, key)
        assert row['controls'] == prior['scores']
        if row['surface_status'] == 'success':
            successes += 1
            candidates = row['candidate_scores']
            assert len(candidates) == len(row['artifacts']) == len(row['observable_scores']) == 8
            assert all(set(score) == set(METRICS) for score in candidates)
            numeric = np.array([[score[m] for m in METRICS] for score in candidates], dtype=float)
            assert np.isfinite(numeric).all()
            selected = row['selected_index']
            assert selected == prior['derivation']['selected_index']
            assert row['artifacts'][selected]['derived_sha256'] == prior['derivation']['derived_sha256']
            assert row['artifacts'][selected]['candidate_sha256'] == prior['derivation']['raw_candidate_sha256']
            assert all(a['fixed_mask_sha256'] == prior['derivation']['fixed_mask_sha256'] for a in row['artifacts'])
            valid = [i for i, observed in enumerate(row['observable_scores']) if observed['valid']]
            semantic = int(valid[np.argmax([row['observable_scores'][i]['candidate_text_cosine'] for i in valid])]) if valid else 0
            decisions = dict(frozen_raw=selected, semantic_only=semantic)
            scores = dict(frozen_raw=candidates[selected], semantic_only=candidates[semantic],
                          uniform_raw=dict(zip(METRICS, numeric.mean(axis=0))), corrected=prior['scores']['surface'])
            for j, metric in enumerate(METRICS):
                index = int(np.argmin(numeric[:, j]) if metric=='LPIPS' else np.argmax(numeric[:, j]))
                decisions['oracle_'+metric] = index
                scores['oracle_'+metric] = candidates[index]
                drift[metric] = max(drift[metric], abs(candidates[selected][metric]-prior['scores']['raw_final'][metric]))
            choices[identity] = decisions
        else:
            assert row['candidate_scores'] == row['artifacts'] == row['observable_scores'] == []
            assert row['selected_index'] is None
            scores = {name:PENALTY.copy() for name in policies}
        for name in policies:
            pairs[name].append(dict(id=identity, source=row['source'], image_identity=row['image_identity'],
                                    baseline=prior['scores']['baseline'], surface=scores[name]))
    assert successes == 958 and max(drift.values()) <= freeze['selected_metric_replay_tolerance']
    reported = json.loads((run/'analysis/summary.json').read_text())
    assert json.loads((run/'analysis/oracle_and_control_choices.json').read_text()) == choices
    comparisons = {}
    for name, items in pairs.items():
        comparisons[name] = {}
        for metric in METRICS:
            result = estimate(items, metric)
            for key, value in result.items():
                np.testing.assert_allclose(value, reported['contrasts'][name][metric][key], rtol=0, atol=1e-10)
            comparisons[name][metric] = result
    assert reported['selected_score_max_absolute_drift'] == drift
    ceilings = {m:comparisons['oracle_'+m][m] for m in METRICS}
    for metric in METRICS:
        for key, value in ceilings[metric].items():
            np.testing.assert_allclose(value, reported['independent_single_metric_oracle_ceilings'][metric][key], rtol=0, atol=1e-10)
    below = [m for m in METRICS if ceilings[m]['relative_gain_percent'] < 3]
    assert reported['any_single_metric_raw_ceiling_below3'] == bool(below)
    runtimes = [json.loads((run/f'runtime-worker-{i}.json').read_text()) for i in range(4)]
    for i, runtime in enumerate(runtimes):
        assert runtime['worker'] == i and runtime['freeze_sha256'] == sha256(freeze_path)
        assert datetime.fromisoformat(freeze['created_at']) < datetime.fromisoformat(runtime['started_at']) < datetime.fromisoformat(runtime['finished_at'])
        assert runtime['environment']['packages'] == runtimes[0]['environment']['packages']
    audit = dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(), targets=len(rows),
        saved_pools=successes, candidates=successes*8, recovered_member_checks=len(receipt['member_sha256']),
        parent_record_checks=len(rows), contrasts=comparisons,
        independent_single_metric_oracle_ceilings=ceilings, raw_ceiling_below3_metrics=below,
        selected_score_max_absolute_drift=drift, oracle_choices_verified=successes*4,
        source_sha256=sha256(Path(__file__)), scope=freeze['scope'],
        native_pngs_locally_verified=False, neural_metrics_recomputed=False)
    write_json(ROOT/'output/surface-lastpass-review-20261111/headroom-audit.json', audit)
    print(json.dumps(dict(passed=True, targets=len(rows), candidates=successes*8,
        raw_ceiling_below3_metrics=below, ceilings=ceilings), indent=2))


if __name__ == '__main__':
    main()
