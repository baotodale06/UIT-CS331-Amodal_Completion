#!/usr/bin/env python3
"""Paired, source-stratified analysis of a frozen fresh Surface-PMC experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

HIGHER = {'object_feature_similarity', 'clip_text_cosine', 'clip_text_logit', 'object_ssim'}


@lru_cache(maxsize=512)
def bootstrap_mean(values, sources, seed, repetitions):
    """Reuse identical resamples for the many paired policy comparisons."""
    values, sources = np.asarray(values), np.asarray(sources)
    rng = np.random.default_rng(seed)
    result = np.zeros(repetitions)
    for source in sorted(set(sources)):
        group = values[sources == source]
        indices = rng.integers(len(group), size=(repetitions, len(group)))
        result += group[indices].sum(1)
    return result / len(values)


def paired_interval(control, method, sources, higher=False, seed=20261110, repetitions=5000):
    control, method, sources = np.asarray(control, float), np.asarray(method, float), np.asarray(sources)
    if not len(control) or not np.isfinite(control).all() or not np.isfinite(method).all():
        raise ValueError('Finite paired observations required')
    cb = bootstrap_mean(tuple(control), tuple(sources), seed, repetitions)
    mb = bootstrap_mean(tuple(method), tuple(sources), seed, repetitions)
    sign = 1 if higher else -1
    gain = sign * (method - control)
    absolute = sign * (mb - cb)
    relative = 100 * absolute / np.maximum(np.abs(cb), 1e-12)
    return dict(n=len(control), baseline=float(control.mean()), method=float(method.mean()),
        absolute_gain=float(gain.mean()), absolute_ci=np.percentile(absolute, [2.5,97.5]).tolist(),
        relative_gain_percent=float(100 * gain.mean() / max(abs(control.mean()), 1e-12)),
        relative_ci_percent=np.percentile(relative, [2.5,97.5]).tolist(),
        wins=int((gain > 1e-10).sum()), losses=int((gain < -1e-10).sum()),
        direction='higher' if higher else 'lower')


def policies(row, metric):
    raw, refined = row['raw'], row['refined']
    def mean(values):
        return float(np.mean(values)) if all(v is not None for v in values) else None
    result = dict(uniform_raw=mean([r[metric] for r in raw]),
                  uniform_refined=mean([r[metric] for r in refined]))
    if 'valid_indices' in row:
        valid = row['valid_indices'] or [0]
        result['uniform_valid_raw'] = mean([raw[i][metric] for i in valid])
        result['uniform_valid_refined'] = mean([refined[i][metric] for i in valid])
    if len(raw) > 4:
        result['uniform_four_raw'] = mean([r[metric] for r in raw[:4]])
    for name, index in row['selection'].items():
        result[name + '_raw'] = raw[index][metric]
        result[name + '_refined'] = refined[index][metric]
    if row.get('calibrated'):
        result['uniform_calibrated'] = mean([r[metric] for r in row['calibrated']])
        result['trained_surface_calibrated'] = row['calibrated'][row['selection']['trained_surface']][metric]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--metrics-dir', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--development-review', action='store_true')
    a = p.parse_args()
    freeze = json.loads(a.freeze.read_text())
    if not a.development_review and freeze['status'] != 'frozen_before_fresh_test':
        raise ValueError('Fresh test requires the final pre-test freeze')
    cases = json.loads((a.experiment_dir / 'cases.json').read_text())
    if not a.development_review:
        if set(r['image_identity'] for r in cases) & set(freeze['development']['image_identities']):
            raise ValueError('Test image identity overlaps development')
        if set(r['source_sha256'] for r in cases) & set(freeze['development']['image_sha256']):
            raise ValueError('Test image bytes overlap development')
    rows = []
    for case in cases:
        row = json.loads((a.metrics_dir / (case['case_id'] + '.json')).read_text())
        if row['case_id'] != case['case_id'] or row['freeze_sha256'] != sha256(a.freeze):
            raise ValueError('Metric policy or identity mismatch')
        rows.append(row)
    summary, table = {}, []
    for metric in rows[0]['raw'][0]:
        policy_rows = [policies(row, metric) for row in rows]
        summary[metric] = {}
        for policy in policy_rows[0]:
            summary[metric][policy] = {}
            controls = ['uniform_raw', 'seed_11_raw', 'consensus_only_raw', 'local_surface_raw']
            controls += [k for k in ('uniform_valid_raw','semantic_only_raw','visible_deep_only_raw',
                         'uniform_refined','uniform_valid_refined','consensus_only_refined',
                         'local_surface_refined','semantic_only_refined','visible_deep_only_refined',
                         'uniform_calibrated')
                         if k in policy_rows[0]]
            for control in controls:
                keep = [i for i,r in enumerate(policy_rows) if r[policy] is not None and r[control] is not None]
                if not keep:
                    continue
                value = paired_interval([policy_rows[i][control] for i in keep],
                    [policy_rows[i][policy] for i in keep], [rows[i]['source'] for i in keep], metric in HIGHER)
                value['excluded_pairs'] = len(rows) - len(keep)
                value['by_source'] = {source: paired_interval(
                    [policy_rows[i][control] for i in keep if rows[i]['source'] == source],
                    [policy_rows[i][policy] for i in keep if rows[i]['source'] == source],
                    [source for i in keep if rows[i]['source'] == source], metric in HIGHER)
                    for source in sorted({rows[i]['source'] for i in keep})}
                summary[metric][policy][control] = value
                table.append(dict(metric=metric, policy=policy, control=control,
                    **{k:v for k,v in value.items() if k != 'by_source'}))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), images=len(rows),
        analysis_source_sha256=sha256(Path(__file__)),
        sources=dict(Counter(r['source'] for r in rows)), freeze_sha256=sha256(a.freeze),
        metrics=summary, generation_failures=sum(r['generation_failures'] for r in rows),
        candidate_errors=sum(r.get('candidate_error_count',0) for r in rows),
        analysis='paired, source-stratified image bootstrap; 5000 resamples; nominal 95% intervals',
        development_review=a.development_review)
    write_json(a.output_dir / 'summary.json', result)
    with (a.output_dir / 'comparisons.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['# Frozen Surface-PMC evaluation', '', f"{len(rows)} images; sources {result['sources']}.", '',
        '| Metric | Uniform baseline | Surface | Relative gain | 95% relative interval |',
        '| --- | ---: | ---: | ---: | --- |']
    for metric in freeze['endpoints']:
        value = summary[metric]['trained_surface_refined'].get('uniform_raw')
        if value:
            low, high = value['relative_ci_percent']
            lines.append(f"| {metric} | {value['baseline']:.6f} | {value['method']:.6f} | "
                         f"{value['relative_gain_percent']:+.2f}% | [{low:+.2f}%, {high:+.2f}%] |")
    lines += ['', 'Positive gain follows each metric\'s stated direction. Uniform is the exact expected score',
              f"from selecting one of the same {len(rows[0]['raw'])} candidates. K=1 and other equal-budget controls",
              'are in comparisons.csv. This controlled artificial-occlusion study is separate from',
              'the full progressive benchmark. Intervals are nominal and comparisons are not multiplicity-adjusted.']
    (a.output_dir / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
