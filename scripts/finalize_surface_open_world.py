#!/usr/bin/env python3
"""Verify and summarize a full surface derivation without changing its protocol."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.protocol import METRICS, aggregate, failure_metrics
from amodal_benchmark.provenance import execution_snapshot
from amodal_benchmark.verify import verify_run
from analyze_surface_followup import paired_interval


def image_cluster_interval(items, metric, repetitions=5000):
    """Resample source images, keeping all targets from each image together."""
    higher = metric != 'LPIPS'
    result = paired_interval([x['baseline'][metric] for x in items],
        [x['surface'][metric] for x in items], [x['source'] for x in items], higher)
    rng = np.random.default_rng(20261110)
    control, method, counts = np.zeros(repetitions), np.zeros(repetitions), np.zeros(repetitions)
    for source in sorted({x['source'] for x in items}):
        groups = {}
        for item in items:
            if item['source'] == source:
                groups.setdefault(item['image_identity'], []).append(item)
        sums = np.array([[sum(x['baseline'][metric] for x in group),
                          sum(x['surface'][metric] for x in group), len(group)]
                         for _, group in sorted(groups.items())])
        indices = rng.integers(len(sums),size=(repetitions,len(sums)))
        sampled = sums[indices].sum(1)
        control += sampled[:,0]
        method += sampled[:,1]
        counts += sampled[:,2]
    absolute = (method-control)/counts * (1 if higher else -1)
    denominator = np.abs(control/counts)
    undefined = int(np.count_nonzero(denominator == 0))
    # A resample containing only failure penalties has zero VGG/SSIM baseline.
    # Do not invent a denominator or silently discard these bootstrap draws.
    relative_ci = [None, None] if undefined else np.percentile(
        100 * absolute / denominator, [2.5, 97.5]).tolist()
    result.update(absolute_ci=np.percentile(absolute,[2.5,97.5]).tolist(),
        relative_ci_percent=relative_ci,
        undefined_relative_bootstrap_samples=undefined,
        bootstrap_unit='source image, with all its targets retained',
        images=len({x['image_identity'] for x in items}))
    if result['baseline'] == 0:
        result['relative_gain_percent'] = None
    if undefined:
        result['relative_interval_unavailable_reason'] = 'At least one bootstrap baseline mean is zero'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    a = p.parse_args()
    baseline, run = a.baseline.resolve(), a.run_dir.resolve()
    official, _ = read_official(a.dataset_root)
    by_id = {r['id']:r for r in official}
    with (baseline / 'per_target.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if [r['id'] for r in rows] != [r['id'] for r in official]:
        raise ValueError('Baseline inventory differs from full official target list')
    paired = []
    with execution_snapshot(run, a, official):
        shutil.copyfile(baseline / 'metric_config.json', run / 'metric_config.json')
        for row in rows:
            folder, old_folder = run / 'targets' / row['id'], baseline / 'targets' / row['id']
            record = json.loads((folder / 'result.json').read_text())
            old = json.loads((old_folder / 'result.json').read_text())
            for key in ('status', 'geometry', 'input_sha256', 'evaluation_reference_artifacts',
                        'iterations', 'iteration_limit_reached', 'failure_reason'):
                if record.get(key) != old.get(key):
                    raise ValueError(f'Postprocessing changed protected field {key}: {row["id"]}')
            if record['status'] == 'success':
                first = np.array(Image.open(old_folder / 'completion.png').convert('RGBA'))
                second = np.array(Image.open(folder / 'completion.png').convert('RGBA'))
                if first.shape != second.shape or not np.array_equal(first[..., 3], second[..., 3]):
                    raise ValueError('Derived completion geometry or alpha changed')
                metric = json.loads((folder / 'metrics.json').read_text())
                if metric['inputs']['completion_sha256'] != sha256(folder / 'completion.png'):
                    raise ValueError('Scored completion changed')
                item = json.loads((run / 'paired_scoring' / row['id'] / 'metrics.json').read_text())
                if item['surface'] != metric['metrics']:
                    raise ValueError('Paired and standard metric caches differ')
                row.update(metric['metrics'])
                paired.append(dict(id=row['id'], source=row['source'],
                                   image_identity=row['source']+'/'+by_id[row['id']]['original_id'],
                                   baseline=item['baseline'], surface=item['surface']))
            else:
                penalty = failure_metrics()
                row.update(penalty)
                paired.append(dict(id=row['id'], source=row['source'],
                    image_identity=row['source']+'/'+by_id[row['id']]['original_id'],
                    baseline=penalty, surface=penalty))
        summary = json.loads((baseline / 'summary.json').read_text())
        summary['statistics'] = aggregate(rows, detailed=True)
        summary['surface_derivation'] = json.loads((run / 'derivation.json').read_text())
        summary['protocol_notes'].append(
            'Bounded observation-only color calibration and visible-pixel restoration; '
            'baseline detector, masks, references, failures and metric operations retained. '
            'This derives cached progressive outputs; it does not rerun progressive inference.')
        write_json(run / 'summary.json', summary)
        with (run / 'per_target.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        comparisons = {}
        for metric in METRICS:
            keep = [item for item in paired if item['baseline'][metric] is not None]
            comparisons[metric] = image_cluster_interval(keep,metric)
        write_json(run / 'paired_comparison.json', dict(metrics=comparisons, targets=len(rows),
            same_status=True, same_masks=True, same_reference_protocol=True,
            failure_penalties_included=True, baseline=str(baseline)))
        shutil.copyfile(Path(__file__), run / 'derivation_source' / Path(__file__).name)
    report = verify_run(run, a.dataset_root, require_full=True, require_provenance=True)
    write_json(run / 'verification.json', report)
    print(json.dumps(dict(verification=report['verified'], comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
