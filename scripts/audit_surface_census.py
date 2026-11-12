#!/usr/bin/env python3
"""Exploratory compatibility audit against pre-existing single-assistant labels.

These historical outputs have no hidden RGB truth and only one candidate each.
This cannot establish a reranking improvement or independent human agreement.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.appearance import aligned_artifacts
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import score_surface


def score_one(arguments):
    run, annotation = arguments
    record_path = Path(run) / 'targets' / annotation['id'] / 'result.json'
    record = json.loads(record_path.read_text())
    reference, candidate = aligned_artifacts(run, record)
    union = (reference[..., 3] > 0) | (candidate[..., 3] > 0)
    rr, cc = np.where(union)
    side = int(np.ceil(max(np.ptp(rr) + 1, np.ptp(cc) + 1) * 1.1))
    cx, cy = (int(cc.min()) + int(cc.max())) // 2, (int(rr.min()) + int(rr.max())) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    # Nearest sampling avoids mixing unknown/transparent source pixels into V.
    a, b = [np.array(Image.fromarray(arr).crop(box).resize((512, 512), Image.Resampling.NEAREST))
            for arr in (reference, candidate)]
    visible, predicted = a[..., 3] > 0, b[..., 3] > 0
    score = score_surface(a[..., :3].copy(), visible, ~visible, b[..., :3].copy(), predicted)
    paths = [record_path, Path(run) / 'visible_references' / record['reference'],
             Path(run) / 'visible_references/masks' / record['reference'],
             Path(run) / 'targets' / annotation['id'] / 'completion.png']
    return dict(id=annotation['id'], source=annotation['source'], label=annotation['label'],
        image_identity=record.get('source', annotation['source']) + '/' + record.get('original_id', annotation['id']),
        labels={key: annotation[key] for key in ('wrong_texture', 'color_inconsistency', 'pattern_inconsistency',
                                                'hidden_region_inconsistency')},
        confound=annotation['identity_or_mask_confound'], score=score, crop_box=box,
        hashes={str(p.relative_to(run)): sha256(p) for p in paths})


def auc(labels, values):
    from scipy.stats import rankdata
    labels, values = np.asarray(labels, bool), np.asarray(values, float)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    ranks = rankdata(values)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def associations(rows, scales):
    from scipy.stats import spearmanr
    output = []
    for row in rows:
        score = row['score']
        score['surface_full'] = (sum(score[k] / scales[k] * w for k, w in
            [('boundary', .4), ('color', .25), ('texture', .35)])
            if score['valid'] and all(score.get(k) is not None for k in ('boundary', 'color', 'texture')) else None)
    for label, component in [('wrong_texture', 'texture'), ('color_inconsistency', 'color'),
                              ('pattern_inconsistency', 'local_boundary'), ('hidden_region_inconsistency', 'surface_full')]:
        for subset in ('all', 'without_mask_identity_low_resolution_confounds'):
            excluded_tags = ('mask', 'target_switch', 'identity', 'low_resolution', 'small_target')
            eligible = [row for row in rows if row['score']['valid'] and row['score'].get(component) is not None and
                        row['labels'][label] not in ('U', 'NA') and
                        (subset == 'all' or not any(tag in row['confound'].lower() for tag in excluded_tags))]
            labels = np.array([int(r['labels'][label]) for r in eligible])
            values = np.array([r['score'][component] for r in eligible])
            rank_correlation = float(spearmanr(labels, values).statistic) if len(set(labels)) > 1 else None
            # Descriptive AUC, no independent-rater or causal claim; clustered
            # uncertainty can be added without treating target rows as independent.
            output.append(dict(label=label, component=component, subset=subset, n=len(eligible),
                positives=int((labels > 0).sum()), negatives=int((labels == 0).sum()),
                auc_higher_loss_predicts_failure=auc(labels > 0, values),
                ordinal_spearman=rank_correlation,
                median_positive=float(np.median(values[labels > 0])) if (labels > 0).any() else None,
                median_negative=float(np.median(values[labels == 0])) if (labels == 0).any() else None))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--freeze', type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / 'scores.json'
    if cache.exists():
        manifest = json.loads((args.output_dir / 'manifest.json').read_text())
        if manifest['annotations_sha256'] != sha256(args.annotations) or manifest['scorer_sha256'] != sha256(ROOT / 'amodal_benchmark/surface.py'):
            raise ValueError('Census inputs/scorer changed; use a new output directory')
        rows = json.loads(cache.read_text())
    else:
        with args.annotations.open(encoding='utf-8-sig', newline='') as stream:
            annotations = list(csv.DictReader(stream))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = []
            for row in pool.map(score_one, [(str(args.run_dir), a) for a in annotations], chunksize=8):
                rows.append(row)
                if len(rows) % 100 == 0:
                    print('SCORED', len(rows), flush=True)
        write_json(cache, rows)
        write_json(args.output_dir / 'manifest.json', dict(
            annotations_sha256=sha256(args.annotations), scorer_sha256=sha256(ROOT / 'amodal_benchmark/surface.py'),
            source_sha256=sha256(Path(__file__)), rows=len(rows), scores_sha256=sha256(cache),
            run_dir=str(args.run_dir), labels='pre-existing single-assistant census; no independent human ratings',
            crop='union of visible and predicted support, square 10% margin, nearest resize to 512',
            occlusion='complement of observed visible mask; generated expansion is not true occlusion'))
    summary = dict(total=len(rows), valid=sum(r['score']['valid'] for r in rows),
        invalid_reasons=Counter(reason for r in rows for reason in r['score']['invalid_reasons']))
    if args.freeze:
        freeze = json.loads(args.freeze.read_text())
        summary['associations'] = associations(rows, freeze['scales'])
        summary['freeze_sha256'] = sha256(args.freeze)
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
