#!/usr/bin/env python3
"""Rescore in another CPU runtime without replacing frozen decisions or results."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface import score_surface, SurfaceConfig
from amodal_benchmark.surface_evaluation import selections, summarize_cases


def rescore(arguments):
    experiment, row, config, scales, development_method = arguments
    experiment = Path(experiment)
    case = row['case_id']
    read_rgb = lambda path: np.array(Image.open(path).convert('RGB'))
    read_mask = lambda path: np.array(Image.open(path).convert('L')) > 0
    inp = experiment / 'inputs' / case
    observed, visible, occlusion = read_rgb(inp / 'observed.png'), read_mask(inp / 'visible.png'), read_mask(inp / 'occlusion.png')
    records, differences = [], []
    for index, metric in enumerate(row['metrics']):
        folder = experiment / 'full/candidates' / case
        original = json.loads((folder / f"seed-{metric['seed']}.json").read_text())
        if original.get('surface'):
            score = score_surface(observed, visible, occlusion,
                read_rgb(folder / f"seed-{metric['seed']}.png"),
                read_mask(folder / f"seed-{metric['seed']}-mask.png"), SurfaceConfig(**config))
            for key, value in score.items():
                if isinstance(value, float) and original['surface'].get(key) is not None:
                    differences.append(dict(seed=metric['seed'], component=key, absolute_difference=abs(value - original['surface'][key])))
            original['surface'] = score
        records.append(original)
    old = row['selected']
    row['scores'] = [r.get('surface', {'valid': False}) for r in records]
    row['selected'] = selections(records, row['metrics'], scales, development_method)
    changes = [name for name in old if (old[name]['index'], old[name]['fallback']) !=
               (row['selected'][name]['index'], row['selected'][name]['fallback'])]
    return row, dict(case_id=case, changed_methods=changes, differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new numerical-audit output directory')
    freeze = json.loads(args.freeze.read_text())
    for relative, expected in freeze['analysis_source_hashes'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError('Frozen scientific source changed')
    protocol = json.loads((args.experiment_dir / 'protocol.json').read_text())
    config = dict(radius=protocol['boundary_radius'], min_region_pixels=protocol['min_region_pixels'],
                  min_visible_recall=protocol['min_visible_recall'])
    rows = json.loads((args.experiment_dir / 'analysis/test_cases.json').read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        replayed = list(pool.map(rescore, [(str(args.experiment_dir), row, config,
            freeze['scales'], freeze['development_method']) for row in rows]))
    new_rows, differences = zip(*replayed)
    summary = summarize_cases(list(new_rows), 'test')
    all_differences = [dict(case_id=row['case_id'], **value) for row in differences for value in row['differences']]
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Post-hoc numerical portability sensitivity; original results and scales are unchanged',
        runtime=dict(python=sys.version, numpy=np.__version__, pillow=PIL.__version__, platform=platform.platform()),
        cases=len(rows), decision_changes={name: sum(name in row['changed_methods'] for row in differences) for name in rows[0]['selected']},
        max_term_difference=max(all_differences, key=lambda value: value['absolute_difference']),
        comparisons_above_1e4=sum(row['absolute_difference'] > 1e-4 for row in all_differences),
        method_results=summary, scorer_sha256=freeze['scorer_sha256'], freeze_sha256=sha256(args.freeze),
        audit_source_sha256=sha256(Path(__file__)))
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'summary.json', result)
    write_json(args.output_dir / 'cases.json', differences)
    print(json.dumps({key: result[key] for key in ('cases', 'runtime', 'decision_changes', 'max_term_difference', 'comparisons_above_1e4')}, indent=2))


if __name__ == '__main__':
    main()
