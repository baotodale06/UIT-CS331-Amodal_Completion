#!/usr/bin/env python3
"""Derive a surface-calibrated full run from immutable baseline completions."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.provenance import execution_snapshot
from amodal_benchmark.references import reference_paths
from amodal_benchmark.surface_refinement import RefinementConfig, refine_surface, observation_in_completion_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    baseline, output = a.baseline.resolve(), a.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a new immutable derived-run directory')
    freeze = json.loads(a.freeze.read_text())
    config = RefinementConfig(**freeze['refinement'])
    rows, _ = read_official(a.dataset_root)
    if a.limit:
        rows = rows[:a.limit]
    contract = dict(method='observation-anchored surface residual calibration',
        baseline=str(baseline), baseline_config_sha256=sha256(baseline / 'run_config.json'),
        baseline_execution_sha256=sha256(baseline / 'execution.json'),
        baseline_summary_sha256=sha256(baseline / 'summary.json'),
        freeze_sha256=sha256(a.freeze), refinement=asdict(config),
        source_sha256=sha256(ROOT / 'amodal_benchmark/surface_refinement.py'),
        adapter_sha256=sha256(Path(__file__)),
        geometry='saved integer offset; RGB and masks translated without resizing',
        mask_policy='preserve the original generated alpha exactly',
        failure_policy='preserve all baseline failures exactly')
    output.mkdir(parents=True)
    saved = json.loads((baseline / 'run_config.json').read_text())
    saved['surface_refinement'] = contract
    write_json(output / 'run_config.json', saved)
    write_json(output / 'derivation.json', contract)
    shutil.copyfile(a.freeze, output / 'method_freeze.json')
    # Copy references, never rewrite or relabel them. This makes the derived run
    # portable and independently verifiable even without the baseline directory.
    shutil.copytree(baseline / 'visible_references', output / 'visible_references')
    with execution_snapshot(output, a, rows):
        manifest = []
        for row in rows:
            source_folder = baseline / 'targets' / row['id']
            folder = output / 'targets' / row['id']
            folder.mkdir(parents=True)
            record = json.loads((source_folder / 'result.json').read_text())
            ancestry = dict(baseline_result_sha256=sha256(source_folder / 'result.json'),
                            baseline_completion_sha256=record.get('completion_sha256'))
            if record['status'] == 'success':
                if sha256(source_folder / 'completion.png') != record['completion_sha256']:
                    raise ValueError('Baseline completion checksum mismatch')
                image_path = a.dataset_root / row['image']
                if sha256(image_path) != record['input_sha256']:
                    raise ValueError('Source observation checksum mismatch')
                rgba = np.array(Image.open(source_folder / 'completion.png').convert('RGBA'))
                source_rgb = np.array(Image.open(image_path).convert('RGB'))
                paths = reference_paths(baseline / 'visible_references', row)
                if sha256(paths['mask']) != record['evaluation_reference_artifacts']['mask']:
                    raise ValueError('Observed SAM mask checksum mismatch')
                source_visible = np.array(Image.open(paths['mask']).convert('L')) > 0
                observed, visible = observation_in_completion_frame(source_rgb, source_visible,
                    rgba.shape, record['geometry']['offset_rc'])
                refined, details = refine_surface(observed, visible, rgba[..., :3], rgba[..., 3] > 0, config)
                rgba[..., :3] = refined
                Image.fromarray(rgba).save(folder / 'completion.png')
                Image.fromarray(rgba[..., 3]).save(folder / 'mask.png')
                record['completion_sha256'] = sha256(folder / 'completion.png')
                record['surface_refinement'] = dict(**ancestry, details=details)
                print('REFINED', row['id'], details['support_pixels'], flush=True)
            else:
                record['surface_refinement'] = dict(**ancestry, unchanged_failure=True)
            write_json(folder / 'result.json', record)
            manifest.append(dict(id=row['id'], **ancestry, result_sha256=sha256(folder / 'result.json')))
        write_json(output / 'derivation_manifest.json', manifest)
        snapshots = output / 'derivation_source'
        snapshots.mkdir()
        for path in (Path(__file__), ROOT / 'amodal_benchmark/surface_refinement.py'):
            shutil.copyfile(path, snapshots / path.name)
    print(json.dumps(dict(targets=len(rows), output=str(output))))


if __name__ == '__main__':
    main()
