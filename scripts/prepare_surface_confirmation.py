#!/usr/bin/env python3
"""Freeze a new confirmation of the already chosen selection-only component."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.surface_policy import verify_frozen_surface_sources
from audit_surface_split_images import fingerprint

PREVIOUS = ('surface-consistency-20261108', 'surface-consistency-replication87-20261108',
            'surface-followup-20261110')
NEW_SOURCES = ('scripts/prepare_surface_confirmation.py',
               'scripts/finalize_surface_confirmation.py',
               'scripts/run_surface_confirmation_slurm.sh',
               'scripts/prepare_surface_experiment.py',
               'scripts/run_surface_candidates.py',
               'scripts/run_surface_followup_slurm.sh',
               'scripts/audit_surface_split_images.py',
               'scripts/analyze_surface_followup.py',
               'scripts/validate_surface_followup.py')


def confirmation_freeze(parent, previous, parent_sha256):
    freeze = deepcopy(parent)
    freeze.update(created_at=datetime.now(timezone.utc).isoformat(),
        method='Surface-PMC selection only; unchanged eight-candidate ranker',
        primary_policy='trained_surface_raw', primary_control='uniform_raw',
        endpoints=['hidden_core_mae', 'object_lpips', 'dreamsim', 'clip_text_cosine'],
        refinement=dict(mode='none', strength=0., max_shift=0., sigma=32.,
                        decay=128., preserve_visible=False),
        refinement_name='selection_only_no_rgb_or_mask_changes',
        parent_freeze_sha256=parent_sha256,
        confirmation=dict(
            reason='Selection-only chosen after the preceding 166-case secondary results; this is a new independent test.',
            expected_images=128, sources={'VG': 64, 'COCOA': 64},
            previous_images=len(previous), no_refitting=True, no_sequential_stopping=True,
            point_target='All four primary relative gains >=3%; gains above5% also satisfy the target.',
            uncertainty='Report paired source-stratified 95% bootstrap intervals; distinguish point estimates from lower bounds.',
            failures='Keep all128 cases and all1024 candidate records. Preserve existing penalties and explicitly report CLIP coverage.',
            scope='Artificial occlusion; source populations VG and COCOA only; not full progressive improvement.',
            original_five_hour_deadline_missed=True))
    freeze.pop('ensemble', None)
    # All previously inspected images are excluded, even though only183 fitted
    # the ranker. The original development metadata is retained unchanged.
    freeze['confirmation']['excluded_image_identities'] = [r['image_identity'] for r in previous]
    freeze['confirmation']['excluded_image_sha256'] = [r['source_sha256'] for r in previous]
    return freeze


def split_audit(previous, fresh, dataset):
    old = [fingerprint(row, dataset) for row in previous]
    new = [fingerprint(row, dataset) for row in fresh]
    thumbnails = np.stack([r['thumbnail'] for r in old])
    aspects = np.array([r['aspect'] for r in old])
    exact, near, within, nearest = [], [], [], []
    for index, row in enumerate(new):
        errors = np.abs(thumbnails - row['thumbnail']).mean(axis=(1, 2, 3))
        errors[np.abs(aspects / row['aspect'] - 1) > .01] = np.inf
        nearest.append(float(errors.min()) if np.isfinite(errors).any() else None)
        exact += [(row['identity'], r['identity']) for r in old
                  if row['decoded_sha256'] == r['decoded_sha256']]
        near += [(row['identity'], old[i]['identity'], float(errors[i]))
                 for i in np.flatnonzero(errors <= .005)]
        within += [(row['identity'], r['identity']) for r in new[:index]
                   if row['decoded_sha256'] == r['decoded_sha256']]
    result = dict(previous_images=len(old), fresh_images=len(new), exact_cross=exact,
        near_cross=near, exact_within=within,
        minimum_aligned_thumbnail_mae=min(v for v in nearest if v is not None),
        near_rule='Aspect within1%; 64x64 BOX RGB MAE <=0.005',
        limitation='Does not rule out transformed or cropped duplicates.',
        passed=not (exact or near or within))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/open_world_amodal'))
    parser.add_argument('--parent-freeze', type=Path, default=Path(
        'output/surface-followup-20261110-design/final-method-freeze.json'))
    args = parser.parse_args()
    design = args.output_dir.parent / (args.output_dir.name + '-design')
    if args.output_dir.exists() or design.exists():
        raise ValueError('Use new immutable experiment and design directories')
    parent = json.loads(args.parent_freeze.read_text())
    verify_frozen_surface_sources(parent)
    previous_paths = [ROOT / 'output' / name / 'cases.json' for name in PREVIOUS]
    previous = [row for path in previous_paths for row in json.loads(path.read_text())]
    assert len(previous) == 349
    seen = {r['image_identity'] for r in previous}
    hashes = {r['source_sha256'] for r in previous}
    counts = Counter()
    rows, _ = read_official(args.dataset_root)
    for row in rows:
        identity = row['source'] + '/' + row['original_id']
        path = args.reference_run / 'visible_references/masks' / row['reference']
        if identity in seen or not path.is_file():
            continue
        with Image.open(path) as image:
            mask = np.array(image.convert('L')) > 0
        if mask.sum() < 4096:
            continue
        rr, cc = np.where(mask)
        if min(np.ptp(rr) + 1, np.ptp(cc) + 1) < 80:
            continue
        digest = sha256(args.dataset_root / row['image'])
        if digest in hashes:
            continue
        seen.add(identity)
        hashes.add(digest)
        counts[row['source']] += 1
    design.mkdir(parents=True)
    write_json(design / 'eligibility.json', dict(previous=len(previous), eligible=counts,
        total=sum(counts.values()), basis='Native support and identity; final resized support checked during preparation'))
    write_json(design / 'excluded_cases.json', previous)
    config = json.loads((ROOT / 'configs/research/surface-consistency-v1.json').read_text())
    config.update(experiment=args.output_dir.name, selection_seed=20261110128,
        development_per_source=0, cases_per_source=64,
        source_case_limits={'VG': 64, 'COCOA': 64, 'free_images': 0, 'laion': 0},
        candidate_seeds=parent['candidate_seeds'],
        primary_control='expected_uniform_selection_from_same_eight_candidates',
        normalization='Unchanged parent ranker; no fitting on this cohort',
        limitations=[*config['limitations'], 'Selection-only independently confirmed after selection using prior166 outcomes.'])
    write_json(design / 'config.json', config)
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_surface_experiment.py'),
        '--dataset-root', str(args.dataset_root), '--reference-run', str(args.reference_run),
        '--output-dir', str(args.output_dir), '--config', str(design / 'config.json'),
        '--exclude-cases', str(design / 'excluded_cases.json')], check=True)
    cases = json.loads((args.output_dir / 'cases.json').read_text())
    audit = split_audit(previous, cases, args.dataset_root)
    write_json(design / 'split-audit.json', audit)
    if not audit['passed']:
        raise ValueError('Split duplicate audit failed; do not generate')
    freeze = confirmation_freeze(parent, previous, sha256(args.parent_freeze))
    freeze['source_hashes'].update({name: sha256(ROOT / name) for name in NEW_SOURCES})
    freeze['confirmation']['input_sha256'] = {
        path.resolve().relative_to(ROOT).as_posix(): sha256(path) for path in [*previous_paths, args.output_dir / 'cases.json',
        args.output_dir / 'generation_manifest.json', args.output_dir / 'protocol.json',
        design / 'excluded_cases.json', design / 'split-audit.json']}
    write_json(design / 'selection-only-freeze.json', freeze)
    for name in freeze['source_hashes']:
        dest = design / 'frozen-source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    print(json.dumps(dict(eligible=counts, images=len(cases), split_audit=audit,
        freeze_sha256=sha256(design / 'selection-only-freeze.json')), indent=2))


if __name__ == '__main__':
    main()
