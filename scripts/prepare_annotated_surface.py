#!/usr/bin/env python3
"""Freeze an outcome-independent annotated-vocabulary progressive experiment."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from prepare_surface_confirmation import split_audit

PREVIOUS = ('surface-consistency-20261108', 'surface-consistency-replication87-20261108',
            'surface-followup-20261110', 'surface-confirmation128-20261110')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'datasets/open_world_amodal')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--images', type=int, default=192)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new immutable design directory')
    previous_paths = [ROOT / 'output' / name / 'cases.json' for name in PREVIOUS]
    previous = [case for path in previous_paths for case in json.loads(path.read_text())]
    previous_ids = {case['image_identity'] for case in previous}
    previous_hashes = {case['source_sha256'] for case in previous}
    official, counts = read_official(args.dataset_root)
    groups = defaultdict(list)
    source_hashes = {}
    for row in official:
        identity = row['source'] + '/' + row['original_id']
        if row['image'] not in source_hashes:
            source_hashes[row['image']] = sha256(args.dataset_root / row['image'])
        if identity in previous_ids or source_hashes[row['image']] in previous_hashes:
            continue
        groups[identity].append(row)
    selection_seed = 20261112
    duplicate_screen = []
    while True:
        eligible = Counter(rows[0]['source'] for rows in groups.values())
        if sum(eligible.values()) < args.images:
            raise ValueError('Insufficient independent images')
        exact = {source: args.images * count / sum(eligible.values()) for source, count in eligible.items()}
        quotas = {source: int(value) for source, value in exact.items()}
        for source in sorted(exact, key=lambda s: (-(exact[s] - quotas[s]), s))[:args.images - sum(quotas.values())]:
            quotas[source] += 1
        selected_images = []
        for source, quota in quotas.items():
            identities = [identity for identity, rows in groups.items() if rows[0]['source'] == source]
            identities.sort(key=lambda value: hashlib.sha256(f'{selection_seed}:{value}'.encode()).hexdigest())
            selected_images.extend(identities[:quota])
        cases = [dict(case_id=identity, image_identity=identity, source=groups[identity][0]['source'],
                      source_image=groups[identity][0]['image'],
                      source_sha256=source_hashes[groups[identity][0]['image']]) for identity in selected_images]
        audit = split_audit(previous, cases, args.dataset_root)
        duplicate_screen.append(audit)
        if audit['passed']:
            break
        rejected = {pair[0] for kind in ('exact_cross', 'near_cross', 'exact_within') for pair in audit[kind]}
        for identity in rejected:
            del groups[identity]
    chosen = set(selected_images)
    rows = [row for row in official if row['source'] + '/' + row['original_id'] in chosen]
    parent = ROOT / 'output/surface-followup-20261110-design/final-method-freeze.json'
    args.output_dir.mkdir(parents=True)
    method = json.loads(parent.read_text())
    method.update(created_at=datetime.now(timezone.utc).isoformat(),
        parent_freeze_sha256=sha256(parent),
        evaluation_scope='annotated-vocabulary natural-occlusion progressive evaluation',
        primary_control='annotated_k1', endpoints=['LPIPS', 'CLIP', 'Feature Similarity', 'SSIM'],
        annotated_transfer=dict(no_refitting=True, no_diffusion_training=True,
            changed_inference_factor='detector vocabulary annotated in both comparison arms',
            adapter_repair='Capture target-only candidate segmenter before per-image scene wrapper installation',
            invariant='Parent ranker coefficients, seeds, correction and stopping rules are unchanged'))
    method.pop('desired_relative_improvement_percent', None)
    for relative in ('amodal_benchmark/surface_runtime.py', 'evaluate_open_world.py'):
        method['source_hashes'][relative] = sha256(ROOT / relative)
    write_json(args.output_dir / 'method-freeze.json', method)
    manifest = dict(schema=1, created_at=datetime.now(timezone.utc).isoformat(),
        selection_seed=selection_seed, selection='source-stratified hash sample of images; retain every associated target',
        images=len(chosen), targets=len(rows), source_image_counts=quotas,
        source_target_counts=dict(Counter(row['source'] for row in rows)),
        eligible_image_counts=dict(eligible), target_ids=[row['id'] for row in rows], cases=cases,
        previous_images=len(previous_ids), previous_manifest_hashes={str(path.relative_to(ROOT)): sha256(path) for path in previous_paths},
        exclusions='Exclude prior controlled-study identities/source bytes; reject decoded or near-aligned duplicates and refill by the same hash rule',
        outcome_access=False, image_audit=audit)
    write_json(args.output_dir / 'duplicate-screen-history.json', duplicate_screen)
    write_json(args.output_dir / 'cohort.json', manifest)
    write_json(args.output_dir / 'unseen-inventory.json', dict(schema=1,
        target_ids=[row['id'] for row in official if row['source'] + '/' + row['original_id'] in groups],
        images=len(groups), exclusions=manifest['exclusions']))
    # Smoke is deterministic, independently named, and not used to choose parameters.
    write_json(args.output_dir / 'smoke.json', dict(schema=1, target_ids=[row['id'] for row in rows[:8]]))
    write_json(args.output_dir / 'previous-images.json', dict(
        image_identities=sorted(previous_ids), image_sha256=sorted(previous_hashes)))
    print(json.dumps({key: manifest[key] for key in ('images', 'targets', 'source_image_counts', 'source_target_counts', 'previous_images')}, indent=2))


if __name__ == '__main__':
    main()
