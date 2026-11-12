#!/usr/bin/env python3
"""Preserve a verified cohort before the same-contract full evaluation resumes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import read_official, sha256, write_json
from amodal_benchmark.cohort import select_manifest_rows
from amodal_benchmark.references import reference_paths
from amodal_benchmark.verify import verify_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'output-dir', 'dataset-root', 'target-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    report = verify_run(a.run_dir, a.dataset_root, require_provenance=True)
    rows, _ = read_official(a.dataset_root)
    rows = select_manifest_rows(rows, a.target_manifest)
    execution = json.loads((a.run_dir / 'execution.json').read_text())
    if execution['target_ids'] != [row['id'] for row in rows]:
        raise ValueError('Snapshot inventory does not equal the frozen cohort')
    if a.output_dir.exists():
        raise ValueError('Snapshot already exists; do not replace immutable evidence')
    a.output_dir.mkdir(parents=True)
    for path in a.run_dir.iterdir():
        if path.is_file() and path.name != '.pdmc.lock':
            shutil.copy2(path, a.output_dir / path.name)
    for name in ('provenance', 'workers', 'cohort_manifests'):
        if (a.run_dir / name).exists():
            shutil.copytree(a.run_dir / name, a.output_dir / name)
    references = a.output_dir / 'visible_references'
    references.mkdir()
    shutil.copy2(a.run_dir / 'visible_references/reference_config.json', references / 'reference_config.json')
    for row in rows:
        target = a.output_dir / 'targets' / row['id']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(a.run_dir / 'targets' / row['id'], target)
        for path in reference_paths(a.run_dir / 'visible_references', row).values():
            destination = a.output_dir / path.relative_to(a.run_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    verification = verify_run(a.output_dir, a.dataset_root, require_provenance=True)
    write_json(a.output_dir / 'snapshot-receipt.json', dict(source=str(a.run_dir),
        cohort_sha256=sha256(a.target_manifest), source_execution_sha256=sha256(a.run_dir / 'execution.json'),
        verified=verification['verified'], targets=report['selected_targets']))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
