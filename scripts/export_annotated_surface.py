#!/usr/bin/env python3
"""Archive completed annotated study evidence, with independent native file hashes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('cohort', 'pool', 'full'), required=True)
    p.add_argument('--metadata-only', action='store_true')
    a = p.parse_args()
    design = ROOT / 'output/surface-annotated-20261112-design'
    study = ROOT / 'output/surface-annotated-20261112'
    if a.phase == 'cohort':
        folders = [study / name for name in ('baseline-cohort', 'surface-cohort', 'cohort-comparison')]
        gate = study / 'cohort-comparison/comparison.json'
    elif a.phase == 'pool':
        folders = [study / 'pool-diagnostic']
        gate = study / 'pool-diagnostic/analysis/summary.json'
    else:
        folders = [study / name for name in ('baseline', 'surface', 'full-comparison')]
        gate = study / 'full-comparison/comparison.json'
    if not gate.is_file():
        raise ValueError('Requested evidence is not complete')
    tag = a.phase + ('-metadata' if a.metadata_only else '-native')
    archive = design / (tag + '-evidence.tar.gz')
    receipt_path = design / (tag + '-receipt.json')
    manifest_path = design / (tag + '-manifest.json')
    if archive.exists() or receipt_path.exists() or manifest_path.exists():
        raise ValueError('Preserve existing evidence export')
    paths = []
    for folder in folders:
        for path in folder.rglob('*'):
            if not path.is_file() or path.name == '.pdmc.lock':
                continue
            if a.metadata_only and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                continue
            relative = path.relative_to(folder)
            if not a.metadata_only and relative.parts[0] == 'targets' and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                final = len(relative.parts) == 4 and path.name in ('completion.png', 'mask.png')
                pool = len(relative.parts) >= 6 and relative.parts[3] == 'surface' and path.suffix.lower() == '.png'
                if not (final or pool):
                    continue
            paths.append(path)
    paths = sorted(set(paths))
    # Independent native hashes are computed before packaging, never copied from
    # inference records. The archive then receives its own integrity hash.
    manifest = {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}
    with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
        for path in paths:
            bundle.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    write_json(manifest_path, manifest)
    receipt = dict(created_at=datetime.now(timezone.utc).isoformat(), phase=a.phase,
        metadata_only=a.metadata_only, members=len(paths), bytes=archive.stat().st_size,
        native_scope='Final completion/mask, all references and all Surface-PMC pool PNGs; upstream debug rasters remain on server',
        archive_sha256=sha256(archive), manifest_sha256=sha256(manifest_path),
        completion_gate_sha256=sha256(gate), archive=archive.relative_to(ROOT).as_posix(),
        manifest=manifest_path.relative_to(ROOT).as_posix(), source_sha256=sha256(Path(__file__)))
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
