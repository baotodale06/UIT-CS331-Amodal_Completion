#!/usr/bin/env python3
"""Package completed research artifacts with fresh independent transport hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-perceptual', action='store_true')
    p.add_argument('--job-id', type=int, action='append', default=[], help='Include logs only for these completed jobs')
    args = p.parse_args()
    root = args.experiment_dir.resolve()
    cases = json.loads((root / 'generation_manifest.json').read_text())
    seeds = json.loads((root / 'protocol.json').read_text())['candidate_seeds']
    for case in cases:
        for seed in seeds:
            if not (root / 'full/candidates' / case['case_id'] / f'seed-{seed}.json').exists():
                raise ValueError('Generation incomplete: ' + case['case_id'])
        if args.require_perceptual and not (root / 'perceptual' / (case['case_id'] + '.json')).exists():
            raise ValueError('Perceptual evaluation incomplete: ' + case['case_id'])
    for index in range(4):
        for directory in ('full', 'perceptual') if args.require_perceptual else ('full',):
            metadata = json.loads((root / directory / f'runtime-worker-{index}.json').read_text())
            if not metadata.get('finished_at'):
                raise ValueError('A worker is still active')
    folders = ['full', 'perceptual', 'analysis']
    files = [path for folder in folders if (root / folder).exists() for path in (root / folder).rglob('*') if path.is_file()]
    if (root / 'job-logs').exists():
        files.extend(path for path in (root / 'job-logs').iterdir()
                     if path.is_file() and any(path.name.startswith(str(job) + '-') for job in args.job_id))
    if (root / 'runtime-model-identity.json').exists():
        files.append(root / 'runtime-model-identity.json')
        if (root / 'job-logs/model-identity.log').exists():
            files.append(root / 'job-logs/model-identity.log')
    records = [dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path))
               for path in sorted(files)]
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), case_count=len(cases),
                    candidate_count=len(cases) * len(seeds), files=records,
                    source_sha256=sha256(Path(__file__)), total_bytes=sum(r['bytes'] for r in records))
    manifest_path = root / 'transport_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with tarfile.open(args.output, 'w:gz', compresslevel=1) as archive:
        for path in sorted(files) + [manifest_path]:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    receipt = dict(archive=str(args.output), bytes=args.output.stat().st_size, sha256=sha256(args.output),
                   manifest_sha256=sha256(manifest_path), files=len(records))
    args.output.with_suffix(args.output.suffix + '.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
