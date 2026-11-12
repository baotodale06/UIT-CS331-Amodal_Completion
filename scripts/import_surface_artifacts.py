#!/usr/bin/env python3
"""Recover a completed research export and verify its independent remote hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    receipt = json.loads(args.receipt.read_text())
    if args.archive.stat().st_size != receipt['bytes'] or sha256(args.archive) != receipt['sha256']:
        raise ValueError('Transport archive differs from the independent remote receipt')
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        manifest_bytes = archive.extractfile('transport_manifest.json').read()
        if hashlib.sha256(manifest_bytes).hexdigest() != receipt['manifest_sha256']:
            raise ValueError('Transport manifest changed')
        manifest = json.loads(manifest_bytes)
        expected = {row['path']: row for row in manifest['files']}
        expected['transport_manifest.json'] = dict(bytes=len(manifest_bytes), sha256=receipt['manifest_sha256'])
        if len(members) != len(expected) or {m.name for m in members} != expected.keys():
            raise ValueError('Archive membership differs from its manifest')
        for member in members:
            destination = (root / member.name).resolve()
            if not member.isfile() or not destination.is_relative_to(root) or member.size != expected[member.name]['bytes']:
                raise ValueError('Unexpected archive member: ' + member.name)
            if destination.exists() and sha256(destination) != expected[member.name]['sha256']:
                raise ValueError('Refuse to replace a different existing artifact: ' + str(destination))
        for member in members:
            destination = root / member.name
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                with archive.extractfile(member) as source:
                    shutil.copyfileobj(source, temporary)
                temporary_path = Path(temporary.name)
            if sha256(temporary_path) != expected[member.name]['sha256']:
                raise ValueError('Extracted content differs: ' + member.name)
            temporary_path.replace(destination)
    for name, record in expected.items():
        if sha256(root / name) != record['sha256']:
            raise ValueError('Recovered file changed: ' + name)
    result = dict(status='passed', verified_at=datetime.now(timezone.utc).isoformat(),
        archive_sha256=receipt['sha256'], manifest_sha256=receipt['manifest_sha256'],
        files=len(manifest['files']), bytes=manifest['total_bytes'],
        source_sha256=sha256(Path(__file__)), remote_manifest_created_at=manifest['created_at'])
    write_json(root / 'transport_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
