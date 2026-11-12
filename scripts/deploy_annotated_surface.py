#!/usr/bin/env python3
"""Deploy this study's source-only archive with backup and per-file hash checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    root = Path.cwd().resolve()
    archive = root / 'output/surface-annotated-20261112-upload.tar.gz'
    backup = root / 'output' / ('source-before-annotated-surface-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir()
    records = []
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            destination = (root / member.name).resolve()
            if not member.isfile() or not destination.is_relative_to(root) or destination.name == 'AGENTS.md':
                raise ValueError('Unsafe archive member: ' + member.name)
            data = stream.extractfile(member).read()
            record = dict(path=member.name, new_sha256=hashlib.sha256(data).hexdigest())
            if destination.exists():
                record['previous_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
                saved = backup / member.name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, saved)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == record['new_sha256']
            records.append(record)
    receipt = dict(deployed_at=datetime.now(timezone.utc).isoformat(), backup=str(backup), files=records,
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (root / 'output/surface-annotated-20261112-design/deployment-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(files=len(records), backup=str(backup), archive_sha256=receipt['archive_sha256'])))


if __name__ == '__main__':
    main()
