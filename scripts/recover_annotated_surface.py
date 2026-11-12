#!/usr/bin/env python3
"""Recover a completed export and verify its independent archive/member hashes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from collect_surface_job_evidence import recover_archive


def download(relative, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as stream:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=30', 'daoqm@slurm.csse.rose-hulman.edu',
            'cat ~/Vu_Hung/cv_nangcao/' + relative], stdout=stream, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('cohort', 'pool', 'full'), required=True)
    p.add_argument('--metadata-only', action='store_true')
    a = p.parse_args()
    tag = a.phase + ('-metadata' if a.metadata_only else '-native')
    destination = ROOT / 'output/surface-annotated-20261112-recovery' / tag
    remote = 'output/surface-annotated-20261112-design/'
    receipt_path = destination / 'remote-receipt.json'
    download(remote + tag + '-receipt.json', receipt_path)
    receipt = json.loads(receipt_path.read_text())
    manifest_path = destination / 'remote-manifest.json'
    download(receipt['manifest'], manifest_path)
    if sha256(manifest_path) != receipt['manifest_sha256']:
        raise ValueError('Remote manifest checksum mismatch')
    archive = destination / 'evidence.tar.gz'
    if not archive.exists():
        partial = destination / 'evidence.tar.gz.partial'
        download(receipt['archive'], partial)
        if sha256(partial) != receipt['archive_sha256'] or partial.stat().st_size != receipt['bytes']:
            raise ValueError('Downloaded archive checksum mismatch')
        partial.replace(archive)
    if sha256(archive) != receipt['archive_sha256']:
        raise ValueError('Existing archive differs from remote')
    members = recover_archive(archive, ROOT, ['output/surface-annotated-20261112'],
                              json.loads(manifest_path.read_text()))
    write_json(destination / 'recovery-receipt.json', dict(verified=True,
        archive_sha256=receipt['archive_sha256'], member_sha256=members, members=len(members)))
    print(json.dumps(dict(verified=True, members=len(members), bytes=receipt['bytes'])))


if __name__ == '__main__':
    main()
