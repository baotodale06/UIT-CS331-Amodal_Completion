#!/usr/bin/env python3
"""Recover a completed export and verify its independent archive/member hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
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


def download_parallel(remote_path, destination, size, expected_sha256, workers=8):
    """Read disjoint immutable ranges, then verify the complete independent hash."""
    if size < 1 or workers < 1:
        raise ValueError('Positive size and worker count required')
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    chunk_root=destination.parent/(destination.name+'.chunks')
    chunk_root.mkdir(exist_ok=True)
    contract=dict(remote_path=remote_path,size=size,sha256=expected_sha256,workers=workers)
    contract_path=chunk_root/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Existing download parts belong to a different archive')
    write_json(contract_path,contract)
    block=1024*1024
    blocks=(size+block-1)//block
    span=(blocks+workers-1)//workers
    ranges=[(index,start,min(span,blocks-start)) for index,start in enumerate(range(0,blocks,span))]
    def fetch(item):
        index,start,count=item
        path=chunk_root/f'part-{index:02d}'
        expected=min(count*block,size-start*block)
        if not path.exists() or path.stat().st_size != expected:
            temporary=path.with_suffix('.partial')
            command='dd if='+shlex.quote(remote_path)+f' bs={block} skip={start} count={count} status=none'
            with temporary.open('wb') as stream:
                subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                    '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',
                    'daoqm@slurm.csse.rose-hulman.edu',command],stdout=stream,check=True)
            if temporary.stat().st_size != expected:
                raise ValueError('Remote archive range has an unexpected size')
            temporary.replace(path)
        return dict(index=index,start_byte=start*block,bytes=expected,path=str(path),sha256=sha256(path))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts=list(executor.map(fetch,ranges))
    with destination.open('wb') as stream:
        for part in sorted(parts,key=lambda row:row['index']):
            with Path(part['path']).open('rb') as source:
                shutil.copyfileobj(source,stream,length=1024*1024)
    if destination.stat().st_size != size or sha256(destination) != expected_sha256:
        raise ValueError('Reassembled archive differs from its independent remote hash')
    write_json(chunk_root/'verification.json',dict(verified=True,contract=contract,parts=parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('cohort', 'pool', 'full'), required=True)
    p.add_argument('--metadata-only', action='store_true')
    a = p.parse_args()
    design = ROOT / 'output/surface-annotated-20261112-design'
    active = json.loads((design / 'active-study.json').read_text())
    study = Path(active['study'])
    tag = study.name + '-' + a.phase + ('-metadata' if a.metadata_only else '-native')
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
        if receipt['bytes'] >= 4*1024*1024:
            download_parallel('/home/daoqm/Vu_Hung/cv_nangcao/'+receipt['archive'],partial,
                              receipt['bytes'],receipt['archive_sha256'])
        else:
            download(receipt['archive'], partial)
        if sha256(partial) != receipt['archive_sha256'] or partial.stat().st_size != receipt['bytes']:
            raise ValueError('Downloaded archive checksum mismatch')
        partial.replace(archive)
    if sha256(archive) != receipt['archive_sha256']:
        raise ValueError('Existing archive differs from remote')
    members = recover_archive(archive, ROOT, [study.as_posix()],
                              json.loads(manifest_path.read_text()))
    write_json(destination / 'recovery-receipt.json', dict(verified=True,
        archive_sha256=receipt['archive_sha256'], member_sha256=members, members=len(members),
        recovery_source_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(verified=True, members=len(members), bytes=receipt['bytes'])))


if __name__ == '__main__':
    main()
