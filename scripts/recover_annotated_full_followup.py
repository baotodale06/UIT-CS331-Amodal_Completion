#!/usr/bin/env python3
"""Recover the completed annotated study's registry, job accounting and receipts."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json

REMOTE = r'''
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, subprocess
root=Path.home()/'Vu_Hung/cv_nangcao'
design=root/'output/surface-annotated-20261112-design'
names=['freshrefs-experiments.csv', 'experiment_report.md',
       'freshrefs-baseline-full-verification.json',
       'freshrefs-surface-full-verification.json',
       'freshrefs-full-reference-copy-39490.json',
       'freshrefs-full-reference-copy-39491.json']
paths=[design/name for name in names]
paths += sorted(design.glob('ann-*-394*.log'))
files={}
for path in paths:
    data=path.read_bytes()
    files[path.relative_to(root).as_posix()]=dict(
        sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
        base64=base64.b64encode(data).decode())
command=['sacct','-j',','.join(str(i) for i in range(39486,39500)),
         '-X','-n','-P','--format=JobIDRaw,JobName%60,Partition,State,ElapsedRaw,Start,End,ExitCode,AllocTRES%120,NodeList']
accounting=subprocess.check_output(command,text=True)
print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),
    files=files, accounting=accounting, accounting_command=command)))
'''


def main():
    destination = ROOT/'output/surface-annotated-full-review-20261114/remote-evidence'
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ServerAliveInterval=30', 'daoqm@slurm.csse.rose-hulman.edu', 'python3 -']
    response = subprocess.run(command, input=REMOTE, text=True, capture_output=True,
                              timeout=120, check=True)
    payload = json.loads(response.stdout)
    manifest = {}
    prefix = 'output/surface-annotated-20261112-design/'
    for relative, record in payload.pop('files').items():
        if not relative.startswith(prefix) or Path(relative).name != relative[len(prefix):]:
            raise ValueError('Unexpected remote artifact path')
        data = base64.b64decode(record.pop('base64'), validate=True)
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Remote artifact checksum mismatch: ' + relative)
        path = destination/relative
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Preserve differing recovered evidence: ' + relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        manifest[relative] = record
    write_json(destination/'receipt.json', dict(verified=True,
        recovered_at=datetime.now(timezone.utc).isoformat(), members=manifest,
        source_sha256=sha256(Path(__file__)), ssh_command=command, remote_program=REMOTE,
        **payload))
    print(json.dumps(dict(verified=True, files=len(manifest),
                         bytes=sum(row['bytes'] for row in manifest.values()))))


if __name__ == '__main__':
    main()
