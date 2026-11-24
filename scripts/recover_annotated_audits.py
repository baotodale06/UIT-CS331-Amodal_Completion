#!/usr/bin/env python3
"""Preserve completed independent cluster audits separately from local replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('cohort', 'pool'), required=True)
    args = parser.parse_args()
    design = ROOT/'output/surface-annotated-20261112-design'
    submission = json.loads((design/'remote-audit-submission.json').read_text())
    job = next(row['job_id'] for row in submission['jobs'] if row['phase'] == args.phase)
    active = json.loads((design/'active-study.json').read_text())
    names = ['audit.json','paired_records.json','decision_replay.json'] if args.phase == 'cohort' else ['pool-audit.json']
    program = '''
from pathlib import Path
import hashlib,json,subprocess
root=Path.home()/'Vu_Hung/cv_nangcao'
state=subprocess.check_output(['sacct','-j',JOB,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
rows=[row.strip().split('|') for row in state.splitlines()]
if not any(row[:3]==[JOB,'COMPLETED','0:0'] for row in rows):
    raise ValueError('Independent audit job is not successfully complete: '+state)
files={}
for name in NAMES:
    data=(root/STUDY/'report'/name).read_bytes()
    files[name]=dict(text=data.decode('utf-8'),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(dict(job_id=JOB,slurm=state,files=files)))
'''.replace('JOB', repr(job)).replace('NAMES', repr(names)).replace('STUDY', repr(active['study']))
    response = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'daoqm@slurm.csse.rose-hulman.edu','python3 -'], input=program,text=True,capture_output=True,check=True)
    record = json.loads(response.stdout)
    output = ROOT/'output/surface-annotated-20261112-recovery/cluster-audits'/args.phase
    output.mkdir(parents=True,exist_ok=True)
    for name, artifact in record['files'].items():
        data = artifact.pop('text').encode('utf-8')
        if hashlib.sha256(data).hexdigest() != artifact['sha256']:
            raise ValueError('Audit checksum mismatch: '+name)
        path = output/name
        if path.exists() and sha256(path) != artifact['sha256']:
            raise ValueError('Existing audit evidence differs: '+name)
        path.write_bytes(data)
    audit = json.loads((output/('audit.json' if args.phase == 'cohort' else 'pool-audit.json')).read_text())
    if not audit['passed'] or audit['targets'] != 204:
        raise ValueError('Independent audit did not pass on the whole frozen cohort')
    record.update(verified=True,source_sha256=sha256(Path(__file__)))
    write_json(output/'recovery-receipt.json',record)
    print(json.dumps(dict(verified=True,job_id=job,files=len(names))))


if __name__ == '__main__':
    main()
