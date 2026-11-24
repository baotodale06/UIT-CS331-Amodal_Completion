#!/usr/bin/env python3
"""Recover completed study exports; never inspect unfinished quality outcomes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'output/surface-annotated-20261112-design'
ACTIVE = json.loads((DESIGN/'active-study.json').read_text())
STUDY = Path(ACTIVE['study'])
DESTINATION = ROOT/'output/surface-annotated-20261112-recovery'
SCOPES = [('cohort',True),('cohort',False),('pool',True),('full',True)]


def main():
    DESTINATION.mkdir(exist_ok=True)
    state = dict(started_at=datetime.now(timezone.utc).isoformat(),study=STUDY.as_posix(),completed=[],errors=[])
    while len(state['completed']) < len(SCOPES):
        for phase, metadata in SCOPES:
            tag = STUDY.name+'-'+phase+('-metadata' if metadata else '-native')
            receipt = DESTINATION/tag/'recovery-receipt.json'
            if tag in state['completed']:
                continue
            if receipt.is_file() and json.loads(receipt.read_text()).get('verified'):
                state['completed'].append(tag)
                continue
            remote = 'test -f ~/Vu_Hung/cv_nangcao/output/surface-annotated-20261112-design/'+tag+'-receipt.json'
            try:
                result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                    'daoqm@slurm.csse.rose-hulman.edu',remote],capture_output=True,timeout=30)
                if result.returncode == 1:
                    continue
                if result.returncode:
                    raise RuntimeError(result.stderr.decode(errors='replace'))
                command = [sys.executable,'-u','scripts/recover_annotated_surface.py','--phase',phase]
                if metadata:
                    command.append('--metadata-only')
                print(datetime.now(timezone.utc).isoformat(),'RECOVER',tag,flush=True)
                subprocess.run(command,cwd=ROOT,check=True)
                state['completed'].append(tag)
            except Exception as error:
                state['errors'].append(dict(at=datetime.now(timezone.utc).isoformat(),tag=tag,error=str(error)))
                print('RETRYABLE',tag,error,flush=True)
            finally:
                state['updated_at'] = datetime.now(timezone.utc).isoformat()
                (DESTINATION/'collector-status.json').write_text(json.dumps(state,indent=2)+'\n')
        print(datetime.now(timezone.utc).isoformat(),'Exports recovered',len(state['completed']),'of',len(SCOPES),flush=True)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        (DESTINATION/'collector-status.json').write_text(json.dumps(state,indent=2)+'\n')
        if len(state['completed']) < len(SCOPES):
            time.sleep(45)
    print('All cohort, pool and full exports recovered with checksum verification',flush=True)


if __name__ == '__main__':
    main()
