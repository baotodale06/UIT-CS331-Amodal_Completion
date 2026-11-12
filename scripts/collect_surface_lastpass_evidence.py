#!/usr/bin/env python3
"""Wait for the diagnostic jobs and recover checksum-verified numerical evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tarfile
import time

from collect_surface_job_evidence import (HOST, PROJECT, TERMINAL, digest,
    recover_archive, remote_json, ssh, successful, write_json)

RUN = 'output/surface-lastpass-ablation-20261111'
DESIGN = RUN+'-design'
FREEZE = '4866d3f2a1a26e207b45a101ef8e3c0316cc18af828188b1edf017229a7210c8'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-job', type=int, required=True)
    parser.add_argument('--finalizer-job', type=int, required=True)
    parser.add_argument('--recovery-dir', type=Path, default=Path('output/surface-lastpass-ablation-20261111-recovery'))
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if min(args.measurement_job,args.finalizer_job)<=0:
        parser.error('Positive Slurm job IDs required')
    ids=','.join(map(str,(args.measurement_job,args.finalizer_job)))
    status_program = 'import subprocess,json\nfrom pathlib import Path\n'
    status_program += f'root=Path.home()/{PROJECT!r}\n'
    status_program += f'raw=subprocess.check_output(["sacct","-j",{ids!r},"-X","-n","-P","--format=JobID,State,ExitCode,Elapsed"],text=True)\n'
    status_program += 'jobs={}\nfor line in raw.splitlines():\n f=line.split("|")\n if len(f)>=4:jobs[f[0]]=dict(state=f[1].split()[0].rstrip("+"),exit_code=f[2],elapsed=f[3])\n'
    status_program += f'print(json.dumps(dict(jobs=jobs,records=sum(1 for p in (root/{RUN!r}).glob("targets/*/*/ablation.json")))))\n'
    recovery=args.recovery_dir.resolve()
    recovery.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            state=remote_json(HOST,status_program)
            state['as_of_utc']=datetime.now(timezone.utc).isoformat()
            state['complete']=False
            if successful(state['jobs'],str(args.finalizer_job)):
                program='import hashlib,json\nfrom pathlib import Path\n'+f'root=Path.home()/{PROJECT!r}\n'
                program+=f'p=root/{(DESIGN+"/evidence.tar.gz")!r}\n'
                program+=f'm=json.loads((root/{(DESIGN+"/evidence-manifest.json")!r}).read_text())\n'
                program+='print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,member_sha256=m)))\n'
                expected=remote_json(HOST,program)
                archive=recovery/'evidence.tar.gz'
                if not archive.exists():
                    partial=recovery/'evidence.tar.gz.partial'
                    with partial.open('wb') as stream:
                        ssh(HOST,'cat ~/'+PROJECT+'/'+DESIGN+'/evidence.tar.gz',stdout=stream,stderr=subprocess.PIPE)
                    if digest(partial)!=expected['sha256']:
                        raise ValueError('Diagnostic archive checksum differs')
                    partial.replace(archive)
                if digest(archive)!=expected['sha256']:
                    raise ValueError('Existing diagnostic archive differs')
                members=recover_archive(archive,recovery/'evidence',[RUN,DESIGN],expected['member_sha256'])
                summary=json.loads((recovery/'evidence'/RUN/'analysis/summary.json').read_text())
                if summary['freeze_sha256']!=FREEZE or summary['targets']!=2565 or summary['derived_images']!=958:
                    raise ValueError('Incomplete or changed diagnostic experiment')
                write_json(recovery/'receipt.json',dict(**expected,recovered_at=state['as_of_utc'],
                    checked_members=len(members),result=summary,native_pngs_recovered=False))
                state['complete']=True
            write_json(recovery/'status.json',state)
            print(json.dumps(state),flush=True)
            if state['complete']:
                return
            failed={job:row for job,row in state['jobs'].items()
                    if row['state'] in TERMINAL and not successful(state['jobs'],job)}
            if failed:
                raise SystemExit('Diagnostic job failed; retained for review: '+json.dumps(failed))
        except (OSError,ValueError,subprocess.SubprocessError,tarfile.TarError) as error:
            record=dict(as_of_utc=datetime.now(timezone.utc).isoformat(),error=str(error),remote_jobs_changed=False)
            write_json(recovery/'last-error.json',record)
            print(json.dumps(record),flush=True)
            if args.once or isinstance(error,(ValueError,tarfile.TarError)):
                raise
        if args.once:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
